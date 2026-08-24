from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

from store.errors import ConfigError

# Deleting a range without touching its keys.
#
# Dropping a prefix, a tenant, a day of logs, is one operation to the caller and a million to
# a store that only has point deletes. The range tombstone fixes the write side: one record
# says everything from a to b is gone, and the read side inherits the cost, because now every
# read has to check the point answer against every range tombstone that might cover the key.
#
# The structure that keeps that check cheap is an interval set kept disjoint and sorted, so
# coverage is one binary search rather than a walk of every tombstone ever written. The
# measurements compare the two write costs, then show the read side cost that point deletes
# never had, which is the shape of every deferral in this package: the work does not vanish,
# it moves to the other path.


@dataclass(frozen=True)
class Span:
    """One range tombstone: everything from start up to but not including stop."""

    start: bytes
    stop: bytes
    sequence: int

    def __post_init__(self) -> None:
        if self.start >= self.stop:
            raise ConfigError(f"{self.start!r} to {self.stop!r} is not a range")

    def covers(self, key: bytes) -> bool:
        """Whether the key falls inside."""
        return self.start <= key < self.stop


@dataclass
class Spans:
    """A sorted, disjoint set of range tombstones with binary search coverage."""

    held: list[Span] = field(default_factory=list)
    checks: int = field(default=0)

    def add(self, start: bytes, stop: bytes, sequence: int) -> None:
        """Install a range, merging whatever it overlaps."""
        fresh_start, fresh_stop = start, stop
        kept: list[Span] = []
        top = sequence
        for span in self.held:
            if span.stop < fresh_start or span.start > fresh_stop:
                kept.append(span)
                continue
            fresh_start = min(fresh_start, span.start)
            fresh_stop = max(fresh_stop, span.stop)
            top = max(top, span.sequence)
        kept.append(Span(start=fresh_start, stop=fresh_stop, sequence=top))
        kept.sort(key=lambda span: span.start)
        self.held = kept

    def covering(self, key: bytes) -> Span | None:
        """The span covering a key, found by binary search, or nothing."""
        self.checks += 1
        low, high = 0, len(self.held) - 1
        while low <= high:
            middle = (low + high) // 2
            span = self.held[middle]
            if span.covers(key):
                return span
            if key < span.start:
                high = middle - 1
            else:
                low = middle + 1
        return None

    def __len__(self) -> int:
        return len(self.held)


@dataclass
class Ranged:
    """A store with both delete shapes, so their costs can be read off one object."""

    entries: dict[bytes, tuple[int, bytes]] = field(default_factory=dict)
    spans: Spans = field(default_factory=Spans)
    sequence: int = field(default=0)
    point_writes: int = field(default=0)
    range_writes: int = field(default=0)

    def put(self, key: bytes, value: bytes) -> None:
        """One write."""
        self.sequence += 1
        self.entries[key] = (self.sequence, value)
        self.point_writes += 1

    def delete(self, key: bytes) -> None:
        """A point delete, one record for one key."""
        self.sequence += 1
        self.entries.pop(key, None)
        self.point_writes += 1

    def delete_range(self, start: bytes, stop: bytes) -> None:
        """A range delete, one record for any number of keys."""
        self.sequence += 1
        self.spans.add(start, stop, self.sequence)
        self.range_writes += 1

    def get(self, key: bytes) -> bytes | None:
        """A read, checked against the spans before it is believed."""
        held = self.entries.get(key)
        if held is None:
            return None
        sequence, value = held
        span = self.spans.covering(key)
        if span is not None and span.sequence > sequence:
            return None
        return value

    def keys(self) -> list[bytes]:
        """Every live key, spans applied."""
        return sorted(key for key in self.entries if self.get(key) is not None)


@functools.cache
def _tenanted(tenants: int = 20, rows: int = 500, seed: int = 37) -> Ranged:
    """A store holding many tenants' rows."""
    source = random.Random(seed)
    store = Ranged()
    for tenant in range(tenants):
        for _ in range(rows):
            store.put(
                f"t{tenant:03d}:{source.randrange(10**6):07d}".encode(),
                source.randbytes(8),
            )
    return store


@functools.cache
def dropping_a_tenant_is_one_write_instead_of_five_hundred() -> bool:
    """The range delete writes one record where point deletes write one per row.

    Same outcome: every read of the tenant's keys answers absent, and other tenants are
    untouched. The write side saving is the row count, which for a real tenant is millions,
    and the saving is exactly why the operation exists: a drop that costs a write per row is
    a drop nobody runs at peak hours.
    """
    ranged = _tenanted(5, 500, 38)
    before = ranged.range_writes
    ranged.delete_range(b"t002:", b"t003:")
    doomed = [key for key in ranged.entries if key.startswith(b"t002:")]
    survivors = [key for key in ranged.entries if key.startswith(b"t001:")]
    return (
        ranged.range_writes == before + 1
        and all(ranged.get(key) is None for key in doomed)
        and all(ranged.get(key) is not None for key in survivors[:50])
    )


@functools.cache
def the_read_side_pays_one_span_check_per_read() -> bool:
    """Every read now consults the span set, hit or miss, covered or not.

    The point delete had no read side cost: the tombstone sits in the same sorted structure
    as the data and the merge hides it for free. The range tombstone is a second structure,
    so every read pays a lookup in it, and the lookup is a binary search only because the
    spans are kept disjoint. This is the deferral shape again: one write bought, a check per
    read sold.
    """
    store = _tenanted(3, 200, 39)
    before = store.spans.checks
    for key in list(store.entries)[:100]:
        store.get(key)
    return store.spans.checks == before + 100


@functools.cache
def overlapping_ranges_merge_into_one_span() -> bool:
    """Twenty overlapping deletes leave one span, so the search stays logarithmic.

    Without merging, the span set grows by one per delete forever and the read check decays
    into a walk. With it, the set's size tracks the geometry of what is deleted rather than
    the history of deleting it, which is the difference between a structure and a log.
    """
    spans = Spans()
    for at in range(20):
        start = f"{at:02d}".encode()
        stop = f"{at + 2:02d}".encode()
        spans.add(start, stop, at + 1)
    return len(spans) == 1


@functools.cache
def a_write_after_the_delete_survives_it() -> bool:
    """A key written after the range delete reads back, inside the dead range.

    Sequence order decides, not geometry: the span hides only what is older than it. Without
    the sequence comparison a tenant could never be re-created, because the old drop would
    swallow every new row forever.
    """
    store = Ranged()
    store.put(b"t001:a", b"old")
    store.delete_range(b"t001:", b"t002:")
    store.put(b"t001:b", b"new")
    return store.get(b"t001:a") is None and store.get(b"t001:b") == b"new"


@functools.cache
def disjoint_ranges_stay_separate() -> bool:
    """Deletes of separate tenants keep separate spans, and the search finds each.

    Merging everything into one span would be wrong, not just untidy: the space between two
    tenants' ranges holds live keys, and a merged span would swallow them.
    """
    store = _tenanted(6, 50, 40)
    store.delete_range(b"t001:", b"t002:")
    store.delete_range(b"t004:", b"t005:")
    alive = [key for key in store.entries if key.startswith(b"t003:")]
    return len(store.spans) == 2 and all(store.get(key) is not None for key in alive[:20])


def compare_the_drop_costs(rows: int = 500) -> list[dict]:
    """One row per delete shape for the same tenant drop."""
    pointwise = _tenanted(3, rows, 41)
    doomed = [key for key in pointwise.entries if key.startswith(b"t001:")]
    before = pointwise.point_writes
    for key in doomed:
        pointwise.delete(key)
    ranged = _tenanted(3, rows, 42)
    ranged.delete_range(b"t001:", b"t002:")
    return [
        {"shape": "point", "writes": pointwise.point_writes - before, "rows": len(doomed)},
        {"shape": "range", "writes": 1, "rows": len(doomed)},
    ]


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "one_write_per_drop": dropping_a_tenant_is_one_write_instead_of_five_hundred(),
        "reads_pay_the_check": the_read_side_pays_one_span_check_per_read(),
        "overlaps_merge": overlapping_ranges_merge_into_one_span(),
        "later_writes_survive": a_write_after_the_delete_survives_it(),
        "disjoint_stays_disjoint": disjoint_ranges_stay_separate(),
    }
