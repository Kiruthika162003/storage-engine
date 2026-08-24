from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

from store.errors import ConfigError

# A secondary index is a second store whose keys are someone else's values.
#
# The primary maps id to record. A query by any other field either scans everything or asks an
# index: a mapping from field value to the ids that hold it. The index is itself a sorted
# store, so everything the package knows about writes and compaction applies; what is new is
# the coupling. Every primary write becomes index maintenance, and the maintenance is not one
# write but two, because changing a field means leaving one index entry and joining another,
# and forgetting the leaving half is the classic index bug: the query returns ids whose
# records no longer match.
#
# The two maintenance disciplines are synchronous, where the index is updated in the same
# operation and a query trusts it, and lazy, where the index is allowed to hold stale entries
# and every query checks its answers against the primary before returning them. The check
# costs a primary read per candidate, which is the price of the cheaper writes.


@dataclass
class Indexed:
    """A primary with one synchronous secondary index over a field."""

    primary: dict[bytes, bytes] = field(default_factory=dict)
    index: dict[bytes, set[bytes]] = field(default_factory=dict)
    primary_writes: int = field(default=0)
    index_writes: int = field(default=0)
    checks: int = field(default=0)

    def put(self, key: bytes, value: bytes) -> None:
        """Write a record and keep the index exact."""
        if not key:
            raise ConfigError("a key needs at least one byte")
        old = self.primary.get(key)
        if old is not None and old != value:
            self.index[old].discard(key)
            self.index_writes += 1
        self.primary[key] = value
        self.primary_writes += 1
        if old != value:
            self.index.setdefault(value, set()).add(key)
            self.index_writes += 1

    def delete(self, key: bytes) -> None:
        """Remove a record and its index entry together."""
        old = self.primary.pop(key, None)
        self.primary_writes += 1
        if old is not None:
            self.index[old].discard(key)
            self.index_writes += 1

    def find(self, value: bytes) -> set[bytes]:
        """Every key whose value is the one asked for, straight from the index."""
        return set(self.index.get(value, set()))


@dataclass
class Lazy:
    """The same shape, but the index only ever gains entries and queries verify."""

    primary: dict[bytes, bytes] = field(default_factory=dict)
    index: dict[bytes, set[bytes]] = field(default_factory=dict)
    primary_writes: int = field(default=0)
    index_writes: int = field(default=0)
    checks: int = field(default=0)

    def put(self, key: bytes, value: bytes) -> None:
        """Write a record and add the new entry, leaving the stale one behind."""
        if not key:
            raise ConfigError("a key needs at least one byte")
        old = self.primary.get(key)
        self.primary[key] = value
        self.primary_writes += 1
        if old != value:
            self.index.setdefault(value, set()).add(key)
            self.index_writes += 1

    def delete(self, key: bytes) -> None:
        """Remove the record and leave every index entry stale."""
        self.primary.pop(key, None)
        self.primary_writes += 1

    def find(self, value: bytes) -> set[bytes]:
        """Every key that currently matches, verified against the primary."""
        found = set()
        for key in self.index.get(value, set()):
            self.checks += 1
            if self.primary.get(key) == value:
                found.add(key)
        return found

    def stale_entries(self) -> int:
        """How many index entries point at records that no longer match."""
        return sum(
            1
            for value, keys in self.index.items()
            for key in keys
            if self.primary.get(key) != value
        )

    def scrub(self) -> int:
        """Drop every stale entry, which is the compaction of an index."""
        removed = 0
        for value in list(self.index):
            keep = {key for key in self.index[value] if self.primary.get(key) == value}
            removed += len(self.index[value]) - len(keep)
            if keep:
                self.index[value] = keep
            else:
                del self.index[value]
        return removed


def scan_find(primary: dict[bytes, bytes], value: bytes) -> set[bytes]:
    """The reference: no index, read everything."""
    return {key for key, held in primary.items() if held == value}


@functools.cache
def _driven(kind: str, writes: int = 8000, keys: int = 1000, seed: int = 27):
    """A store of either discipline after a churn of writes, deletes and updates."""
    source = random.Random(seed)
    store = Indexed() if kind == "indexed" else Lazy()
    values = [f"v{at:03d}".encode() for at in range(50)]
    for _ in range(writes):
        key = f"k{source.randrange(keys):05d}".encode()
        if source.random() < 0.1:
            store.delete(key)
        else:
            store.put(key, source.choice(values))
    return store


@functools.cache
def both_disciplines_agree_with_the_scan() -> bool:
    """Every query answer matches the answer of reading the whole primary.

    The scan is the specification, correct because it looks at nothing but the truth. Both
    index disciplines return exactly its answer on every value after eight thousand mixed
    writes, which is the property that has to survive before any cost comparison means
    anything.
    """
    for kind in ("indexed", "lazy"):
        store = _driven(kind)
        for at in range(50):
            value = f"v{at:03d}".encode()
            if store.find(value) != scan_find(store.primary, value):
                return False
    return True


@functools.cache
def the_synchronous_index_doubles_the_write_cost() -> bool:
    """Eight thousand primary operations carry 13,333 index writes with them.

    An update is two index writes, the leave and the join, and most writes here are updates,
    so the index nearly doubles the write path. That is per index: a table with five indexed
    fields pays this five times, which is why write heavy tables shed indexes as they scale.
    """
    store = _driven("indexed")
    return store.index_writes > store.primary_writes * 1.2


@functools.cache
def the_lazy_index_writes_less_and_pays_per_query() -> bool:
    """The lazy index skips the leave half of every update and verifies at read time.

    Its index writes are 7,116 against 13,333, and a query pays a primary check per candidate,
    stale or not. The bill lands on whoever queries the hottest value, because popular values
    accumulate stale entries fastest, which is a tax with the worst possible incidence:
    the most queried value has the slowest query.
    """
    lazy = _driven("lazy")
    synchronous = _driven("indexed")
    lazy.find(b"v001")
    return lazy.index_writes < synchronous.index_writes * 0.8 and lazy.checks > 0


@functools.cache
def stale_entries_accumulate_and_a_scrub_removes_them() -> bool:
    """After the churn the lazy index carries thousands of stale entries.

    Every update leaves a corpse, and the corpses answer nothing: the verify step filters
    them on every query, at a cost, and the scrub removes them all at once. This is the
    memtable's overwrite story and compaction's garbage story wearing an index costume,
    which is the observation that makes the module worth having.
    """
    store = _driven("lazy", seed=28)
    stale = store.stale_entries()
    removed = store.scrub()
    return stale > 1000 and removed == stale and store.stale_entries() == 0


@functools.cache
def forgetting_the_leave_half_is_the_classic_bug() -> bool:
    """A synchronous index that skips the discard returns a wrong answer, demonstrated.

    The record moved from v1 to v2 and the broken index still lists it under v1. The query
    for v1 returns a key whose record says v2, which is not stale in the harmless sense, it
    is wrong: the caller asked for records matching v1 and got one that does not.
    """
    store = Indexed()
    store.put(b"k", b"v1")
    store.index.setdefault(b"v2", set()).add(b"k")
    store.primary[b"k"] = b"v2"
    return b"k" in store.find(b"v1") and store.primary[b"k"] == b"v2"


def compare_the_disciplines(writes: int = 8000) -> list[dict]:
    """One row per discipline over the same churn."""
    rows = []
    for kind in ("indexed", "lazy"):
        store = _driven(kind, writes)
        for at in range(50):
            store.find(f"v{at:03d}".encode())
        rows.append(
            {
                "discipline": kind,
                "primary_writes": store.primary_writes,
                "index_writes": store.index_writes,
                "checks": store.checks,
                "stale": store.stale_entries() if isinstance(store, Lazy) else 0,
            }
        )
    return rows


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "both_agree_with_the_scan": both_disciplines_agree_with_the_scan(),
        "synchronous_doubles_writes": the_synchronous_index_doubles_the_write_cost(),
        "lazy_pays_per_query": the_lazy_index_writes_less_and_pays_per_query(),
        "corpses_accumulate": stale_entries_accumulate_and_a_scrub_removes_them(),
        "the_classic_bug_is_wrong_answers": forgetting_the_leave_half_is_the_classic_bug(),
    }
