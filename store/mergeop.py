from __future__ import annotations

import functools
import random
import struct
from dataclasses import dataclass, field

from store.errors import BadFormat
from store.record import MERGE, PUT, Record

# The merge operator: writing the change instead of the value.
#
# A counter incremented through a store costs a read, an add, and a write, and the read is the
# expensive part: it walks the whole read path to fetch a value the writer only wants to add
# one to. The merge operator removes the read. The writer appends a record that says add one,
# and the store folds the additions into the base value later, at read time or at compaction,
# whichever comes first.
#
# The price is that the store no longer knows the value of the key without folding, so a read
# of a heavily merged key does the work all the skipped reads avoided, once. Whether that is a
# win is a ratio of writes to reads, and it is measured here rather than asserted.

COUNTER = struct.Struct("<q")


def pack(value: int) -> bytes:
    """A counter value or delta as bytes."""
    return COUNTER.pack(value)


def unpack(raw: bytes) -> int:
    """The counter back."""
    if len(raw) != COUNTER.size:
        raise BadFormat(f"{len(raw)} bytes is not a counter")
    return COUNTER.unpack(raw)[0]


def fold(base: bytes | None, deltas: list[bytes]) -> bytes:
    """The value a base and a run of additions come to."""
    total = unpack(base) if base is not None else 0
    for delta in deltas:
        total += unpack(delta)
    return pack(total)


@dataclass
class Counters:
    """A store of counters, with both write paths and meters on each."""

    versions: dict[bytes, list[Record]] = field(default_factory=dict)
    sequence: int = field(default=0)
    reads: int = field(default=0)
    writes: int = field(default=0)
    folds: int = field(default=0)
    records_folded: int = field(default=0)

    def _append(self, key: bytes, kind: int, value: bytes) -> None:
        self.sequence += 1
        record = Record(key=key, sequence=self.sequence, kind=kind, value=value)
        self.versions.setdefault(key, []).insert(0, record)
        self.writes += 1

    def put(self, key: bytes, value: int) -> None:
        """Set a counter outright, which starts a new base."""
        self._append(key, PUT, pack(value))

    def add_by_reading(self, key: bytes, delta: int) -> None:
        """The read modify write path: fetch, add, store."""
        held = self.get(key)
        self._append(key, PUT, pack(held + delta))

    def add(self, key: bytes, delta: int) -> None:
        """The merge path: append the delta, fold later."""
        self._append(key, MERGE, pack(delta))

    def get(self, key: bytes) -> int:
        """The counter's value, folding whatever has accumulated."""
        self.reads += 1
        held = self.versions.get(key, [])
        deltas: list[bytes] = []
        base: bytes | None = None
        walked = 0
        for record in held:
            walked += 1
            if record.kind == MERGE:
                deltas.append(record.value)
                continue
            base = record.value
            break
        if walked > 1 or (walked == 1 and deltas):
            self.folds += 1
            self.records_folded += walked
        folded = fold(base, list(reversed(deltas)))
        return unpack(folded)

    def compact(self, key: bytes) -> None:
        """Fold a key's history into one base record, which is what compaction does."""
        value = self.get(key)
        self.versions[key] = []
        self._append(key, PUT, pack(value))

    def depth(self, key: bytes) -> int:
        """How many records a read of the key must walk."""
        held = self.versions.get(key, [])
        walked = 0
        for record in held:
            walked += 1
            if record.kind != MERGE:
                break
        return walked

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "keys": len(self.versions),
            "writes": self.writes,
            "reads": self.reads,
            "folds": self.folds,
            "records_folded": self.records_folded,
        }


@functools.cache
def the_merge_path_writes_what_the_read_path_reads() -> bool:
    """Ten thousand increments: the read path does 10,000 reads, the merge path does none.

    Both paths end with the counter at ten thousand. The read modify write path walked the
    read path ten thousand times to learn values it immediately overwrote. The merge path
    deferred all of it to the single read at the end, which folded 10,000 records once.

    The total work is not smaller, it has moved. Whether the move pays depends on whether
    anyone reads between the writes, and the next measurement is that case.
    """
    reader = Counters()
    merger = Counters()
    for _ in range(10000):
        reader.add_by_reading(b"hits", 1)
        merger.add(b"hits", 1)
    reads_before = merger.reads
    return (
        reader.get(b"hits") == merger.get(b"hits") == 10000
        and reader.reads == 10001
        and reads_before == 0
    )


@functools.cache
def a_read_between_every_write_cancels_the_saving() -> bool:
    """Alternating add and get, the merge path folds as often as the read path read.

    One read per write means every deferred fold happens immediately, so the merge operator
    saved nothing and added record depth. The ratio of writes to reads is the entire decision:
    a metrics counter written a thousand times per scrape wins hugely, a balance checked after
    every deposit does not.
    """
    merger = Counters()
    for _ in range(1000):
        merger.add(b"k", 1)
        merger.get(b"k")
    return merger.records_folded >= 1000


@functools.cache
def an_unread_counter_grows_without_bound_until_compaction() -> bool:
    """Ten thousand increments leave ten thousand records, and one compaction leaves one.

    The merge operator converts read cost into space until something folds. Compaction is that
    something, and after it the key is one base record holding the same value. The depth
    before and after is the measurement: 10,000 to 1.
    """
    merger = Counters()
    for _ in range(10000):
        merger.add(b"k", 1)
    deep = merger.depth(b"k")
    merger.compact(b"k")
    return deep == 10000 and merger.depth(b"k") == 1 and merger.get(b"k") == 10000


@functools.cache
def a_put_cuts_the_fold_short() -> bool:
    """A set after a thousand adds makes the adds invisible and the fold one record deep.

    The fold stops at the first non merge record, so a put is a wall: everything below it is
    dead the moment the put lands, without compaction. Counters that get reset periodically
    self clean, and counters that only grow do not, which is a workload property the operator
    inherits rather than a design choice.
    """
    merger = Counters()
    for _ in range(1000):
        merger.add(b"k", 1)
    merger.put(b"k", 5)
    merger.add(b"k", 2)
    return merger.get(b"k") == 7 and merger.depth(b"k") == 2


@functools.cache
def folding_is_associative_so_partial_folds_are_safe() -> bool:
    """Folding in two stages gives what one stage gives, for a thousand random splits.

    This is the property compaction relies on: a fold of the bottom half followed by a fold
    of the result with the top half must equal folding everything at once. Addition has it.
    An operator without it, subtraction of maxima say, silently corrupts under compaction,
    and the store cannot check because the store does not know the semantics.
    """
    source = random.Random(3)
    for _ in range(1000):
        deltas = [pack(source.randrange(-50, 50)) for _ in range(20)]
        cut = source.randrange(1, 19)
        base = pack(source.randrange(100))
        once = fold(base, deltas)
        staged = fold(fold(base, deltas[:cut]), deltas[cut:])
        if once != staged:
            return False
    return True


def compare_the_ratios(writes: int = 2000) -> list[dict]:
    """One row per read to write ratio, folded records against reads saved."""
    rows = []
    for reads_per_write in (0.0, 0.01, 0.1, 1.0):
        merger = Counters()
        gap = int(1 / reads_per_write) if reads_per_write else writes + 1
        for at in range(writes):
            merger.add(b"k", 1)
            if reads_per_write and at % gap == gap - 1:
                merger.get(b"k")
        merger.get(b"k")
        rows.append(
            {
                "reads_per_write": reads_per_write,
                "reads": merger.reads,
                "records_folded": merger.records_folded,
                "folded_per_read": round(merger.records_folded / merger.reads, 1),
            }
        )
    return rows


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "the_read_moves_not_shrinks": the_merge_path_writes_what_the_read_path_reads(),
        "reads_cancel_the_saving": a_read_between_every_write_cancels_the_saving(),
        "unread_counters_grow": an_unread_counter_grows_without_bound_until_compaction(),
        "a_put_is_a_wall": a_put_cuts_the_fold_short(),
        "folding_is_associative": folding_is_associative_so_partial_folds_are_safe(),
        "read_folding_is_quadratic": read_time_folding_without_write_back_is_quadratic(),
    }


@functools.cache
def read_time_folding_without_write_back_is_quadratic() -> bool:
    """Two thousand writes read after every write fold two million records.

    The table above measures it: a get folds the accumulated history and throws the folded
    result away, so the next get folds the same records again plus one. Reading after every
    write folds 2,003,000 records for 2,000 writes, which is the arithmetic series, n squared
    over two.

    The fix is either of the fold's two homes. Compaction writes the folded base back, so
    each record is folded once ever. Or the read itself can write back, which is what real
    engines do for hot keys. What is not viable is exactly what this store does, and it is
    the naive reading of how a merge operator works, which is why it is measured.
    """
    merger = Counters()
    for _ in range(2000):
        merger.add(b"k", 1)
        merger.get(b"k")
    return merger.records_folded > 1_900_000
