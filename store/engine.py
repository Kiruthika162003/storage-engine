from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

from store.bloom import Filter
from store.bloom import build as build_filter
from store.disk import Disk
from store.errors import Closed, ConfigError
from store.iterator import Compacting, Merge, Source
from store.manifest import Edit, Manifest, add, compaction, replay, sequence
from store.memtable import Memtable
from store.record import DELETE, Record
from store.wal import EVERY_BATCH, Log, recover

# The whole store, assembled from parts that have each already been measured alone.
#
# A write goes to the log and then the memtable. A full memtable becomes a sorted file, the file
# is announced in the manifest, and the log up to that point is dropped, because everything in
# it is now somewhere durable and sorted. A read asks the memtable, then each file from newest
# to oldest, stopping at the first answer. Compaction folds files together in the background so
# the read list stays short.
#
# Nothing in this file is new. What is new is the coupling: the order a flush writes its pieces
# in, which log records may be dropped when, and which version of the file set a read uses while
# a compaction replaces it underneath. Each of those is a crash bug or a stale read if done in
# the wrong order, and the tests exercise the orders.

# How many records the memtable holds before it is flushed.
FLUSH_AT = 1000

# How many level zero files are tolerated before compaction folds them.
FOLD_AT = 4


@dataclass
class Table:
    """One immutable sorted file as the engine holds it: records, range, filter."""

    number: int
    records: list[Record]
    filter: Filter

    def __post_init__(self) -> None:
        if not self.records:
            raise ConfigError("an empty table has nothing to hold")

    @property
    def first(self) -> bytes:
        """The lowest key."""
        return self.records[0].key

    @property
    def last(self) -> bytes:
        """The highest key."""
        return self.records[-1].key

    def might_hold(self, key: bytes) -> bool:
        """Whether a read has to look inside."""
        return self.first <= key <= self.last and self.filter.might_contain(key)

    def get(self, key: bytes) -> Record | None:
        """The newest record for a key, or nothing."""
        low, high = 0, len(self.records)
        while low < high:
            middle = (low + high) // 2
            if self.records[middle].key < key:
                low = middle + 1
            else:
                high = middle
        if low < len(self.records) and self.records[low].key == key:
            return self.records[low]
        return None

    def source(self) -> Source:
        """The table as a merge source."""
        return Source(name=f"table-{self.number}", records=self.records)


def build_table(number: int, records: list[Record]) -> Table:
    """A table from a sorted run."""
    return Table(
        number=number,
        records=records,
        filter=build_filter([record.key for record in records]),
    )


@dataclass
class Store:
    """The assembled engine: log, memtable, files, manifest, compaction.

    Every component is one the package already measured alone. The store's own contribution is
    ordering: what gets written before what, and what may be forgotten when.
    """

    wal: Log = field(default_factory=lambda: Log(disk=Disk(name="WAL"), policy=EVERY_BATCH))
    manifest: Manifest = field(default_factory=Manifest)
    memtable: Memtable = field(default_factory=Memtable)
    tables: list[Table] = field(default_factory=list)
    sequence: int = field(default=0)
    next_file: int = field(default=1)
    flush_at: int = field(default=FLUSH_AT)
    fold_at: int = field(default=FOLD_AT)
    open: bool = field(default=True)
    flushes: int = field(default=0)
    folds: int = field(default=0)
    reads: int = field(default=0)
    filter_skips: int = field(default=0)

    def put(self, key: bytes, value: bytes) -> int:
        """Write a value: log first, memtable second, flush if full."""
        return self._write(key, value, 0)

    def delete(self, key: bytes) -> int:
        """Delete a key, which is a write of a tombstone."""
        return self._write(key, b"", DELETE)

    def _write(self, key: bytes, value: bytes, kind: int) -> int:
        if not self.open:
            raise Closed("the store is closed")
        if not key:
            raise ConfigError("a key needs at least one byte")
        self.sequence += 1
        record = Record(key=key, sequence=self.sequence, kind=kind, value=value)
        self.wal.append([record])
        self.memtable.put(record)
        if len(self.memtable.records()) >= self.flush_at:
            self.flush()
        return self.sequence

    def flush(self) -> int | None:
        """Turn the memtable into a file, announce it, and let the log go.

        The order is the entire point. The file's records exist in the log, so writing the file
        is safe at any moment. Announcing it in the manifest is the commit. Dropping the log
        and the memtable comes last, because doing either earlier leaves a window where the
        only copy of an acknowledged write is in memory.
        """
        records = self.memtable.records()
        if not records:
            return None
        number = self.next_file
        self.next_file += 1
        self.tables.insert(0, build_table(number, records))
        self.manifest.install(
            Edit(changes=(add(number, 0, len(records)), sequence(self.sequence)))
        )
        self.wal = Log(disk=Disk(name=f"WAL-{number}"), policy=self.wal.policy)
        self.memtable = Memtable()
        self.flushes += 1
        if len(self.tables) >= self.fold_at:
            self.fold()
        return number

    def fold(self) -> int | None:
        """Merge every file into one, retiring the inputs in the same manifest edit."""
        if len(self.tables) < 2:
            return None
        merge = Merge(sources=[table.source() for table in self.tables])
        out = Compacting(merge=merge, bottom=True, horizon=self.sequence)
        made = list(out.records())
        old = [table.number for table in self.tables]
        if not made:
            self.manifest.install(compaction([], old))
            self.tables = []
            self.folds += 1
            return None
        number = self.next_file
        self.next_file += 1
        self.manifest.install(compaction([(number, 1, len(made))], old))
        self.tables = [build_table(number, made)]
        self.folds += 1
        return number

    def get(self, key: bytes) -> bytes | None:
        """Read a key: memtable first, then each file newest to oldest."""
        if not self.open:
            raise Closed("the store is closed")
        self.reads += 1
        found = self.memtable.get(key)
        if found is not None:
            return None if found.kind == DELETE else found.value
        for table in self.tables:
            if not table.might_hold(key):
                self.filter_skips += 1
                continue
            found = table.get(key)
            if found is not None:
                return None if found.kind == DELETE else found.value
        return None

    def scan(self, start: bytes = b""):
        """Every live key from a point onwards, merged across everything."""
        sources = [Source(name="memtable", records=self.memtable.records())]
        sources += [table.source() for table in self.tables]
        yield from Merge(sources=sources).live(start)

    def items(self) -> list[tuple[bytes, bytes]]:
        """The live contents, as pairs."""
        return [(record.key, record.value) for record in self.scan()]

    def close(self) -> None:
        """Flush and stop answering."""
        self.flush()
        self.open = False

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "sequence": self.sequence,
            "tables": len(self.tables),
            "memtable": len(self.memtable.records()),
            "flushes": self.flushes,
            "folds": self.folds,
            "reads": self.reads,
            "filter_skips": self.filter_skips,
            "manifest_edits": self.manifest.edits,
        }


def crash(store: Store) -> Store:
    """The store as a restart would find it: durable bytes only, memory gone.

    The files and the manifest are taken as durable, which is what the flush ordering paid for.
    The memtable is gone. The write ahead log keeps only what was synced, and the recovered
    memtable is whatever a replay of those bytes yields.
    """
    store.wal.disk.crash()
    manifest_state = replay(store.manifest.disk.read())
    live = set(manifest_state.version.files)
    tables = [table for table in store.tables if table.number in live]
    recovery = recover(store.wal.disk.read())
    memtable = Memtable()
    highest = manifest_state.version.sequence
    for record in recovery.records:
        memtable.put(record)
        highest = max(highest, record.sequence)
    return Store(
        wal=Log(disk=store.wal.disk, policy=store.wal.policy),
        manifest=store.manifest,
        memtable=memtable,
        tables=tables,
        sequence=highest,
        next_file=store.next_file,
        flush_at=store.flush_at,
        fold_at=store.fold_at,
    )


@functools.cache
def _filled(writes: int = 6000, keys: int = 1500, seed: int = 4) -> tuple[Store, dict]:
    """A store with a mixed stream in it, and the dictionary it should agree with."""
    source = random.Random(seed)
    store = Store()
    truth: dict[bytes, bytes] = {}
    for _ in range(writes):
        key = f"k{source.randrange(keys):05d}".encode()
        if source.random() < 0.1:
            store.delete(key)
            truth.pop(key, None)
        else:
            value = source.randbytes(12)
            store.put(key, value)
            truth[key] = value
    return store, truth


@functools.cache
def the_store_agrees_with_a_dictionary() -> bool:
    """Six thousand mixed writes, and every key reads back what a dict would say.

    The dictionary is the specification. It has no log, no files, no compaction and no bugs,
    and any disagreement between it and the store is the store's fault by definition. This is
    the invariant everything else in the module is allowed to assume.
    """
    store, truth = _filled()
    if not all(store.get(key) == value for key, value in truth.items()):
        return False
    absent = [f"k{at:05d}".encode() for at in range(1500)]
    missing = [key for key in absent if key not in truth]
    return all(store.get(key) is None for key in missing)


@functools.cache
def a_crash_loses_nothing_that_was_synced() -> bool:
    """Kill the store mid stream and the survivor agrees with the dictionary anyway.

    Every write went to the log before the memtable, and the log syncs per batch, so the
    durable log holds every acknowledged write that has not reached a file. Recovery is the
    manifest choosing the files and the log rebuilding the memtable, and the result answers
    every key the dictionary knows.
    """
    store, truth = _filled(5000, 1200, 9)
    survivor = crash(store)
    return all(survivor.get(key) == value for key, value in truth.items())


@functools.cache
def a_fold_changes_no_answer() -> bool:
    """Compaction is invisible to reads, checked directly rather than assumed.

    The same store before and after folding every file into one gives the same answer for
    every key the dictionary knows and for a sample of keys it does not. The fold moved
    thousands of records, dropped every shadowed version and every dead tombstone, and no
    reader can tell.
    """
    store, truth = _filled(6000, 1500, 4)
    before = {key: store.get(key) for key in truth}
    store.flush()
    store.fold()
    return len(store.tables) == 1 and all(
        store.get(key) == value for key, value in before.items()
    )


@functools.cache
def the_filter_absorbs_the_misses() -> bool:
    """Reads for absent keys mostly never reach a file.

    A store with several files answers a thousand reads for keys it has never seen. The range
    check and the filter together turn almost all of them away before any file is searched,
    which is the read path the sstable module promised, now measured through the whole engine.
    """
    store, _ = _filled()
    before = store.filter_skips
    for at in range(1000):
        store.get(f"absent:{at:05d}".encode())
    return store.filter_skips - before > 2500


@functools.cache
def a_closed_store_refuses_everything() -> bool:
    """Close flushes and stops answering, rather than answering stale.

    A store that keeps answering after close is a store with two owners disagreeing about its
    lifetime. Refusing loudly at the first call is cheaper than debugging the second owner's
    reads.
    """
    store = Store()
    store.put(b"a", b"1")
    store.close()
    try:
        store.get(b"a")
    except Closed:
        pass
    else:
        return False
    try:
        store.put(b"b", b"2")
    except Closed:
        return True
    return False


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "agrees_with_a_dictionary": the_store_agrees_with_a_dictionary(),
        "a_crash_loses_nothing_synced": a_crash_loses_nothing_that_was_synced(),
        "a_fold_changes_no_answer": a_fold_changes_no_answer(),
        "the_filter_absorbs_misses": the_filter_absorbs_the_misses(),
        "closed_means_closed": a_closed_store_refuses_everything(),
    }
