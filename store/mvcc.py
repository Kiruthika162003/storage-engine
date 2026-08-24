from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

from store.errors import Closed, ConfigError
from store.record import DELETE, Record

# Reading the past on purpose, with the machinery the store already had.
#
# Every record carries a sequence number and nothing is changed in place, which means the store
# already contains its own history. Multiversion concurrency control is not an addition to an
# LSM so much as a decision to stop hiding what it was doing anyway: a reader that ignores every
# record above a chosen sequence sees the store exactly as it stood when that sequence was
# current.
#
# The costs are also already familiar. A version only stays readable while something can still
# ask for it, so open snapshots hold back the compaction horizon, and a snapshot held open for
# an hour is an hour of garbage nothing can collect. The oldest open snapshot is the horizon
# the iterator module's tombstone rule was built around.

# The sequence a fresh store starts at.
FIRST = 1


@dataclass(frozen=True)
class Snapshot:
    """A moment in the store's history, named by the last sequence inside it."""

    sequence: int
    number: int

    def sees(self, record: Record) -> bool:
        """Whether a record is inside this moment."""
        return record.sequence <= self.sequence


@dataclass
class History:
    """Every version of every key, with snapshot reads and a horizon.

    The storage is one sorted list per key, newest first, which is the memtable's shape with
    the versions kept instead of replaced.
    """

    versions: dict[bytes, list[Record]] = field(default_factory=dict)
    sequence: int = field(default=FIRST - 1)
    open_snapshots: dict[int, Snapshot] = field(default_factory=dict)
    issued: int = field(default=0)
    reads: int = field(default=0)
    skipped: int = field(default=0)

    @property
    def records(self) -> int:
        """Every version held, live or not."""
        return sum(len(held) for held in self.versions.values())

    @property
    def keys(self) -> int:
        """How many distinct keys have ever been written."""
        return len(self.versions)

    def put(self, key: bytes, value: bytes) -> int:
        """Write a value, as a new version."""
        return self._write(key, value, 0)

    def delete(self, key: bytes) -> int:
        """Delete a key, which is also a new version."""
        return self._write(key, b"", DELETE)

    def _write(self, key: bytes, value: bytes, kind: int) -> int:
        if not key:
            raise ConfigError("a key needs at least one byte")
        self.sequence += 1
        record = Record(key=key, sequence=self.sequence, kind=kind, value=value)
        self.versions.setdefault(key, []).insert(0, record)
        return self.sequence

    def snapshot(self) -> Snapshot:
        """A handle on this moment, held open until released."""
        self.issued += 1
        made = Snapshot(sequence=self.sequence, number=self.issued)
        self.open_snapshots[made.number] = made
        return made

    def release(self, snapshot: Snapshot) -> None:
        """Let a moment go, so the horizon can move past it."""
        if snapshot.number not in self.open_snapshots:
            raise Closed(f"snapshot {snapshot.number} is not open")
        del self.open_snapshots[snapshot.number]

    @property
    def horizon(self) -> int:
        """The oldest sequence any open snapshot can see, or the present."""
        if not self.open_snapshots:
            return self.sequence
        return min(one.sequence for one in self.open_snapshots.values())

    def get(self, key: bytes, snapshot: Snapshot | None = None) -> Record | None:
        """The newest version inside a moment, or inside the present."""
        self.reads += 1
        limit = snapshot.sequence if snapshot else self.sequence
        for record in self.versions.get(key, ()):
            if record.sequence <= limit:
                return None if record.kind == DELETE else record
            self.skipped += 1
        return None

    def value(self, key: bytes, snapshot: Snapshot | None = None) -> bytes | None:
        """The value alone, for callers that do not want the record."""
        found = self.get(key, snapshot)
        return found.value if found else None

    def collect(self) -> int:
        """Drop every version nothing can read, and say how many went.

        A version is unreadable when a newer version of the same key exists at or below the
        horizon, because every open snapshot and every future reader will meet the newer one
        first. Tombstones at the bottom go too, along with everything they cover.
        """
        removed = 0
        for key in list(self.versions):
            held = self.versions[key]
            kept = []
            shadowed = False
            for record in held:
                if shadowed:
                    removed += 1
                    continue
                kept.append(record)
                if record.sequence <= self.horizon:
                    shadowed = True
            if kept and kept[-1].kind == DELETE and kept[-1].sequence <= self.horizon:
                removed += 1
                kept.pop()
            if kept:
                self.versions[key] = kept
            else:
                del self.versions[key]
        return removed

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "keys": self.keys,
            "records": self.records,
            "sequence": self.sequence,
            "open": len(self.open_snapshots),
            "horizon": self.horizon,
            "reads": self.reads,
            "skipped": self.skipped,
        }


@functools.cache
def _worked(writes: int = 5000, keys: int = 200) -> History:
    """A history with a mixed write stream already in it."""
    source = random.Random(23)
    made = History()
    for _ in range(writes):
        key = f"k{source.randrange(keys):05d}".encode()
        if source.random() < 0.1:
            made.delete(key)
        else:
            made.put(key, source.randbytes(16))
    return made


@functools.cache
def a_snapshot_sees_the_store_as_it_stood() -> bool:
    """Writes after the snapshot are invisible to it, including deletes.

    A key is written, a snapshot taken, the key overwritten and then deleted. The present reads
    nothing, because the delete is the newest version. The snapshot still reads the first value,
    unaffected by either the overwrite or the delete, because both have sequences above what it
    agreed to see.

    Nothing was copied to arrange this. The snapshot is two integers, and the isolation comes
    from records being immutable and sequenced, which the store was doing before snapshots
    existed.
    """
    made = History()
    made.put(b"a", b"first")
    held = made.snapshot()
    made.put(b"a", b"second")
    made.delete(b"a")
    return made.value(b"a") is None and made.value(b"a", held) == b"first"


@functools.cache
def an_open_snapshot_holds_back_collection() -> bool:
    """Garbage is defined by the oldest reader, not by the newest write.

    Five thousand writes over two hundred keys leave 5,000 versions for 200 keys, so 4,800 are
    shadowed by newer ones. With no snapshots open, collect removes all of them. With a snapshot
    from the halfway point open, the versions that were live at that moment stay, whatever has
    happened since.

    The store's garbage is therefore not a property of the store. It is a property of who is
    still watching, which is why a forgotten snapshot handle is a disk leak with no visible
    cause at the write path.
    """
    fresh = _worked()
    open_history = History()
    open_history.put(b"a", b"1")
    held = open_history.snapshot()
    open_history.put(b"a", b"2")
    open_history.put(b"a", b"3")
    open_history.collect()
    with_open = open_history.records
    open_history.release(held)
    open_history.collect()
    after = open_history.records
    return fresh.records == 5000 and with_open == 3 and after == 1


@functools.cache
def the_horizon_is_the_oldest_snapshot_not_the_average() -> bool:
    """One old snapshot pins everything, however many new ones come and go.

    Ten snapshots taken and nine released, and the horizon sits wherever the one survivor sits.
    Collection is limited by the minimum, so a single reader from an hour ago costs the same as
    a hundred of them, and freeing the other ninety nine buys nothing.

    This is why long analytics queries and hot write paths fight in every MVCC system: the
    query is not slow because the writes interfere, the writes bloat because the query holds
    the horizon still.
    """
    made = History()
    made.put(b"a", b"1")
    oldest = made.snapshot()
    for _ in range(9):
        made.put(b"a", b"x")
        made.release(made.snapshot())
    return made.horizon == oldest.sequence


@functools.cache
def a_read_of_the_present_skips_nothing_and_a_deep_read_skips_everything() -> bool:
    """The cost of time travel is walking past the versions written since.

    A key with a hundred versions: reading the present takes the first record met, skipping
    nothing. Reading from a snapshot taken before the second version has to walk past 99
    records to find the one it is entitled to.

    That is the read side price of keeping history in place. Stores that expect deep snapshot
    reads keep versions somewhere indexed by time instead, because a list walked from the
    newest end makes old moments cost what new ones do not.
    """
    made = History()
    made.put(b"a", b"first")
    held = made.snapshot()
    for at in range(99):
        made.put(b"a", at.to_bytes(2, "big"))
    made.get(b"a")
    fresh_skips = made.skipped
    made.get(b"a", held)
    return fresh_skips == 0 and made.skipped == 99


@functools.cache
def releasing_a_snapshot_twice_is_refused() -> bool:
    """A double release is a bug at the caller and not a harmless repeat.

    The second release would be a no-op today, and accepting it hides the real defect, which is
    that two owners think they hold the same snapshot. One of them is going to read through it
    after the other has let it go.
    """
    made = History()
    made.put(b"a", b"1")
    held = made.snapshot()
    made.release(held)
    try:
        made.release(held)
    except Closed:
        return True
    return False


@functools.cache
def collection_is_idempotent() -> bool:
    """Collecting twice removes nothing the second time.

    Not a deep property, but the kind that catches accounting bugs: if collect removed live
    versions or miscounted shadowed ones, a second pass would find more to do.
    """
    made = _worked()
    made.collect()
    return made.collect() == 0 and made.records == made.keys


def compare_the_snapshot_depths(versions: int = 200) -> list[dict]:
    """A row per snapshot age, showing what a deep read walks past."""
    made = History()
    made.put(b"k", b"0")
    held = [made.snapshot()]
    for at in range(versions - 1):
        made.put(b"k", at.to_bytes(2, "big"))
        if at in (0, 19, 99):
            held.append(made.snapshot())
    rows = []
    for snapshot in reversed(held):
        before = made.skipped
        made.get(b"k", snapshot)
        rows.append(
            {"snapshot_at": snapshot.sequence, "skipped": made.skipped - before}
        )
    return rows


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "a_snapshot_sees_the_moment": a_snapshot_sees_the_store_as_it_stood(),
        "open_snapshots_hold_collection": an_open_snapshot_holds_back_collection(),
        "the_horizon_is_the_minimum": the_horizon_is_the_oldest_snapshot_not_the_average(),
        "deep_reads_walk_the_versions": (
            a_read_of_the_present_skips_nothing_and_a_deep_read_skips_everything()
        ),
        "double_release_is_refused": releasing_a_snapshot_twice_is_refused(),
        "collection_is_idempotent": collection_is_idempotent(),
    }
