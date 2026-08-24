from __future__ import annotations

import functools
from dataclasses import dataclass, field

from store.engine import Store, crash
from store.errors import Closed, ConfigError, Corrupt
from store.record import Record, decode_all
from store.wal import frame, unframe

# The write batch: several keys, one fate.
#
# A put moves one key. An application move, debit here and credit there, index entry and row
# together, is several keys that must land together or not at all, and the transaction module
# solved this with snapshots and validation. The write batch is the smaller tool: no reads,
# no conflicts, just atomicity of application, bought with one trick in the log format. The
# batch's records are framed as one payload, and recovery applies a frame's records all or
# nothing, so the crash cannot land inside a batch. The measurements walk every crash point
# and check exactly that.


@dataclass
class Batch:
    """Writes accumulated in insertion order, going nowhere until commit.

    One list, not a puts list and a deletes list, and the first draft had two. The order
    test caught it: a put and a delete of one key inside a batch must resolve in insertion
    order, and separate lists erase the interleaving, so both orders resolved the same way.
    The test written to catch that bug in imagined implementations caught it in this one.
    """

    ops: list[tuple[str, bytes, bytes]] = field(default_factory=list)
    committed: bool = field(default=False)

    def put(self, key: bytes, value: bytes) -> Batch:
        """Add one write."""
        if self.committed:
            raise Closed("the batch is committed")
        if not key:
            raise ConfigError("a key needs at least one byte")
        self.ops.append(("put", key, value))
        return self

    def delete(self, key: bytes) -> Batch:
        """Add one delete."""
        if self.committed:
            raise Closed("the batch is committed")
        self.ops.append(("delete", key, b""))
        return self

    @property
    def operations(self) -> int:
        """Everything the batch will do."""
        return len(self.ops)


def commit(store: Store, batch: Batch) -> int:
    """One frame for the whole batch, then the memtable, then the sync.

    The engine's own put framed each record separately, which was atomic per record and
    nothing more. Here the batch's records share one frame, so the log holds all of them or
    none, and the recovery side needs no batch awareness at all: the frame is the atom.
    """
    if batch.committed:
        raise Closed("the batch is committed")
    if not batch.operations:
        raise ConfigError("an empty batch commits nothing")
    records = []
    for kind, key, value in batch.ops:
        store.sequence += 1
        if kind == "put":
            records.append(Record(key=key, sequence=store.sequence, value=value))
        else:
            records.append(Record(key=key, sequence=store.sequence, kind=1, value=b""))
    payload = b"".join(record.encode() for record in records)
    store.wal.disk.append(frame(payload))
    store.wal.disk.sync()
    for record in records:
        store.memtable.put(record)
    batch.committed = True
    return store.sequence


def recover_batched(raw: bytes) -> list[Record]:
    """Replay frames, each frame's records together, stopping at damage."""
    found: list[Record] = []
    at = 0
    while at < len(raw):
        try:
            payload, end = unframe(raw, at)
        except Corrupt:
            break
        found.extend(decode_all(payload))
        at = end
    return found


@functools.cache
def no_crash_point_splits_a_batch() -> bool:
    """The log truncated at every byte recovers whole batches only, checked exhaustively.

    Three batches of mixed sizes, and the recovered record count at every one of the log's
    byte positions is always a batch boundary: zero, three, seven, or nine records, never
    one, never five. The atom is the frame, and this walk is the proof that no torn write
    can ever surface half a move.
    """
    store = Store(flush_at=10**9, fold_at=10**9)
    commit(store, Batch().put(b"a1", b"v").put(b"a2", b"v").put(b"a3", b"v"))
    commit(store, Batch().put(b"b1", b"v").put(b"b2", b"v").put(b"b3", b"v").put(b"b4", b"v"))
    commit(store, Batch().put(b"c1", b"v").delete(b"a1"))
    raw = store.wal.disk.read()
    legal = {0, 3, 7, 9}
    return all(len(recover_batched(raw[:cut])) in legal for cut in range(len(raw) + 1))


@functools.cache
def a_crash_after_commit_keeps_the_whole_batch() -> bool:
    """The engine's crash path recovers every key of a committed batch together.

    The commit synced its frame, so the batch is in the durable prefix, and the survivor
    answers every key in it. The complementary case, a crash before the sync, loses every
    key together, which the frame guarantees by the same argument.
    """
    store = Store(flush_at=10**9, fold_at=10**9)
    commit(store, Batch().put(b"debit", b"-100").put(b"credit", b"+100"))
    survivor = crash(store)
    both = survivor.get(b"debit") == b"-100" and survivor.get(b"credit") == b"+100"
    fresh = Store(flush_at=10**9, fold_at=10**9)
    fresh.wal.policy = "never"
    batch = Batch().put(b"x", b"1").put(b"y", b"2")
    if batch.operations != 2:
        return False
    records = []
    for _, key, value in batch.ops:
        fresh.sequence += 1
        records.append(Record(key=key, sequence=fresh.sequence, value=value))
    payload = b"".join(record.encode() for record in records)
    fresh.wal.disk.append(frame(payload))
    for record in records:
        fresh.memtable.put(record)
    lost = crash(fresh)
    neither = lost.get(b"x") is None and lost.get(b"y") is None
    return both and neither


@functools.cache
def batch_order_within_is_preserved() -> bool:
    """A put and a delete of the same key inside one batch resolve in batch order.

    The batch assigns sequences in insertion order, so put then delete deletes, and delete
    then put survives. An implementation that grouped puts before deletes would resolve
    both the same way, and this pair of cases is the test that catches it.
    """
    first = Store(flush_at=10**9, fold_at=10**9)
    commit(first, Batch().put(b"k", b"v").delete(b"k"))
    second = Store(flush_at=10**9, fold_at=10**9)
    batch = Batch()
    batch.delete(b"k")
    batch.put(b"k", b"v")
    commit(second, batch)
    return first.get(b"k") is None and second.get(b"k") == b"v"


@functools.cache
def a_committed_batch_refuses_reuse() -> bool:
    """Adding to or recommitting a committed batch raises, the stale handle rule again."""
    store = Store(flush_at=10**9, fold_at=10**9)
    batch = Batch().put(b"k", b"v")
    commit(store, batch)
    try:
        batch.put(b"more", b"v")
        return False
    except Closed:
        pass
    try:
        commit(store, batch)
        return False
    except Closed:
        return True


@functools.cache
def an_empty_batch_is_refused() -> bool:
    """Committing nothing is a caller bug, not a no-op."""
    store = Store(flush_at=10**9, fold_at=10**9)
    try:
        commit(store, Batch())
    except ConfigError:
        return True
    return False


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "no_crash_splits_a_batch": no_crash_point_splits_a_batch(),
        "commits_keep_batches_whole": a_crash_after_commit_keeps_the_whole_batch(),
        "order_within_is_preserved": batch_order_within_is_preserved(),
        "committed_batches_refuse_reuse": a_committed_batch_refuses_reuse(),
        "empty_batches_are_refused": an_empty_batch_is_refused(),
    }
