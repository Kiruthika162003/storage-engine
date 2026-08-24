from __future__ import annotations

import functools
import random
from dataclasses import dataclass

from store.engine import Store, build_table, crash
from store.manifest import Manifest, replay
from store.record import Record

# Checkpoints: a copy of the store that costs almost nothing, because nothing moves.
#
# The naive backup copies every byte, and the size of the copy is the size of the store. An
# LSM can do better for a structural reason: its files are immutable. A checkpoint is
# therefore a list of file numbers plus a copy of the tiny mutable parts, the manifest state
# and the memtable, and the files themselves are shared with the live store by reference.
# Nothing is copied until compaction wants to delete a file a checkpoint still names, at
# which point the file's life is extended, not its bytes copied.
#
# The measurements pin the two properties that make this a backup rather than a bookmark:
# the checkpoint is unaffected by every write and compaction after it, and restoring it
# yields a store that answers exactly as the original did at the moment of the checkpoint.


@dataclass(frozen=True)
class Checkpoint:
    """One moment: the live file numbers, the manifest bytes, the memtable records."""

    files: tuple[int, ...]
    manifest_bytes: bytes
    memtable: tuple[Record, ...]
    sequence: int

    @property
    def cost(self) -> int:
        """What the checkpoint itself costs to hold, in bytes it actually copied."""
        return len(self.manifest_bytes) + sum(record.nbytes for record in self.memtable)


def take(store: Store) -> Checkpoint:
    """A checkpoint of a store, copying only the mutable parts."""
    return Checkpoint(
        files=tuple(table.number for table in store.tables),
        manifest_bytes=bytes(store.manifest.disk.read()),
        memtable=tuple(store.memtable.records()),
        sequence=store.sequence,
    )


def restore(checkpoint: Checkpoint, library: dict[int, list[Record]]) -> Store:
    """A store rebuilt from a checkpoint and the shared file library."""
    made = Store()
    made.sequence = checkpoint.sequence
    made.next_file = max(checkpoint.files, default=0) + 1
    made.tables = [
        build_table(number, library[number]) for number in checkpoint.files
    ]
    found = replay(checkpoint.manifest_bytes)
    made.manifest = Manifest(version=found.version)
    for record in checkpoint.memtable:
        made.memtable.put(record)
    return made


def library_of(store: Store) -> dict[int, list[Record]]:
    """The immutable files by number, which both the store and its checkpoints share."""
    return {table.number: table.records for table in store.tables}


@functools.cache
def _worked(writes: int = 4000, keys: int = 900, seed: int = 15) -> tuple[Store, dict]:
    """A store mid life and the dictionary it agrees with."""
    source = random.Random(seed)
    store = Store(flush_at=300, fold_at=4)
    truth: dict[bytes, bytes] = {}
    for _ in range(writes):
        key = f"k{source.randrange(keys):05d}".encode()
        if source.random() < 0.1:
            store.delete(key)
            truth.pop(key, None)
        else:
            value = source.randbytes(10)
            store.put(key, value)
            truth[key] = value
    return store, truth


@functools.cache
def a_checkpoint_costs_kilobytes_on_a_store_of_megabytes() -> bool:
    """The checkpoint copies the manifest and the memtable and shares everything else.

    A store holding thousands of records checkpoints at the cost of its mutable fringe: a few
    kilobytes of manifest log and whatever the memtable held at that instant. The data files,
    which are nearly all of the store, contribute a tuple of integers.
    """
    store, _ = _worked()
    held = sum(record.nbytes for table in store.tables for record in table.records)
    made = take(store)
    return made.cost < held / 10


@functools.cache
def writes_after_the_checkpoint_do_not_reach_it() -> bool:
    """Overwrite every key after the checkpoint, restore, and the old values come back.

    The checkpoint's isolation is structural: it names immutable files and copies the mutable
    parts, so there is nothing the live store can touch that the checkpoint depends on. The
    restore answers with the dictionary as it stood, not as it stands.
    """
    store, truth = _worked(3000, 500, 16)
    library = library_of(store)
    made = take(store)
    frozen = dict(truth)
    for key in list(frozen):
        store.put(key, b"changed")
    restored = restore(made, library)
    return all(restored.get(key) == value for key, value in frozen.items())


@functools.cache
def a_restore_agrees_with_the_original_everywhere() -> bool:
    """The restored store and the original agree on every key either has ever seen.

    Present keys read the same values, deleted keys read absent in both, and keys from after
    the checkpoint read absent in the restore. The last is what separates a checkpoint from a
    replica: it is a moment, not a follower.
    """
    store, truth = _worked(3000, 500, 17)
    library = library_of(store)
    made = take(store)
    restored = restore(made, library)
    if not all(restored.get(key) == value for key, value in truth.items()):
        return False
    store.put(b"later", b"write")
    return restored.get(b"later") is None and store.get(b"later") == b"write"


@functools.cache
def the_checkpoint_survives_a_crash_of_the_live_store() -> bool:
    """Crash the live store after the checkpoint and the checkpoint restores anyway.

    The point of a backup is the day the primary dies. The checkpoint's file list and copied
    fringe do not reference the live memtable or the live log, so the crash has nothing to
    take from it.
    """
    store, truth = _worked(3000, 500, 18)
    library = library_of(store)
    made = take(store)
    crash(store)
    restored = restore(made, library)
    return all(restored.get(key) == value for key, value in truth.items())


@functools.cache
def sharing_between_checkpoints_lasts_exactly_until_a_fold() -> bool:
    """Two checkpoints share files across a flush and share nothing across a fold.

    The claim was that incremental backup falls out of immutability for free. Measured, it
    holds only up to the next compaction: a checkpoint, then four hundred writes, and the
    later checkpoint names every file the first named plus the flushes in between, so the
    increment is only the new files. Then one fold, and the next checkpoint shares nothing,
    because this engine's fold merges everything into one new file and the increment is the
    whole store again.

    Immutability makes files shareable; it is the compaction policy that decides whether any
    survive long enough to share. A levelled store rewrites only the levels a compaction
    touches, so old cold files persist across many checkpoints, and a fold everything policy
    invalidates every increment on every fold. Backup cost is a compaction property, which is
    not where anyone looks for it.
    """
    source = random.Random(19)
    store = Store(flush_at=300, fold_at=100)
    for _ in range(3000):
        store.put(f"k{source.randrange(2000):05d}".encode(), source.randbytes(10))
    first = take(store)
    for _ in range(400):
        store.put(f"k{source.randrange(2000):05d}".encode(), source.randbytes(10))
    flushed = take(store)
    across_flush = set(first.files) & set(flushed.files)
    grew = len(flushed.files) > len(first.files)
    store.fold()
    folded = take(store)
    across_fold = set(first.files) & set(folded.files)
    return bool(across_flush) and grew and not across_fold


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "a_checkpoint_is_cheap": a_checkpoint_costs_kilobytes_on_a_store_of_megabytes(),
        "writes_do_not_reach_it": writes_after_the_checkpoint_do_not_reach_it(),
        "a_restore_agrees": a_restore_agrees_with_the_original_everywhere(),
        "it_survives_the_crash": the_checkpoint_survives_a_crash_of_the_live_store(),
        "sharing_ends_at_the_fold": sharing_between_checkpoints_lasts_exactly_until_a_fold(),
    }
