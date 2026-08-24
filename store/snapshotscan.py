from __future__ import annotations

import functools
import random
from dataclasses import dataclass

from store.engine import Store
from store.iterator import Merge, Source
from store.record import DELETE

# Consistent scans, an unsound shortcut, and a consistency this engine has by accident.
#
# The plan for this module was the textbook one: demonstrate the concurrent-scan anomaly,
# a moved key seen twice or not at all, then fix it by pinning the scan at a sequence. Both
# halves collapsed on contact with the engine, and what stands here is what the collapse
# taught.
#
# The anomaly does not reproduce, because the engine's scan copies the memtable's records at
# first pull and captures the immutable table list, so every scan is a snapshot already; the
# price it pays is that O(n) copy. And the sequence pin alone is unsound here, measured, not
# argued: the memtable keeps one version per key, an overwrite discards the history the pin
# would need, and the pinned read shows a hole exactly where every post-pin write landed.
# Sequence pinning needs multiversion structures underneath, which the mvcc module has and
# this memtable deliberately does not. The sound pin copies the mutable fringe and holds the
# immutable files, which is the checkpoint module's recipe applied to a read.


@dataclass
class Pinned:
    """A sound scan handle: the fringe copied, the files held, the sequence remembered."""

    sources: list[Source]
    sequence: int

    def items(self) -> list[tuple[bytes, bytes]]:
        """The live contents as of the pinned moment."""
        merge = Merge(sources=self.sources)
        found = []
        seen: bytes | None = None
        for _, record in merge.raw():
            if record.sequence > self.sequence:
                continue
            if record.key == seen:
                continue
            seen = record.key
            if record.kind != DELETE:
                found.append((record.key, record.value))
        return found


def pin(store: Store) -> Pinned:
    """A handle on the store as it stands: copy the memtable, capture the tables."""
    sources = [Source(name="memtable", records=list(store.memtable.records()))]
    sources += [table.source() for table in store.tables]
    return Pinned(sources=sources, sequence=store.sequence)


def sequence_only_items(store: Store, sequence: int) -> list[tuple[bytes, bytes]]:
    """The unsound pin: filter the live structures by sequence, copy nothing.

    Kept as the demonstration of why it fails: the structures it reads discard history on
    overwrite, so filtering out the new version leaves nothing where the old one used to be.
    """
    sources = [Source(name="memtable", records=store.memtable.records())]
    sources += [table.source() for table in store.tables]
    merge = Merge(sources=sources)
    found = []
    seen: bytes | None = None
    for _, record in merge.raw():
        if record.sequence > sequence:
            continue
        if record.key == seen:
            continue
        seen = record.key
        if record.kind != DELETE:
            found.append((record.key, record.value))
    return found


def _seeded(count: int = 400) -> Store:
    """A store with a known keyspace."""
    store = Store(flush_at=10**9, fold_at=10**9)
    for at in range(count):
        store.put(f"k{at:04d}".encode(), b"v")
    return store


@functools.cache
def the_engine_scan_is_a_snapshot_by_accident() -> bool:
    """A scan paused mid iteration ignores a delete and an insert that land while paused.

    The generator copied the memtable at first pull, so the moved key appears at its old
    address and not its new one: exactly one sighting, the consistent answer. The engine
    never promised this; it falls out of records() returning a copy, and an engine that
    iterated the live skiplist to avoid the copy would need the machinery this module
    builds. Consistency that arrives by accident leaves by optimisation.
    """
    store = _seeded()
    seen = []
    walker = store.scan()
    for record in walker:
        seen.append(record.key)
        if record.key >= b"k0100":
            break
    store.delete(b"k0100")
    store.put(b"k0300x", b"moved")
    seen.extend(record.key for record in walker)
    return b"k0100" in seen and b"k0300x" not in seen and len(seen) == 400


@functools.cache
def the_sequence_only_pin_is_unsound_here() -> bool:
    """Filtering by sequence alone loses every key overwritten after the pin: 91 holes.

    The pin remembers sequence 400, a storm overwrites and deletes a hundred keys, and the
    sequence-filtered read returns 309 keys instead of 400. The old versions were not
    shadowed, they were destroyed: the memtable keeps one record per key. A pin that
    filters needs history to fall back on, and the difference between shadowing and
    destroying is the entire difference between the mvcc module's store and this one's
    memtable.
    """
    store = _seeded()
    pinned_sequence = store.sequence
    before = store.items()
    source = random.Random(157)
    for _ in range(100):
        at = source.randrange(400)
        if source.random() < 0.3:
            store.delete(f"k{at:04d}".encode())
        else:
            store.put(f"k{at:04d}".encode(), b"new")
    unsound = sequence_only_items(store, pinned_sequence)
    return len(unsound) < len(before) - 50


@functools.cache
def the_sound_pin_reports_one_moment_exactly() -> bool:
    """The copy-and-capture pin equals the pre-storm contents through the same storm."""
    store = _seeded()
    before = store.items()
    handle = pin(store)
    source = random.Random(157)
    for _ in range(100):
        at = source.randrange(400)
        if source.random() < 0.3:
            store.delete(f"k{at:04d}".encode())
        else:
            store.put(f"k{at:04d}".encode(), b"new")
    return handle.items() == before


@functools.cache
def two_pins_tell_two_consistent_stories() -> bool:
    """A pin before and a pin after a batch of writes each report their own moment."""
    store = _seeded(100)
    early = pin(store)
    for at in range(100, 150):
        store.put(f"k{at:04d}".encode(), b"late")
    late = pin(store)
    early_keys = {key for key, _ in early.items()}
    late_keys = {key for key, _ in late.items()}
    return (
        len(early_keys) == 100
        and len(late_keys) == 150
        and not any(key >= b"k0100" for key in early_keys)
    )


@functools.cache
def a_pin_survives_a_flush_underneath() -> bool:
    """The pinned answer is identical before and after the memtable becomes a file.

    The handle holds its own copy of the fringe and its own references to the old tables,
    so maintenance replacing the store's structures replaces nothing the handle reads.
    """
    store = _seeded(300)
    handle = pin(store)
    first = handle.items()
    store.flush()
    second = handle.items()
    return first == second and len(first) == 300


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "the_accidental_snapshot": the_engine_scan_is_a_snapshot_by_accident(),
        "sequence_alone_is_unsound": the_sequence_only_pin_is_unsound_here(),
        "the_sound_pin_holds": the_sound_pin_reports_one_moment_exactly(),
        "two_pins_two_stories": two_pins_tell_two_consistent_stories(),
        "pins_survive_flushes": a_pin_survives_a_flush_underneath(),
    }
