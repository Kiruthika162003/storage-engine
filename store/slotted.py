from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

from store.errors import ConfigError, NotFound, TooLarge

# The slotted page: stable ids for moving bytes.
#
# The B-tree module's pages held Python lists; a real page is one fixed buffer that must
# hand out record ids which survive everything the page does internally. The slotted layout
# is the universal answer: a slot directory grows from the front, record bytes grow from
# the back, a record id is a slot index, and the slot holds the record's current offset.
# Deletes leave holes, updates that grow move the record, and neither changes the id,
# because everything external points through the slot. The page compacts by sliding live
# records together and rewriting slot offsets, ids untouched, which is the whole point
# demonstrated: identity and placement are different problems, and the slot is the
# indirection that separates them.

PAGE_BYTES = 4096
SLOT_BYTES = 4


@dataclass
class Page:
    """One fixed buffer with a slot directory."""

    payload: bytearray = field(default_factory=lambda: bytearray(PAGE_BYTES))
    slots: list[tuple[int, int]] = field(default_factory=list)
    free_start: int = field(default=0)
    free_end: int = field(default=PAGE_BYTES)
    compactions: int = field(default=0)

    def _directory_bytes(self) -> int:
        return len(self.slots) * SLOT_BYTES

    @property
    def contiguous_free(self) -> int:
        """Bytes available without compacting."""
        return self.free_end - self._directory_bytes()

    @property
    def live_bytes(self) -> int:
        """Bytes live records occupy."""
        return sum(length for _, length in self.slots if length > 0)

    @property
    def reclaimable(self) -> int:
        """Bytes a compaction would recover."""
        return (PAGE_BYTES - self._directory_bytes()) - self.live_bytes - self.contiguous_free

    def insert(self, record: bytes) -> int:
        """A record in, a stable slot id out, compacting once if the hole space allows."""
        if not record:
            raise ConfigError("an empty record stores nothing")
        needed = len(record) + SLOT_BYTES
        if needed > self.contiguous_free:
            if needed <= self.contiguous_free + self.reclaimable:
                self.compact()
            else:
                raise TooLarge(f"{len(record)} bytes do not fit")
        start = self.free_end - len(record)
        self.payload[start : self.free_end] = record
        self.free_end = start
        self.slots.append((start, len(record)))
        return len(self.slots) - 1

    def read(self, slot: int) -> bytes:
        """The record behind an id."""
        if not 0 <= slot < len(self.slots):
            raise NotFound(f"slot {slot} does not exist")
        start, length = self.slots[slot]
        if length == 0:
            raise NotFound(f"slot {slot} is deleted")
        return bytes(self.payload[start : start + length])

    def delete(self, slot: int) -> None:
        """The record goes, the id remains, permanently dead."""
        self.read(slot)
        start, _ = self.slots[slot]
        self.slots[slot] = (start, 0)

    def update(self, slot: int, record: bytes) -> None:
        """New bytes behind the same id, moving if they no longer fit in place."""
        old = self.read(slot)
        start, _ = self.slots[slot]
        if len(record) <= len(old):
            self.payload[start : start + len(record)] = record
            self.slots[slot] = (start, len(record))
            return
        self.slots[slot] = (start, 0)
        if len(record) > self.contiguous_free:
            if len(record) <= self.contiguous_free + self.reclaimable:
                self.compact()
            else:
                self.slots[slot] = (start, len(old))
                self.payload[start : start + len(old)] = old
                raise TooLarge(f"{len(record)} bytes do not fit")
        new_start = self.free_end - len(record)
        self.payload[new_start : self.free_end] = record
        self.free_end = new_start
        self.slots[slot] = (new_start, len(record))

    def compact(self) -> int:
        """Slide live records to the back, ids untouched, holes gone."""
        records = []
        for slot, (start, length) in enumerate(self.slots):
            if length > 0:
                records.append((slot, bytes(self.payload[start : start + length])))
        write_at = PAGE_BYTES
        for slot, record in sorted(records, key=lambda pair: -len(pair[1])):
            write_at -= len(record)
            self.payload[write_at : write_at + len(record)] = record
            self.slots[slot] = (write_at, len(record))
        recovered = write_at - self.free_end
        self.free_end = write_at
        self.compactions += 1
        return recovered


@functools.cache
def ids_survive_every_internal_upheaval() -> bool:
    """Deletes, growing updates and compactions move bytes; no id ever changes meaning.

    Forty records inserted, a third deleted, a third grown past their old homes, then a
    compaction, and every surviving id still reads its own record. The slot directory is
    the indirection that makes it true, and the claim is the module's reason to exist:
    identity and placement are different problems.
    """
    source = random.Random(281)
    page = Page()
    contents: dict[int, bytes] = {}
    for at in range(40):
        record = f"rec-{at:03d}-".encode() + source.randbytes(20)
        contents[page.insert(record)] = record
    doomed = sorted(contents)[::3]
    for slot in doomed:
        page.delete(slot)
        del contents[slot]
    for slot in sorted(contents)[::3]:
        grown = contents[slot] + source.randbytes(40)
        page.update(slot, grown)
        contents[slot] = grown
    page.compact()
    return all(page.read(slot) == record for slot, record in contents.items())


@functools.cache
def compaction_recovers_exactly_the_holes() -> bool:
    """The reclaimable meter equals what compaction returns, byte for byte.

    The accounting invariant: contiguous free plus reclaimable plus live plus directory is
    the page, before and after, and the compaction's return value is the reclaimable meter
    read just before it ran.
    """
    page = Page()
    for at in range(30):
        page.insert(bytes([at]) * 50)
    for slot in range(0, 30, 2):
        page.delete(slot)
    promised = page.reclaimable
    recovered = page.compact()
    return promised == recovered and page.reclaimable == 0


@functools.cache
def a_full_page_refuses_and_a_holey_page_compacts_first() -> bool:
    """The same insert fails on a truly full page and lands on one that is full of holes.

    The distinction is the slotted layout's operating point: contiguous free space says no,
    total free space says yes, and the insert path compacts on demand rather than on a
    schedule, spending the slide only when a request actually needs the holes.
    """
    solid = Page()
    while True:
        try:
            solid.insert(b"x" * 100)
        except TooLarge:
            break
    try:
        solid.insert(b"y" * 100)
        return False
    except TooLarge:
        pass
    holey = Page()
    slots = []
    while True:
        try:
            slots.append(holey.insert(b"x" * 100))
        except TooLarge:
            break
    for slot in slots[::2]:
        holey.delete(slot)
    before = holey.compactions
    holey.insert(b"y" * 100)
    return holey.compactions == before + 1


@functools.cache
def shrinking_updates_stay_in_place_and_growing_ones_move() -> bool:
    """An update that fits rewrites in place; one that grows relocates; the id is oblivious.

    Measured through the slot offsets: the shrink keeps its offset, the growth changes it,
    and both read back exactly. In-place shrink is why variable length updates do not
    always fragment, and the move is why they sometimes do.
    """
    page = Page()
    slot = page.insert(b"a" * 100)
    offset_before = page.slots[slot][0]
    page.update(slot, b"b" * 60)
    shrunk_offset = page.slots[slot][0]
    page.update(slot, b"c" * 300)
    grown_offset = page.slots[slot][0]
    return (
        shrunk_offset == offset_before
        and grown_offset != offset_before
        and page.read(slot) == b"c" * 300
    )


@functools.cache
def a_deleted_id_stays_dead() -> bool:
    """Reading or updating a deleted slot raises forever; the id is never reissued.

    Reuse would let a stale external pointer read somebody else's record, the classic
    dangling reference, so the slot is a tombstone at the cost of four directory bytes.
    """
    page = Page()
    slot = page.insert(b"gone")
    page.delete(slot)
    for act in (lambda: page.read(slot), lambda: page.update(slot, b"x")):
        try:
            act()
            return False
        except NotFound:
            continue
    fresh = page.insert(b"new")
    return fresh != slot


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "ids_survive_upheaval": ids_survive_every_internal_upheaval(),
        "compaction_matches_the_meter": compaction_recovers_exactly_the_holes(),
        "full_refuses_holey_compacts": a_full_page_refuses_and_a_holey_page_compacts_first(),
        "shrink_stays_growth_moves": shrinking_updates_stay_in_place_and_growing_ones_move(),
        "dead_ids_stay_dead": a_deleted_id_stays_dead(),
    }
