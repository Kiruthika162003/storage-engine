"""Overflow pages: where a large value should live depends on who asks.

A row page holds slots up to a threshold; a value past it moves to an
overflow page and leaves a pointer behind. Inlining everything bloats
the pages every scan must read. Spilling everything makes every point
read of a big value a two page trip. The threshold is a bet on the
question mix, and both costs are counted here against the same rows.
"""

from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

PAGE_BYTES = 4096
POINTER = 16


@dataclass
class Heap:
    threshold: int
    pages: list[int] = field(default_factory=lambda: [0])
    where: dict[int, tuple[int, bool]] = field(default_factory=dict)
    overflow_pages: int = 0

    def add(self, row: int, size: int) -> None:
        spilled = size > self.threshold
        cost = POINTER if spilled else size
        if self.pages[-1] + cost > PAGE_BYTES:
            self.pages.append(0)
        self.pages[-1] += cost
        if spilled:
            self.overflow_pages += 1
        self.where[row] = (len(self.pages) - 1, spilled)

    def scan_pages(self) -> int:
        return len(self.pages)

    def point_read_pages(self, row: int) -> int:
        _, spilled = self.where[row]
        return 2 if spilled else 1

    def total_pages(self) -> int:
        return len(self.pages) + self.overflow_pages


def _rows(seed: int, count: int = 2000) -> list[int]:
    source = random.Random(seed)
    sizes = []
    for _ in range(count):
        if source.random() < 0.1:
            sizes.append(source.randrange(1500, 3500))
        else:
            sizes.append(source.randrange(40, 200))
    return sizes


def build(threshold: int, seed: int = 13) -> Heap:
    heap = Heap(threshold=threshold)
    for row, size in enumerate(_rows(seed)):
        heap.add(row, size)
    return heap


@functools.cache
def the_big_tenth_owns_the_scan() -> bool:
    """216 of 2000 rows are big, hold 72 percent of the bytes, and
    inlining them makes every scan read 228 pages instead of 54.

    A scan that wants the small columns still pages through every inlined
    blob. The 4.2x scan tax is set by the byte share of the big rows, not
    their count.
    """
    inlined = build(4096)
    spilled = build(1000)
    sizes = _rows(13)
    big_bytes = sum(size for size in sizes if size > 1000)
    return (
        inlined.scan_pages() == 228
        and spilled.scan_pages() == 54
        and big_bytes / sum(sizes) > 0.7
    )


@functools.cache
def spilling_the_tail_charges_only_its_readers() -> bool:
    """With the big tenth spilled, 216 point reads in 2000 pay a second page.

    The scan gets its 4.2x and the bill lands precisely on the reads
    that ask for a big value: 10.8 percent of point reads cost two pages,
    the rest still cost one, and total storage grows 18 percent for the
    pointers and page remainders.
    """
    heap = build(1000)
    doubled = sum(
        1 for row in range(2000) if heap.point_read_pages(row) == 2
    )
    return doubled == 216 and heap.total_pages() == 270


@functools.cache
def an_eager_threshold_shatters_the_heap() -> bool:
    """Threshold 100 spills 1315 rows into 1315 pages: 5.8x the storage.

    Every spilled value occupies its own overflow page no matter how
    small, so spilling the medium rows buys a 54 to 17 scan improvement
    at the cost of a thousand nearly empty pages and 1315 doubled reads.
    The threshold's job is to catch the tail, not the body.
    """
    eager = build(100)
    modest = build(1000)
    return (
        eager.total_pages() == 1332
        and eager.total_pages() > modest.total_pages() * 4
        and eager.scan_pages() == 17
    )


@functools.cache
def the_row_pages_hold_pointers_not_bodies() -> bool:
    """The spilled heap's row pages hold small bytes plus 16 per pointer.

    Row page bytes for threshold 1000 equal the small rows' 213074 bytes
    plus 216 pointers of 16, which is what makes the scan cheap: the big
    bodies are simply somewhere else.
    """
    heap = build(1000)
    sizes = _rows(13)
    small_bytes = sum(size for size in sizes if size <= 1000)
    stored = sum(heap.pages)
    return stored == small_bytes + 216 * POINTER


@functools.cache
def summarise() -> dict:
    return {
        "module": "store.overflow",
        "the_big_tenth_owns_the_scan": the_big_tenth_owns_the_scan(),
        "spilling_the_tail_charges_only_its_readers": (
            spilling_the_tail_charges_only_its_readers()
        ),
        "an_eager_threshold_shatters_the_heap": an_eager_threshold_shatters_the_heap(),
        "the_row_pages_hold_pointers_not_bodies": (
            the_row_pages_hold_pointers_not_bodies()
        ),
    }
