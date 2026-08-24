"""Interleaved spatial keys: how many ranges a rectangle costs.

A sorted store has one dimension. Points have two. Concatenating x then y
sorts all of x before any of y matters, so a rectangle query scans every
x-stripe end to end. Interleaving the bits of x and y walks a Z through
the plane and keeps near things near, at the price of the Z's jumps. The
measurements count scanned keys against matching keys for both layouts.
"""

from __future__ import annotations

import functools
import random
from dataclasses import dataclass

BITS = 8
SIDE = 1 << BITS


def interleave(x: int, y: int) -> int:
    key = 0
    for bit in range(BITS):
        key |= ((x >> bit) & 1) << (2 * bit + 1)
        key |= ((y >> bit) & 1) << (2 * bit)
    return key


def deinterleave(key: int) -> tuple[int, int]:
    x = y = 0
    for bit in range(BITS):
        x |= ((key >> (2 * bit + 1)) & 1) << bit
        y |= ((key >> (2 * bit)) & 1) << bit
    return x, y


def concatenate(x: int, y: int) -> int:
    return (x << BITS) | y


@dataclass(frozen=True)
class Box:
    x_low: int
    x_high: int
    y_low: int
    y_high: int

    def holds(self, x: int, y: int) -> bool:
        return self.x_low <= x <= self.x_high and self.y_low <= y <= self.y_high

    def area(self) -> int:
        return (self.x_high - self.x_low + 1) * (self.y_high - self.y_low + 1)


def _points(seed: int, count: int = 5000) -> list[tuple[int, int]]:
    source = random.Random(seed)
    return [(source.randrange(SIDE), source.randrange(SIDE)) for _ in range(count)]


def scan_concatenated(points: list[tuple[int, int]], box: Box) -> tuple[int, int]:
    """Scan the concat layout between the box's corner keys; count touched."""
    low = concatenate(box.x_low, box.y_low)
    high = concatenate(box.x_high, box.y_high)
    ordered = sorted(concatenate(x, y) for x, y in points)
    touched = matched = 0
    for key in ordered:
        if low <= key <= high:
            touched += 1
            x, y = key >> BITS, key & (SIDE - 1)
            if box.holds(x, y):
                matched += 1
    return touched, matched


def scan_zorder(points: list[tuple[int, int]], box: Box) -> tuple[int, int]:
    """Scan the z layout between the box corners' z keys; count touched."""
    low = interleave(box.x_low, box.y_low)
    high = interleave(box.x_high, box.y_high)
    ordered = sorted(interleave(x, y) for x, y in points)
    touched = matched = 0
    for key in ordered:
        if low <= key <= high:
            touched += 1
            if box.holds(*deinterleave(key)):
                matched += 1
    return touched, matched


def cover(box: Box) -> list[tuple[int, int]]:
    """Decompose the box into aligned quads, each one contiguous in z."""
    ranges: list[tuple[int, int]] = []

    def descend(cell_x: int, cell_y: int, size: int) -> None:
        if (
            cell_x > box.x_high
            or cell_x + size - 1 < box.x_low
            or cell_y > box.y_high
            or cell_y + size - 1 < box.y_low
        ):
            return
        if (
            box.x_low <= cell_x
            and cell_x + size - 1 <= box.x_high
            and box.y_low <= cell_y
            and cell_y + size - 1 <= box.y_high
        ):
            low = interleave(cell_x, cell_y)
            ranges.append((low, low + size * size - 1))
            return
        half = size // 2
        for dx in (0, half):
            for dy in (0, half):
                descend(cell_x + dx, cell_y + dy, half)

    descend(0, 0, SIDE)
    return sorted(ranges)


def scan_covered(points: list[tuple[int, int]], box: Box) -> tuple[int, int, int]:
    """Touched, matched and range count when scanning the decomposition."""
    ranges = cover(box)
    ordered = sorted(interleave(x, y) for x, y in points)
    touched = matched = 0
    for key in ordered:
        for low, high in ranges:
            if low <= key <= high:
                touched += 1
                if box.holds(*deinterleave(key)):
                    matched += 1
                break
    return touched, matched, len(ranges)


@functools.cache
def the_corner_to_corner_z_range_is_a_trap() -> bool:
    """A mid-plane 32x32 box: naive z scans 2570 keys for 80 matches.

    Even the untuned concatenated layout scans only 608 for the same 80.
    The Z between the corner keys leaves the box at every quadrant seam
    it straddles, so interleaving alone, queried naively, is worse than
    not interleaving. The curve is only as good as the query planner.
    """
    points = _points(3)
    box = Box(100, 131, 100, 131)
    concat_touched, _ = scan_concatenated(points, box)
    z_touched, _ = scan_zorder(points, box)
    return z_touched > concat_touched * 4 and concat_touched == 608


@functools.cache
def an_aligned_quadrant_is_one_perfect_range() -> bool:
    """A 32-aligned 32x32 box scans 66 keys, all 66 matching, in one range.

    When the box is a quadtree cell the Z through it is contiguous:
    touched equals matched and one seek serves the whole query. The
    concatenated layout scans 581 keys for the same 66.
    """
    points = _points(3)
    box = Box(0, 31, 224, 255)
    z_touched, z_matched = scan_zorder(points, box)
    concat_touched, _ = scan_concatenated(points, box)
    return z_touched == z_matched == 66 and concat_touched == 581


@functools.cache
def decomposition_makes_every_touch_a_match() -> bool:
    """Covering the trap box with aligned quads: 80 touched, 80 matched.

    The cure for the corner trap is not a better curve but a better
    query: split the box into 34 aligned quads, each contiguous in z, and
    scan those. Nothing outside the box is ever touched, for any box.
    """
    points = _points(3)
    for box in (Box(100, 131, 100, 131), Box(64, 191, 64, 191)):
        touched, matched, _ = scan_covered(points, box)
        if touched != matched:
            return False
    return scan_covered(points, Box(100, 131, 100, 131))[2] == 34


@functools.cache
def alignment_is_worth_34_seeks() -> bool:
    """The same 32x32 box costs 1 range at offset 96 and 34 at offset 100.

    Four pixels of misalignment explode one quad into 34 because no large
    cell fits inside the shifted box. Range count, which is seek count, is
    set by where the box sits, not how big it is: the 128x128 box costs 4.
    """
    aligned = scan_covered(_points(3), Box(96, 127, 96, 127))[2]
    shifted = scan_covered(_points(3), Box(100, 131, 100, 131))[2]
    big = scan_covered(_points(3), Box(64, 191, 64, 191))[2]
    return aligned == 1 and shifted == 34 and big == 4


@functools.cache
def summarise() -> dict:
    return {
        "module": "store.zorder",
        "the_corner_to_corner_z_range_is_a_trap": (
            the_corner_to_corner_z_range_is_a_trap()
        ),
        "an_aligned_quadrant_is_one_perfect_range": (
            an_aligned_quadrant_is_one_perfect_range()
        ),
        "decomposition_makes_every_touch_a_match": (
            decomposition_makes_every_touch_a_match()
        ),
        "alignment_is_worth_34_seeks": alignment_is_worth_34_seeks(),
    }
