"""Offset against cursor pagination: deep pages, drifting feeds, counted rows.

Page twenty of a listing can be reached two ways: skip four hundred rows
and take twenty, or remember where page nineteen ended and scan from
there. The skip is easy and the remember is honest, and the difference
is measured here twice: once in rows touched as pages deepen, once in
what each strategy shows when writers insert rows mid-pagination.
"""

from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

PAGE = 20
ROWS = 2000


@dataclass
class Listing:
    rows: list[bytes] = field(default_factory=list)
    touched: int = 0

    def insert(self, row: bytes) -> None:
        at = 0
        while at < len(self.rows) and self.rows[at] < row:
            at += 1
        self.rows.insert(at, row)

    def by_offset(self, offset: int) -> list[bytes]:
        self.touched += min(offset + PAGE, len(self.rows))
        return self.rows[offset : offset + PAGE]

    def by_cursor(self, after: bytes | None) -> list[bytes]:
        start = 0
        if after is not None:
            low, high = 0, len(self.rows)
            while low < high:
                middle = (low + high) // 2
                if self.rows[middle] <= after:
                    low = middle + 1
                else:
                    high = middle
            start = low
        self.touched += PAGE
        return self.rows[start : start + PAGE]


def _filled(seed: int) -> Listing:
    source = random.Random(seed)
    listing = Listing()
    listing.rows = sorted(f"row:{source.randrange(10**9):09d}".encode() for _ in range(ROWS))
    return listing


def walk_by_offset(listing: Listing, pages: int) -> list[bytes]:
    seen = []
    for page in range(pages):
        seen.extend(listing.by_offset(page * PAGE))
    return seen


def walk_by_cursor(listing: Listing, pages: int) -> list[bytes]:
    seen: list[bytes] = []
    cursor = None
    for _ in range(pages):
        got = listing.by_cursor(cursor)
        if not got:
            break
        seen.extend(got)
        cursor = got[-1]
    return seen


@functools.cache
def the_offset_walk_touches_fifty_times_the_data() -> bool:
    """Reading all 100 pages by offset touches 101000 rows; by cursor, 2000.

    Each offset page rescans everything before it, so the full walk sums
    to n squared over two. The cursor walk touches each row once. Same
    listing, same pages, fifty times the work.
    """
    by_offset = _filled(3)
    walk_by_offset(by_offset, ROWS // PAGE)
    by_cursor = _filled(3)
    walk_by_cursor(by_cursor, ROWS // PAGE)
    return by_offset.touched == 101000 and by_cursor.touched == ROWS


@functools.cache
def one_deep_page_pays_for_eighty() -> bool:
    """Page eighty alone touches 1620 rows to return 20.

    The deep page is the expensive one even in isolation: offset 1600
    walks past everything it skips. This is why offset pagination gets
    slower the further the user scrolls, and why crawlers that walk to
    page four hundred take databases down.
    """
    listing = _filled(3)
    got = listing.by_offset(80 * PAGE)
    return listing.touched == 1620 and len(got) == PAGE


@functools.cache
def front_inserts_make_offset_pages_stutter() -> bool:
    """Eight inserts per page boundary: offset repeats 32 rows of 100.

    Rows inserted before the current position shift the listing under
    the reader, and the next offset window re-serves the tail of the
    last one. A busy feed read by offset shows a third of every page
    twice. The cursor walk over the same storm repeats nothing.
    """
    by_offset = _filled(3)
    seen: list[bytes] = []
    fresh = 0
    for page in range(5):
        seen.extend(by_offset.by_offset(page * PAGE))
        for _ in range(8):
            by_offset.insert(f"row:000000{fresh:03d}".encode())
            fresh += 1
    duplicated = len(seen) - len(set(seen))
    by_cursor = _filled(3)
    seen_by_cursor: list[bytes] = []
    cursor = None
    fresh = 0
    for _ in range(5):
        got = by_cursor.by_cursor(cursor)
        seen_by_cursor.extend(got)
        cursor = got[-1]
        for _ in range(8):
            by_cursor.insert(f"row:000000{fresh:03d}".encode())
            fresh += 1
    return duplicated == 32 and len(seen_by_cursor) == len(set(seen_by_cursor))


@functools.cache
def the_cursor_page_costs_the_page() -> bool:
    """Every cursor page touches exactly PAGE rows at any depth.

    The binary search finds the resume point without walking to it, so
    page one and page ninety cost the same twenty touches. Depth stops
    being a cost dimension entirely.
    """
    listing = _filled(3)
    listing.by_cursor(None)
    shallow = listing.touched
    listing.touched = 0
    deep_cursor = listing.rows[1800]
    listing.by_cursor(deep_cursor)
    return shallow == PAGE and listing.touched == PAGE


@functools.cache
def summarise() -> dict:
    return {
        "module": "store.pagination",
        "the_offset_walk_touches_fifty_times_the_data": (
            the_offset_walk_touches_fifty_times_the_data()
        ),
        "one_deep_page_pays_for_eighty": one_deep_page_pays_for_eighty(),
        "front_inserts_make_offset_pages_stutter": (
            front_inserts_make_offset_pages_stutter()
        ),
        "the_cursor_page_costs_the_page": the_cursor_page_costs_the_page(),
    }
