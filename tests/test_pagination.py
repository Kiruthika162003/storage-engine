from __future__ import annotations

import pytest

from store.pagination import (
    PAGE,
    ROWS,
    Listing,
    _filled,
    front_inserts_make_offset_pages_stutter,
    one_deep_page_pays_for_eighty,
    summarise,
    the_cursor_page_costs_the_page,
    the_offset_walk_touches_fifty_times_the_data,
    walk_by_cursor,
    walk_by_offset,
)


class TestListing:
    def test_insert_keeps_the_rows_sorted(self):
        listing = Listing()
        for row in (b"c", b"a", b"b"):
            listing.insert(row)
        assert listing.rows == [b"a", b"b", b"c"]

    def test_by_offset_returns_the_window(self):
        listing = _filled(3)
        assert listing.by_offset(0) == listing.rows[:PAGE]
        assert listing.by_offset(PAGE) == listing.rows[PAGE : 2 * PAGE]

    def test_by_cursor_resumes_after_the_cursor(self):
        listing = _filled(3)
        first = listing.by_cursor(None)
        second = listing.by_cursor(first[-1])
        assert second == listing.rows[PAGE : 2 * PAGE]

    def test_a_cursor_past_the_end_returns_empty(self):
        listing = _filled(3)
        assert listing.by_cursor(listing.rows[-1]) == []

    def test_the_filled_listing_is_deterministic(self):
        assert _filled(3).rows == _filled(3).rows


class TestWalks:
    def test_both_walks_see_the_same_rows(self):
        by_offset = walk_by_offset(_filled(3), ROWS // PAGE)
        by_cursor = walk_by_cursor(_filled(3), ROWS // PAGE)
        assert by_offset == by_cursor

    def test_the_cursor_walk_stops_at_the_end(self):
        listing = _filled(3)
        seen = walk_by_cursor(listing, 10**6)
        assert len(seen) == ROWS

    def test_offset_touch_counts_accumulate(self):
        listing = _filled(3)
        listing.by_offset(0)
        listing.by_offset(PAGE)
        assert listing.touched == PAGE + 2 * PAGE


class TestClaims:
    @pytest.mark.parametrize(
        "claim",
        [
            the_offset_walk_touches_fifty_times_the_data,
            one_deep_page_pays_for_eighty,
            front_inserts_make_offset_pages_stutter,
            the_cursor_page_costs_the_page,
        ],
    )
    def test_claim_holds(self, claim):
        assert claim() is True

    def test_summary_is_all_true(self):
        told = summarise()
        assert all(value for name, value in told.items() if name != "module")
