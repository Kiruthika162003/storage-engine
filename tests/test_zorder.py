from __future__ import annotations

import pytest

from store.zorder import (
    SIDE,
    Box,
    alignment_is_worth_34_seeks,
    an_aligned_quadrant_is_one_perfect_range,
    concatenate,
    cover,
    decomposition_makes_every_touch_a_match,
    deinterleave,
    interleave,
    scan_covered,
    summarise,
    the_corner_to_corner_z_range_is_a_trap,
)


class TestInterleave:
    def test_round_trip_everywhere_sampled(self):
        for x in range(0, SIDE, 37):
            for y in range(0, SIDE, 41):
                assert deinterleave(interleave(x, y)) == (x, y)

    def test_the_origin_is_zero(self):
        assert interleave(0, 0) == 0

    def test_neighbours_in_a_quad_are_adjacent_keys(self):
        assert sorted(
            interleave(x, y) for x in (0, 1) for y in (0, 1)
        ) == [0, 1, 2, 3]

    def test_concatenate_orders_x_first(self):
        assert concatenate(1, 0) > concatenate(0, SIDE - 1)


class TestBox:
    def test_holds_its_corners(self):
        box = Box(2, 5, 3, 7)
        assert box.holds(2, 3) and box.holds(5, 7)

    def test_excludes_outside(self):
        box = Box(2, 5, 3, 7)
        assert not box.holds(1, 4) and not box.holds(3, 8)

    def test_area_counts_inclusive(self):
        assert Box(0, 3, 0, 1).area() == 8


class TestCover:
    def test_the_whole_plane_is_one_range(self):
        ranges = cover(Box(0, SIDE - 1, 0, SIDE - 1))
        assert ranges == [(0, SIDE * SIDE - 1)]

    def test_an_aligned_cell_is_one_range(self):
        ranges = cover(Box(0, 31, 0, 31))
        assert len(ranges) == 1
        low, high = ranges[0]
        assert high - low + 1 == 32 * 32

    def test_a_single_point_is_one_range_of_one(self):
        ranges = cover(Box(5, 5, 9, 9))
        assert ranges == [(interleave(5, 9), interleave(5, 9))]

    def test_ranges_are_disjoint_and_sorted(self):
        ranges = cover(Box(100, 131, 100, 131))
        for (low_a, high_a), (low_b, _) in zip(ranges, ranges[1:]):
            assert low_a <= high_a < low_b

    def test_the_cover_area_equals_the_box_area(self):
        box = Box(100, 131, 100, 131)
        covered = sum(high - low + 1 for low, high in cover(box))
        assert covered == box.area()

    def test_scan_covered_finds_exactly_the_inside(self):
        points = [(x, y) for x in range(90, 140, 7) for y in range(90, 140, 7)]
        box = Box(100, 131, 100, 131)
        touched, matched, _ = scan_covered(points, box)
        inside = sum(1 for x, y in points if box.holds(x, y))
        assert touched == matched == inside


class TestClaims:
    @pytest.mark.parametrize(
        "claim",
        [
            the_corner_to_corner_z_range_is_a_trap,
            an_aligned_quadrant_is_one_perfect_range,
            decomposition_makes_every_touch_a_match,
            alignment_is_worth_34_seeks,
        ],
    )
    def test_claim_holds(self, claim):
        assert claim() is True

    def test_summary_is_all_true(self):
        told = summarise()
        assert all(value for name, value in told.items() if name != "module")
