from __future__ import annotations

import pytest

from store.overflow import (
    PAGE_BYTES,
    POINTER,
    Heap,
    an_eager_threshold_shatters_the_heap,
    build,
    spilling_the_tail_charges_only_its_readers,
    summarise,
    the_big_tenth_owns_the_scan,
    the_row_pages_hold_pointers_not_bodies,
)


class TestHeap:
    def test_a_small_value_inlines(self):
        heap = Heap(threshold=100)
        heap.add(0, 50)
        assert heap.overflow_pages == 0
        assert heap.point_read_pages(0) == 1

    def test_a_big_value_spills(self):
        heap = Heap(threshold=100)
        heap.add(0, 500)
        assert heap.overflow_pages == 1
        assert heap.point_read_pages(0) == 2

    def test_a_spilled_value_leaves_only_a_pointer(self):
        heap = Heap(threshold=100)
        heap.add(0, 500)
        assert heap.pages == [POINTER]

    def test_pages_split_at_the_boundary(self):
        heap = Heap(threshold=PAGE_BYTES)
        heap.add(0, PAGE_BYTES - 10)
        heap.add(1, 100)
        assert heap.scan_pages() == 2

    def test_total_pages_counts_both_kinds(self):
        heap = Heap(threshold=100)
        heap.add(0, 50)
        heap.add(1, 500)
        assert heap.total_pages() == 2

    def test_where_remembers_the_page(self):
        heap = Heap(threshold=100)
        heap.add(0, 50)
        assert heap.where[0] == (0, False)


class TestBuild:
    def test_build_is_deterministic(self):
        assert build(1000).pages == build(1000).pages

    def test_a_lower_threshold_never_grows_row_pages(self):
        assert build(100).scan_pages() <= build(1000).scan_pages()
        assert build(1000).scan_pages() <= build(4096).scan_pages()

    def test_every_row_is_placed(self):
        assert len(build(1000).where) == 2000


class TestClaims:
    @pytest.mark.parametrize(
        "claim",
        [
            the_big_tenth_owns_the_scan,
            spilling_the_tail_charges_only_its_readers,
            an_eager_threshold_shatters_the_heap,
            the_row_pages_hold_pointers_not_bodies,
        ],
    )
    def test_claim_holds(self, claim):
        assert claim() is True

    def test_summary_is_all_true(self):
        told = summarise()
        assert all(value for name, value in told.items() if name != "module")
