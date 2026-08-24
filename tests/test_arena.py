from __future__ import annotations

import pytest

from store import arena as mod
from store.arena import Arena, FreeList
from store.errors import ConfigError, TooLarge


class TestArena:
    def test_a_zero_chunk_is_refused(self):
        with pytest.raises(ConfigError):
            Arena(chunk_size=0)

    def test_a_zero_allocation_is_refused(self):
        with pytest.raises(ConfigError):
            Arena().take(0)

    def test_an_oversized_allocation_is_refused(self):
        with pytest.raises(TooLarge):
            Arena(chunk_size=64).take(100)

    def test_allocations_bump_within_a_chunk(self):
        arena = Arena()
        assert arena.take(10) == 0 and arena.take(10) == 10

    def test_a_full_chunk_opens_a_new_one(self):
        arena = Arena(chunk_size=64)
        arena.take(60)
        arena.take(10)
        assert arena.chunks == 2

    def test_the_footprint_counts_whole_chunks(self):
        arena = Arena(chunk_size=64)
        arena.take(10)
        assert arena.footprint == 64

    def test_reset_returns_the_footprint(self):
        arena = Arena(chunk_size=64)
        arena.take(60)
        arena.take(10)
        assert arena.reset() == 128

    def test_reset_starts_over(self):
        arena = Arena()
        arena.take(100)
        arena.reset()
        assert arena.take(10) == 0

    def test_waste_is_zero_when_chunks_fill_exactly(self):
        arena = Arena(chunk_size=64)
        for _ in range(8):
            arena.take(64)
        assert arena.waste == 0.0


class TestFreeList:
    def test_a_zero_heap_is_refused(self):
        with pytest.raises(ConfigError):
            FreeList(heap_size=0)

    def test_a_take_returns_an_offset(self):
        assert FreeList(heap_size=64).take(16) == 0

    def test_takes_advance_through_the_span(self):
        heap = FreeList(heap_size=64)
        heap.take(16)
        assert heap.take(16) == 16

    def test_an_impossible_take_raises(self):
        heap = FreeList(heap_size=64)
        with pytest.raises(TooLarge):
            heap.take(100)

    def test_failures_are_counted(self):
        heap = FreeList(heap_size=64)
        with pytest.raises(TooLarge):
            heap.take(100)
        assert heap.failures == 1

    def test_a_give_makes_the_space_reusable(self):
        heap = FreeList(heap_size=64)
        start = heap.take(64)
        heap.give(start)
        assert heap.take(64) == 0

    def test_a_give_of_an_unknown_block_is_refused(self):
        with pytest.raises(ConfigError):
            FreeList(heap_size=64).give(7)

    def test_a_double_give_is_refused(self):
        heap = FreeList(heap_size=64)
        start = heap.take(16)
        heap.give(start)
        with pytest.raises(ConfigError):
            heap.give(start)

    def test_neighbours_coalesce(self):
        heap = FreeList(heap_size=64)
        first = heap.take(32)
        second = heap.take(32)
        heap.give(first)
        heap.give(second)
        assert len(heap.spans) == 1 and heap.largest_span == 64

    def test_free_bytes_track_the_ledger(self):
        heap = FreeList(heap_size=64)
        heap.take(16)
        assert heap.free_bytes == 48

    def test_fragmentation_of_a_whole_heap_is_zero(self):
        assert FreeList(heap_size=64).fragmentation == 0.0

    def test_fragmentation_of_an_empty_free_set_is_zero(self):
        heap = FreeList(heap_size=64)
        heap.take(64)
        assert heap.fragmentation == 0.0


class TestMeasurements:
    def test_arenas_waste_only_tails(self):
        assert mod.the_arena_wastes_only_chunk_tails()

    def test_reset_is_total(self):
        assert mod.the_arena_reset_is_constant_and_total()

    def test_churn_heals_the_board_shatters(self):
        assert mod.churn_barely_fragments_and_the_checkerboard_shatters()

    def test_coalescing_needs_neighbours(self):
        assert mod.coalescing_heals_what_ordered_frees_allow()

    def test_double_frees_are_refused(self):
        assert mod.a_double_free_is_refused()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5
