from __future__ import annotations

import pytest

from store.doublecache import (
    BUDGET,
    Lru,
    Stack,
    duplication_costs_eighty_five_percent_more_disk,
    exclusion_makes_two_caches_one,
    single,
    split,
    summarise,
    the_inclusive_split_holds_the_hot_set_twice,
    the_lower_layer_earns_its_keep_only_under_exclusion,
)


class TestLru:
    def test_a_miss_then_a_hit(self):
        cache = Lru(size=2)
        assert not cache.get(1)
        cache.admit(1)
        assert cache.get(1)

    def test_admit_returns_the_evicted_block(self):
        cache = Lru(size=1)
        assert cache.admit(1) is None
        assert cache.admit(2) == 1

    def test_recency_decides_the_victim(self):
        cache = Lru(size=2)
        cache.admit(1)
        cache.admit(2)
        cache.get(1)
        assert cache.admit(3) == 2

    def test_a_zero_size_cache_admits_nothing(self):
        cache = Lru(size=0)
        cache.admit(1)
        assert not cache.get(1)

    def test_drop_removes_without_counting(self):
        cache = Lru(size=2)
        cache.admit(1)
        cache.drop(1)
        cache.drop(99)
        assert not cache.get(1)


class TestStack:
    def test_a_disk_read_is_counted_once(self):
        stack = Stack(upper=Lru(size=4), lower=Lru(size=4), exclusive=False)
        stack.read(7)
        stack.read(7)
        assert stack.disk_reads == 1

    def test_inclusive_admits_to_both_layers(self):
        stack = Stack(upper=Lru(size=4), lower=Lru(size=4), exclusive=False)
        stack.read(7)
        assert stack.duplicated() == 1

    def test_exclusive_keeps_the_layers_disjoint(self):
        stack = Stack(upper=Lru(size=2), lower=Lru(size=2), exclusive=True)
        for block in range(5):
            stack.read(block)
        assert stack.duplicated() == 0

    def test_a_demoted_block_is_served_from_below(self):
        stack = Stack(upper=Lru(size=1), lower=Lru(size=2), exclusive=True)
        stack.read(1)
        stack.read(2)
        stack.read(1)
        assert stack.disk_reads == 2
        assert stack.lower.hits == 1

    def test_the_single_stack_never_uses_the_lower(self):
        stack = single(3)
        assert stack.lower.hits == 0 and stack.lower.misses > 0

    def test_split_halves_the_budget(self):
        stack = split(3, exclusive=True)
        assert stack.upper.size == BUDGET // 2
        assert stack.lower.size == BUDGET // 2


class TestClaims:
    @pytest.mark.parametrize(
        "claim",
        [
            the_inclusive_split_holds_the_hot_set_twice,
            duplication_costs_eighty_five_percent_more_disk,
            exclusion_makes_two_caches_one,
            the_lower_layer_earns_its_keep_only_under_exclusion,
        ],
    )
    def test_claim_holds(self, claim):
        assert claim() is True

    def test_summary_is_all_true(self):
        told = summarise()
        assert all(value for name, value in told.items() if name != "module")
