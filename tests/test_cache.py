from __future__ import annotations

import pytest

from store import cache as mod
from store.cache import (
    CAPACITY,
    POLICIES,
    Chance,
    Clock,
    Frequent,
    Recent,
    Reference,
    Stats,
    block,
    measure,
    run,
)
from store.errors import ConfigError


def fill(cache, count: int, start: int = 0) -> None:
    for at in range(start, start + count):
        cache.put(at, block(at))


class TestStats:
    def test_a_fresh_stats_has_no_lookups(self):
        assert Stats().lookups == 0

    def test_the_rate_survives_no_lookups(self):
        assert Stats().rate == 0.0

    def test_the_rate_is_hits_over_lookups(self):
        assert Stats(hits=3, misses=1).rate == 0.75

    def test_the_lookups_add_hits_and_misses(self):
        assert Stats(hits=3, misses=2).lookups == 5

    def test_as_dict_carries_every_field(self):
        made = Stats(hits=1, misses=2, evictions=3).as_dict()
        assert set(made) == {"hits", "misses", "lookups", "rate", "evictions"}


class TestEveryPolicy:
    @pytest.fixture(params=POLICIES, ids=lambda one: one.__name__)
    def cache(self, request):
        return request.param(capacity=8)

    def test_a_zero_capacity_is_refused(self, cache):
        with pytest.raises(ConfigError):
            type(cache)(capacity=0)

    def test_a_fresh_cache_is_empty(self, cache):
        assert len(cache) == 0

    def test_a_miss_returns_nothing(self, cache):
        assert cache.get(1) is None

    def test_a_miss_is_counted(self, cache):
        cache.get(1)
        assert cache.stats.misses == 1

    def test_a_put_then_get_hits(self, cache):
        cache.put(1, b"x")
        assert cache.get(1) == b"x"

    def test_a_hit_is_counted(self, cache):
        cache.put(1, b"x")
        cache.get(1)
        assert cache.stats.hits == 1

    def test_the_capacity_is_respected(self, cache):
        fill(cache, 20)
        assert len(cache) == 8

    def test_an_eviction_is_counted(self, cache):
        fill(cache, 9)
        assert cache.stats.evictions == 1

    def test_a_repeated_put_does_not_grow(self, cache):
        cache.put(1, b"x")
        cache.put(1, b"y")
        assert len(cache) == 1

    def test_a_repeated_put_updates_the_value(self, cache):
        cache.put(1, b"x")
        cache.put(1, b"y")
        assert cache.get(1) == b"y"

    def test_a_repeated_put_never_evicts(self, cache):
        fill(cache, 8)
        cache.put(3, b"z")
        assert cache.stats.evictions == 0

    def test_the_name_is_the_class(self, cache):
        assert cache.name == type(cache).__name__.lower()

    def test_as_dict_names_the_policy(self, cache):
        assert cache.as_dict()["policy"] == cache.name

    def test_as_dict_counts_what_is_held(self, cache):
        fill(cache, 3)
        assert cache.as_dict()["held"] == 3

    def test_everything_put_can_be_got_before_eviction(self, cache):
        fill(cache, 8)
        assert all(cache.get(at) is not None for at in range(8))


class TestRecent:
    def test_the_oldest_untouched_is_evicted(self):
        cache = Recent(capacity=2)
        cache.put(1, b"a")
        cache.put(2, b"b")
        cache.put(3, b"c")
        assert cache.get(1) is None and cache.get(2) == b"b"

    def test_a_hit_rescues_a_block(self):
        cache = Recent(capacity=2)
        cache.put(1, b"a")
        cache.put(2, b"b")
        cache.get(1)
        cache.put(3, b"c")
        assert cache.get(1) == b"a" and cache.get(2) is None


class TestFrequent:
    def test_the_least_asked_for_is_evicted(self):
        cache = Frequent(capacity=2)
        cache.put(1, b"a")
        cache.put(2, b"b")
        cache.get(1)
        cache.get(1)
        cache.get(2)
        cache.put(3, b"c")
        assert cache.get(2) is None and cache.get(1) == b"a"

    def test_a_new_block_starts_with_one_count(self):
        cache = Frequent(capacity=2)
        cache.put(1, b"a")
        cache.get(1)
        cache.put(2, b"b")
        cache.put(3, b"c")
        assert cache.get(1) == b"a" and cache.get(2) is None


class TestClock:
    def test_an_untouched_block_is_evicted(self):
        cache = Clock(capacity=2)
        cache.put(1, b"a")
        cache.put(2, b"b")
        cache.get(2)
        cache.put(3, b"c")
        assert cache.get(1) is None and cache.get(2) == b"b"

    def test_the_bit_buys_one_pass_of_grace(self):
        cache = Clock(capacity=2)
        cache.put(1, b"a")
        cache.put(2, b"b")
        cache.get(1)
        cache.get(2)
        cache.put(3, b"c")
        assert len(cache) == 2

    def test_the_hand_wraps(self):
        cache = Clock(capacity=3)
        fill(cache, 9)
        assert len(cache) == 3


class TestChance:
    def test_the_same_seed_evicts_the_same_way(self):
        left, right = Chance(capacity=4, seed=9), Chance(capacity=4, seed=9)
        fill(left, 20)
        fill(right, 20)
        assert set(left.held) == set(right.held)

    def test_different_seeds_differ_eventually(self):
        left, right = Chance(capacity=4, seed=1), Chance(capacity=4, seed=2)
        fill(left, 200)
        fill(right, 200)
        assert set(left.held) != set(right.held)


class TestReference:
    def test_a_uniform_stream_has_the_right_length(self):
        assert len(Reference(blocks=10, length=100).stream()) == 100

    def test_a_uniform_stream_stays_in_range(self):
        assert all(0 <= one < 10 for one in Reference(blocks=10, length=100).stream())

    def test_a_scan_walks_in_order(self):
        assert Reference(blocks=100, length=5, shape="scan").stream() == [0, 1, 2, 3, 4]

    def test_a_scan_wraps(self):
        assert Reference(blocks=3, length=5, shape="scan").stream() == [0, 1, 2, 0, 1]

    def test_a_hot_stream_concentrates(self):
        made = Reference(blocks=1000, length=10000, shape="hot").stream()
        hot = sum(1 for one in made if one < 100)
        assert hot > 8500

    def test_a_hot_then_scan_stream_changes_character(self):
        made = Reference(blocks=1000, length=1000, shape="hot_then_scan").stream()
        assert made[500:] == list(range(500))

    def test_a_mixed_stream_interleaves(self):
        made = Reference(blocks=1000, length=100, shape="hot_with_scan").stream()
        assert made[0] == 0 and made[4] == 1 and made[8] == 2

    def test_an_unknown_shape_is_refused(self):
        with pytest.raises(ConfigError):
            Reference(blocks=10, length=10, shape="spiral").stream()

    def test_the_same_seed_gives_the_same_stream(self):
        assert (
            Reference(blocks=10, length=100).stream()
            == Reference(blocks=10, length=100).stream()
        )

    def test_as_dict_carries_the_shape(self):
        assert Reference(blocks=1, length=1, shape="scan").as_dict()["shape"] == "scan"


class TestRun:
    def test_a_run_counts_every_lookup(self):
        made = run(Recent(capacity=4), [1, 2, 3, 1, 2, 3])
        assert made.lookups == 6

    def test_a_run_fills_on_a_miss(self):
        cache = Recent(capacity=4)
        run(cache, [1, 2, 3])
        assert len(cache) == 3

    def test_a_repeated_block_hits(self):
        assert run(Recent(capacity=4), [1, 1, 1]).hits == 2

    def test_a_block_is_what_its_number_says(self):
        assert block(7) == (7).to_bytes(8, "little") * 8

    def test_two_blocks_differ(self):
        assert block(1) != block(2)


class TestMeasurements:
    def test_nothing_beats_random_without_locality(self):
        assert mod.no_policy_beats_random_on_a_workload_with_no_locality()

    def test_recency_needs_the_fit(self):
        assert mod.least_recently_used_beats_the_coin_only_while_the_hot_set_fits()

    def test_a_scan_zeroes_everyone(self):
        assert mod.a_scan_takes_every_policy_to_zero()

    def test_a_scan_hurts_recency_most(self):
        assert mod.a_scan_through_a_working_set_hurts_recency_most()

    def test_frequency_cannot_forget(self):
        assert mod.frequency_cannot_forget_and_pays_for_it_after_a_shift()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5

    def test_the_policy_table_covers_every_pair(self):
        assert len(mod.compare_the_policies(32, 300, 4000)) == 20

    def test_the_scan_rows_are_all_zero(self):
        rows = mod.compare_the_policies(32, 300, 4000)
        assert all(row["rate"] == 0.0 for row in rows if row["shape"] == "scan")

    def test_the_capacity_table_has_ten_rows(self):
        assert len(mod.compare_the_capacities(1000, 4000)) == 10

    def test_a_larger_cache_never_hits_less(self):
        rows = mod.compare_the_capacities(1000, 40000)
        rows = [row for row in rows if row["policy"] == "recent"]
        rates = [row["rate"] for row in rows]
        assert rates == sorted(rates)

    def test_measure_is_cached(self):
        first = measure("recent", 32, 100, 1000, "hot")
        assert first is measure("recent", 32, 100, 1000, "hot")

    def test_the_default_capacity_stands(self):
        assert CAPACITY == 256
