from __future__ import annotations

import pytest

from store import planner as mod
from store.errors import ConfigError
from store.planner import RANDOM_COST, SEQUENTIAL_COST, Planner, Stats, true_cost


def fitted(values):
    stats = Stats(low=min(values), high=max(values) + 1)
    for value in values:
        stats.note(value)
    return Planner(stats=stats)


class TestStats:
    def test_an_inverted_range_is_refused(self):
        with pytest.raises(ConfigError):
            Stats(low=10, high=5)

    def test_an_empty_stats_estimates_zero(self):
        assert Stats(low=0, high=100).selectivity(0, 50) == 0.0

    def test_a_full_range_estimates_one(self):
        stats = Stats(low=0, high=100)
        for value in range(0, 100, 5):
            stats.note(value)
        assert stats.selectivity(0, 100) > 0.95

    def test_a_backwards_range_estimates_zero(self):
        stats = Stats(low=0, high=100)
        stats.note(50)
        assert stats.selectivity(60, 40) == 0.0

    def test_a_half_range_estimates_near_half_on_uniform_data(self):
        stats = Stats(low=0, high=1000)
        for value in range(0, 1000, 2):
            stats.note(value)
        assert 0.4 < stats.selectivity(0, 499) < 0.6

    def test_values_at_the_edges_are_bucketed(self):
        stats = Stats(low=0, high=100)
        stats.note(0)
        stats.note(99)
        assert stats.rows == 2

    def test_rows_count_every_note(self):
        stats = Stats(low=0, high=10)
        for _ in range(7):
            stats.note(5)
        assert stats.rows == 7


class TestPlanner:
    def test_a_needle_chooses_the_index(self):
        planner = fitted(list(range(10000)))
        assert planner.choose(5, 10) == "index"

    def test_a_sweep_chooses_the_scan(self):
        planner = fitted(list(range(10000)))
        assert planner.choose(0, 9999) == "scan"

    def test_the_choices_are_counted(self):
        planner = fitted(list(range(10000)))
        planner.choose(5, 10)
        planner.choose(0, 9999)
        assert planner.chose_index == 1 and planner.chose_scan == 1

    def test_the_scan_cost_is_the_rows(self):
        planner = fitted(list(range(1000)))
        assert planner.scan_cost() == 1000 * SEQUENTIAL_COST

    def test_the_index_cost_scales_with_selectivity(self):
        planner = fitted(list(range(1000)))
        narrow = planner.index_cost(0, 9)
        wide = planner.index_cost(0, 499)
        assert wide > narrow * 10


class TestTrueCost:
    def test_a_scan_costs_the_rows(self):
        assert true_cost([1, 2, 3], 0, 10, "scan") == 3 * SEQUENTIAL_COST

    def test_an_index_costs_the_matches(self):
        assert true_cost([1, 2, 3, 50], 0, 10, "index") == 3 * RANDOM_COST

    def test_an_index_miss_costs_nothing(self):
        assert true_cost([50, 60], 0, 10, "index") == 0


class TestMeasurements:
    def test_the_planner_beats_both_fixed_policies(self):
        assert mod.the_planner_beats_always_scan_and_always_index()

    def test_the_crossover_is_displaced_by_error(self):
        assert mod.the_crossover_is_arithmetic_in_the_estimate_and_displaced_in_truth()

    def test_errors_live_in_wide_buckets(self):
        assert mod.estimation_errors_live_where_the_buckets_are_wide()

    def test_empty_ranges_are_free(self):
        assert mod.an_empty_range_is_estimated_free()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4

    def test_the_column_is_cached(self):
        assert mod._column() is mod._column()

    def test_the_column_is_skewed(self):
        values = sorted(mod._column())
        median = values[len(values) // 2]
        assert values[-1] > median * 10
