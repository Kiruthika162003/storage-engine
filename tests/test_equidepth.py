from __future__ import annotations

import pytest

from store import equidepth as mod
from store.equidepth import DepthStats
from store.errors import ConfigError
from store.planner import _column as planner_column


class TestFit:
    def test_no_rows_is_refused(self):
        with pytest.raises(ConfigError):
            DepthStats.fit([])

    def test_one_bucket_is_refused(self):
        with pytest.raises(ConfigError):
            DepthStats.fit([1, 2], buckets=1)

    def test_the_bounds_are_quantiles(self):
        made = DepthStats.fit(list(range(100)), buckets=4)
        assert made.bounds == [25, 50, 75]

    def test_the_per_bucket_count_is_even(self):
        made = DepthStats.fit(list(range(100)), buckets=4)
        assert made.per_bucket == 25

    def test_skewed_data_gets_skewed_bounds(self):
        values = [1] * 90 + list(range(1000, 1010))
        made = DepthStats.fit(values, buckets=4)
        assert made.bounds[0] == 1 and made.bounds[-1] <= 1010


class TestSelectivity:
    def test_a_backwards_range_is_zero(self):
        made = DepthStats.fit(list(range(100)))
        assert made.selectivity(50, 10) == 0.0

    def test_the_full_range_is_near_one(self):
        made = DepthStats.fit(list(range(1000)))
        assert made.selectivity(0, 999) > 0.9

    def test_a_half_range_is_near_half(self):
        made = DepthStats.fit(list(range(1000)))
        assert 0.4 < made.selectivity(0, 499) < 0.6

    def test_a_miss_below_is_near_zero(self):
        made = DepthStats.fit(list(range(1000, 2000)))
        assert made.selectivity(0, 10) < 0.05

    def test_estimates_never_exceed_one(self):
        made = DepthStats.fit(list(range(100)))
        assert made.selectivity(-10**9, 10**9) <= 1.0


class TestMeasurements:
    def test_the_error_halves(self):
        assert mod.equidepth_halves_the_error_on_the_convicting_queries()

    def test_the_crossover_comes_home(self):
        assert mod.the_displaced_crossover_comes_home()

    def test_the_easy_cases_still_work(self):
        assert mod.both_histograms_agree_on_the_easy_cases()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_three_claims(self):
        assert len(mod.summarise()) == 3

    def test_the_column_matches_the_planners(self):
        assert mod._column() == planner_column()
