from __future__ import annotations

import pytest

from store import metrics as mod
from store.errors import ConfigError
from store.metrics import GROWTH, Exact, Histogram, linear_histogram


def loaded(values) -> tuple[Exact, Histogram]:
    exact, histogram = Exact(), Histogram()
    for value in values:
        exact.add(value)
        histogram.add(value)
    return exact, histogram


class TestExact:
    def test_a_sample_is_kept(self):
        made = Exact()
        made.add(1.5)
        assert made.count == 1

    def test_the_median_of_a_run_is_the_middle(self):
        made = Exact(samples=[1.0, 2.0, 3.0])
        assert made.percentile(50) == 2.0

    def test_the_zeroth_percentile_is_the_minimum(self):
        made = Exact(samples=[3.0, 1.0, 2.0])
        assert made.percentile(0) == 1.0

    def test_the_hundredth_percentile_is_the_maximum(self):
        made = Exact(samples=[3.0, 1.0, 2.0])
        assert made.percentile(100) == 3.0

    def test_an_empty_exact_refuses_to_rank(self):
        with pytest.raises(ConfigError):
            Exact().percentile(50)

    def test_a_rank_out_of_range_is_refused(self):
        with pytest.raises(ConfigError):
            Exact(samples=[1.0]).percentile(150)

    def test_the_cost_is_eight_bytes_a_sample(self):
        made = Exact(samples=[1.0] * 10)
        assert made.nbytes == 80

    def test_as_dict_carries_both_percentiles(self):
        made = Exact(samples=[1.0, 2.0])
        assert {"p50", "p99"} <= set(made.as_dict())


class TestHistogram:
    def test_a_growth_at_or_below_one_is_refused(self):
        with pytest.raises(ConfigError):
            Histogram(growth=1.0)

    def test_a_non_positive_floor_is_refused(self):
        with pytest.raises(ConfigError):
            Histogram(lowest=0.0)

    def test_a_negative_sample_is_refused(self):
        with pytest.raises(ConfigError):
            Histogram().add(-1.0)

    def test_a_sample_is_counted(self):
        made = Histogram()
        made.add(0.001)
        assert made.count == 1

    def test_an_empty_histogram_refuses_to_rank(self):
        with pytest.raises(ConfigError):
            Histogram().percentile(50)

    def test_a_rank_out_of_range_is_refused(self):
        made = Histogram()
        made.add(1.0)
        with pytest.raises(ConfigError):
            made.percentile(-1)

    def test_the_percentile_is_within_one_bucket(self):
        exact, histogram = loaded(mod._lognormal(5000))
        true = exact.percentile(99)
        approx = histogram.percentile(99)
        assert true / GROWTH <= approx <= true * GROWTH

    def test_the_median_is_within_one_bucket(self):
        exact, histogram = loaded(mod._lognormal(5000))
        true = exact.percentile(50)
        approx = histogram.percentile(50)
        assert true / GROWTH <= approx <= true * GROWTH

    def test_the_extremes_are_tracked(self):
        _, histogram = loaded([0.001, 0.5])
        assert histogram.low == 0.001 and histogram.high == 0.5

    def test_a_tiny_sample_lands_in_bucket_zero(self):
        made = Histogram(lowest=1.0)
        made.add(0.5)
        assert made.counts[0] == 1

    def test_the_memory_is_bounded_by_the_range(self):
        _, histogram = loaded(mod._lognormal(20000))
        assert histogram.nbytes < 2000

    def test_as_dict_reports_none_when_empty(self):
        assert Histogram().as_dict()["p50"] is None


class TestMerge:
    def test_merged_counts_add(self):
        left, right = Histogram(), Histogram()
        left.add(0.001)
        right.add(0.002)
        assert left.merge(right).count == 2

    def test_merge_matches_a_single_build(self):
        samples = list(mod._lognormal(4000))
        whole = Histogram()
        left, right = Histogram(), Histogram()
        for at, value in enumerate(samples):
            whole.add(value)
            (left if at % 2 else right).add(value)
        assert left.merge(right).counts == whole.counts

    def test_mismatched_buckets_refuse_to_merge(self):
        with pytest.raises(ConfigError):
            Histogram(growth=1.1).merge(Histogram(growth=1.5))

    def test_merge_keeps_the_extremes(self):
        left, right = Histogram(), Histogram()
        left.add(0.001)
        right.add(0.9)
        merged = left.merge(right)
        assert merged.low == 0.001 and merged.high == 0.9

    def test_merge_leaves_the_inputs_alone(self):
        left, right = Histogram(), Histogram()
        left.add(0.001)
        right.add(0.002)
        left.merge(right)
        assert left.count == 1 and right.count == 1


class TestLinear:
    def test_linear_buckets_count_everything(self):
        made = linear_histogram(0.1, [0.05, 0.15, 0.25])
        assert sum(made) == 3

    def test_linear_buckets_place_by_width(self):
        made = linear_histogram(0.1, [0.05, 0.15])
        assert made[0] == 1 and made[1] == 1


class TestMeasurements:
    def test_percentiles_match_to_growth(self):
        assert mod.the_histogram_matches_the_exact_percentiles_to_its_growth_factor()

    def test_memory_is_a_thousandth(self):
        assert mod.the_histogram_costs_a_thousandth_of_the_samples()

    def test_linear_buckets_blur_the_lower_half(self):
        assert mod.linear_buckets_put_the_whole_lower_half_in_bucket_zero()

    def test_merge_is_exact(self):
        assert mod.merged_histograms_agree_with_one_built_whole()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4

    def test_the_cost_table_has_two_rows(self):
        assert len(mod.compare_the_costs(5000)) == 2

    def test_the_histogram_row_is_smaller(self):
        rows = mod.compare_the_costs(5000)
        assert rows[1]["bytes"] < rows[0]["bytes"]

    def test_the_samples_are_cached(self):
        assert mod._lognormal(100) is mod._lognormal(100)
