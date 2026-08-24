from __future__ import annotations

import pytest

from store import parallelscan as mod
from store.errors import ConfigError
from store.parallelscan import (
    cut_by_keyspace,
    cut_by_quantiles,
    makespan_ratio,
)


class TestKeyspaceCut:
    def test_zero_workers_is_refused(self):
        with pytest.raises(ConfigError):
            cut_by_keyspace([1], 0, 100)

    def test_every_row_is_assigned(self):
        splits = cut_by_keyspace([1, 50, 99], 2, 100)
        assert sum(split.rows for split in splits) == 3

    def test_rows_land_in_their_span(self):
        splits = cut_by_keyspace([10, 60], 2, 100)
        assert splits[0].rows == 1 and splits[1].rows == 1

    def test_a_key_at_the_top_lands_in_the_last(self):
        splits = cut_by_keyspace([99], 2, 100)
        assert splits[1].rows == 1

    def test_one_worker_takes_everything(self):
        splits = cut_by_keyspace([1, 2, 3], 1, 100)
        assert splits[0].rows == 3


class TestQuantileCut:
    def test_zero_workers_is_refused(self):
        with pytest.raises(ConfigError):
            cut_by_quantiles([1], 0)

    def test_every_row_is_assigned(self):
        splits = cut_by_quantiles(list(range(100)), 4)
        assert sum(split.rows for split in splits) == 100

    def test_uniform_rows_split_evenly(self):
        splits = cut_by_quantiles(list(range(100)), 4)
        assert all(split.rows == 25 for split in splits)

    def test_clustered_rows_still_split_evenly(self):
        keys = [5] * 0 + list(range(50)) + [1000 + at for at in range(50)]
        splits = cut_by_quantiles(keys, 4)
        assert all(split.rows == 25 for split in splits)


class TestMakespan:
    def test_a_perfect_split_scores_one(self):
        splits = cut_by_quantiles(list(range(100)), 4)
        assert makespan_ratio(splits) == 1.0

    def test_a_lopsided_split_scores_high(self):
        splits = cut_by_keyspace([1] * 90 + [99] * 10, 2, 100)
        assert makespan_ratio(splits) == 1.8

    def test_an_empty_workload_scores_one(self):
        assert makespan_ratio(cut_by_keyspace([], 4, 100)) == 1.0


class TestMeasurements:
    def test_keyspace_cuts_need_uniformity(self):
        assert mod.keyspace_cuts_balance_only_uniform_data()

    def test_quantile_cuts_balance_anything(self):
        assert mod.quantile_cuts_balance_anything()

    def test_rows_are_conserved(self):
        assert mod.both_cuts_assign_every_row_exactly_once()

    def test_workers_multiply_the_cut_rule(self):
        assert mod.eight_times_the_workers_buys_three_under_the_bad_cut()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4

    def test_the_key_sets_are_cached(self):
        assert mod._clustered(100) is mod._clustered(100)
