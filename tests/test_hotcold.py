from __future__ import annotations

import pytest

from store import hotcold as mod
from store.errors import ConfigError
from store.hotcold import FAST_COST, SLOW_COST, File, Placement, by_age, by_heat


def fleet(kind: str = "recency"):
    return mod._fleet(kind, 50, 3)


class TestPlacement:
    def test_an_over_budget_placement_is_refused(self):
        files = fleet()
        with pytest.raises(ConfigError):
            Placement(fast=set(files), files=files, budget=1)

    def test_an_empty_placement_is_legal(self):
        files = fleet()
        assert Placement(fast=set(), files=files, budget=0).daily_cost() > 0

    def test_everything_fast_costs_the_floor(self):
        files = fleet()
        total = sum(file.size for file in files.values())
        made = Placement(fast=set(files), files=files, budget=total)
        floor = sum(file.reads_per_day * FAST_COST for file in files.values())
        assert made.daily_cost() == floor

    def test_everything_slow_costs_the_ceiling(self):
        files = fleet()
        made = Placement(fast=set(), files=files, budget=0)
        ceiling = sum(file.reads_per_day * SLOW_COST for file in files.values())
        assert made.daily_cost() == ceiling

    def test_as_dict_counts_the_fast_bytes(self):
        files = fleet()
        chosen = next(iter(files))
        made = Placement(fast={chosen}, files=files, budget=10**6)
        assert made.as_dict()["fast_bytes"] == files[chosen].size


class TestPolicies:
    def test_by_age_prefers_the_newest(self):
        files = fleet()
        budget = files[0].size
        made = by_age(files, budget)
        assert 0 in made.fast

    def test_by_heat_prefers_the_hottest_per_byte(self):
        files = {
            1: File(number=1, age=1, size=100, reads_per_day=1),
            2: File(number=2, age=2, size=100, reads_per_day=1000),
        }
        made = by_heat(files, budget=100)
        assert made.fast == {2}

    def test_both_policies_respect_the_budget(self):
        files = fleet("scattered")
        budget = 500
        for policy in (by_age, by_heat):
            made = policy(files, budget)
            assert sum(files[number].size for number in made.fast) <= budget

    def test_heat_never_costs_more_than_age(self):
        for kind in ("recency", "scattered", "archival"):
            files = mod._fleet(kind, 100, 5)
            budget = mod._budget(files)
            assert by_heat(files, budget).daily_cost() <= by_age(files, budget).daily_cost()


class TestFleets:
    def test_an_unknown_shape_is_refused(self):
        with pytest.raises(ConfigError):
            mod._fleet("spiral")

    def test_the_recency_fleet_cools_with_age(self):
        files = mod._fleet("recency", 50, 3)
        assert files[0].reads_per_day > files[49].reads_per_day

    def test_the_archival_fleet_heats_the_tail(self):
        files = mod._fleet("archival", 50, 3)
        assert files[49].reads_per_day > files[0].reads_per_day

    def test_the_fleet_is_cached(self):
        assert mod._fleet("recency", 50, 3) is mod._fleet("recency", 50, 3)


class TestMeasurements:
    def test_age_wins_on_recency(self):
        assert mod.age_placement_wins_when_heat_follows_age()

    def test_age_loses_on_scatter(self):
        assert mod.age_placement_loses_badly_when_heat_scatters()

    def test_archival_inverts_age(self):
        assert mod.the_archival_shape_inverts_the_age_policy()

    def test_the_budget_binds(self):
        assert mod.the_budget_is_enforced()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4

    def test_the_shape_table_has_three_rows(self):
        rows = mod.compare_the_shapes()
        assert [row["workload"] for row in rows] == ["recency", "scattered", "archival"]

    def test_heat_wins_or_ties_every_row(self):
        assert all(row["by_heat"] <= row["by_age"] for row in mod.compare_the_shapes())
