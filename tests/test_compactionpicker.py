from __future__ import annotations

import pytest

from store import compactionpicker as mod
from store.compactionpicker import (
    Candidate,
    efficiency,
    pick_by_overlap,
    pick_largest,
    pick_oldest,
)
from store.errors import ConfigError


def one(number=0, age=1, size=100, dead=10, estimate=None) -> Candidate:
    return Candidate(
        number=number,
        age=age,
        size=size,
        dead_bytes=dead,
        overlap_estimate=dead if estimate is None else estimate,
    )


class TestPickers:
    def test_an_empty_fleet_is_refused(self):
        for picker in (pick_oldest, pick_largest, pick_by_overlap):
            with pytest.raises(ConfigError):
                picker([])

    def test_oldest_picks_by_age(self):
        fleet = [one(number=1, age=5), one(number=2, age=50)]
        assert pick_oldest(fleet).number == 2

    def test_largest_picks_by_size(self):
        fleet = [one(number=1, size=100), one(number=2, size=900)]
        assert pick_largest(fleet).number == 2

    def test_overlap_picks_by_garbage_density(self):
        fleet = [
            one(number=1, size=1000, dead=50),
            one(number=2, size=200, dead=150),
        ]
        assert pick_by_overlap(fleet).number == 2

    def test_overlap_uses_the_estimate_not_the_truth(self):
        fleet = [
            one(number=1, size=100, dead=90, estimate=1),
            one(number=2, size=100, dead=10, estimate=99),
        ]
        assert pick_by_overlap(fleet).number == 2


class TestEfficiency:
    def test_efficiency_divides_size_by_dead(self):
        assert efficiency(one(size=100, dead=25)) == 4.0

    def test_a_clean_file_is_infinite_churn(self):
        assert efficiency(one(dead=0)) == float("inf")


class TestFleet:
    def test_the_fleet_is_cached(self):
        assert mod._fleet(30) is mod._fleet(30)

    def test_garbage_is_uncorrelated_with_age(self):
        fleet = mod._fleet()
        oldest = pick_oldest(list(fleet))
        assert oldest.dead_bytes < oldest.size * 0.2


class TestMeasurements:
    def test_proxies_pay(self):
        assert mod.the_proxies_pick_clean_files_and_pay_for_it()

    def test_overlap_finds_the_garbage(self):
        assert mod.the_overlap_picker_finds_the_dense_garbage()

    def test_the_gap_compounds(self):
        assert mod.a_sequence_of_picks_compounds_the_gap()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_three_claims(self):
        assert len(mod.summarise()) == 3
