from __future__ import annotations

import pytest

from store import timekey as mod
from store.errors import ConfigError
from store.timekey import (
    CEILING,
    ascending,
    descending,
    latest,
    read_ascending,
    read_descending,
)


class TestEncodings:
    def test_ascending_sorts_oldest_first(self):
        assert ascending(1) < ascending(2)

    def test_descending_sorts_newest_first(self):
        assert descending(2) < descending(1)

    def test_zero_round_trips_both_ways(self):
        assert read_ascending(ascending(0)) == 0
        assert read_descending(descending(0)) == 0

    def test_the_ceiling_round_trips_both_ways(self):
        assert read_ascending(ascending(CEILING)) == CEILING
        assert read_descending(descending(CEILING)) == CEILING

    def test_a_negative_moment_is_refused(self):
        with pytest.raises(ConfigError):
            ascending(-1)

    def test_an_oversized_moment_is_refused(self):
        with pytest.raises(ConfigError):
            descending(CEILING + 1)

    def test_a_short_key_is_refused(self):
        with pytest.raises(ConfigError):
            read_ascending(b"\x00\x01")

    def test_the_widths_are_fixed(self):
        assert len(ascending(5)) == len(descending(5)) == 8


class TestQueries:
    def test_latest_takes_the_front(self):
        keys = [descending(moment) for moment in (10, 30, 20)]
        found = [read_descending(key) for key in latest(keys, 2)]
        assert found == [30, 20]

    def test_latest_of_more_than_held_returns_everything(self):
        keys = [descending(moment) for moment in (10, 30)]
        assert len(latest(keys, 10)) == 2

    def test_latest_of_zero_returns_nothing(self):
        assert latest([descending(1)], 0) == []

    def test_ties_are_stable(self):
        keys = [descending(5), descending(5)]
        assert latest(keys, 2) == keys


class TestMeasurements:
    def test_the_complement_mirrors(self):
        assert mod.the_complement_reverses_the_sort_exactly()

    def test_latest_n_is_a_prefix(self):
        assert mod.the_latest_n_is_a_prefix_read()

    def test_edges_round_trip(self):
        assert mod.the_round_trip_is_exact_at_the_edges()

    def test_the_donation_is_final(self):
        assert mod.the_donated_direction_cannot_be_taken_back()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4

    def test_the_moments_are_cached(self):
        assert mod._moments(100) is mod._moments(100)

    def test_the_moments_increase(self):
        moments = mod._moments(500)
        assert list(moments) == sorted(moments)
