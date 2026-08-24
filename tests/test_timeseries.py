from __future__ import annotations

import pytest

from store import timeseries as mod
from store.errors import BadFormat, ConfigError
from store.timeseries import decode, encode, flat_bytes, unzigzag, zigzag


class TestZigzag:
    def test_zero_stays_zero(self):
        assert zigzag(0) == 0

    def test_minus_one_becomes_one(self):
        assert zigzag(-1) == 1

    def test_one_becomes_two(self):
        assert zigzag(1) == 2

    def test_the_fold_round_trips(self):
        for value in (-1000, -3, -1, 0, 1, 7, 12345):
            assert unzigzag(zigzag(value)) == value

    def test_small_magnitudes_stay_small(self):
        assert zigzag(-5) < 16 and zigzag(5) < 16


class TestEncode:
    def test_an_empty_series_is_refused(self):
        with pytest.raises(ConfigError):
            encode([])

    def test_a_backwards_series_is_refused(self):
        with pytest.raises(ConfigError):
            encode([10, 5])

    def test_a_singleton_round_trips(self):
        assert decode(encode([42])) == [42]

    def test_a_pair_round_trips(self):
        assert decode(encode([42, 52])) == [42, 52]

    def test_repeated_moments_round_trip(self):
        assert decode(encode([5, 5, 5])) == [5, 5, 5]

    def test_a_regular_series_round_trips(self):
        moments = [100 + at * 10 for at in range(500)]
        assert decode(encode(moments)) == moments

    def test_a_jittery_series_round_trips(self):
        moments = list(mod._scrape(500))
        assert decode(encode(moments)) == moments

    def test_an_irregular_series_round_trips(self):
        moments = list(mod._events(500))
        assert decode(encode(moments)) == moments

    def test_an_empty_buffer_is_refused(self):
        with pytest.raises(BadFormat):
            decode(b"")

    def test_a_regular_series_costs_about_a_byte_each(self):
        moments = [100 + at * 10 for at in range(1000)]
        assert len(encode(moments)) < 1100

    def test_flat_bytes_are_eight_each(self):
        assert flat_bytes([1, 2, 3]) == 24


class TestMeasurements:
    def test_regularity_needs_bit_packing(self):
        assert mod.the_metronome_buys_nothing_over_the_jittery_scrape_here()

    def test_jitter_costs_a_byte(self):
        assert mod.jitter_costs_one_byte_per_sample()

    def test_events_fall_back_to_deltas(self):
        assert mod.irregular_events_fall_back_to_delta_cost()

    def test_round_trips_hold(self):
        assert mod.the_round_trip_is_exact_on_every_shape()

    def test_backwards_time_is_refused(self):
        assert mod.backwards_time_is_refused()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5

    def test_the_shape_table_has_three_rows(self):
        rows = mod.compare_the_shapes()
        assert [row["shape"] for row in rows] == ["metronome", "scrape", "events"]

    def test_every_shape_beats_flat(self):
        rows = mod.compare_the_shapes()
        assert all(row["encoded_bytes"] < row["flat_bytes"] for row in rows)

    def test_the_streams_are_cached(self):
        assert mod._metronome(100) is mod._metronome(100)
