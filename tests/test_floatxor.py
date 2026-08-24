from __future__ import annotations

import pytest

from store import floatxor as mod
from store.errors import BadFormat, ConfigError
from store.floatxor import decode, encode, flat_bytes


class TestCodec:
    def test_an_empty_series_is_refused(self):
        with pytest.raises(ConfigError):
            encode([])

    def test_a_singleton_round_trips(self):
        assert decode(encode([3.25])) == [3.25]

    def test_a_short_buffer_is_refused(self):
        with pytest.raises(BadFormat):
            decode(b"\x00\x01")

    def test_a_torn_window_is_refused(self):
        raw = encode([1.0, 2.0])
        with pytest.raises(BadFormat):
            decode(raw[:-1])

    def test_repeats_cost_one_byte(self):
        raw = encode([7.5, 7.5, 7.5])
        assert len(raw) == 10

    def test_a_changing_pair_round_trips(self):
        assert decode(encode([1.5, 2.5])) == [1.5, 2.5]

    def test_negative_values_round_trip(self):
        values = [-1.5, -2.25, -1.5]
        assert decode(encode(values)) == values

    def test_negative_zero_survives_bitwise(self):
        back = decode(encode([0.0, -0.0]))
        assert mod._bits(back[1]) == mod._bits(-0.0)

    def test_infinities_round_trip(self):
        values = [float("inf"), float("-inf"), 0.0]
        assert decode(encode(values)) == values

    def test_the_gauge_round_trips(self):
        values = list(mod._gauge(2000))
        assert decode(encode(values)) == values

    def test_the_noise_round_trips(self):
        values = list(mod._noise(2000))
        assert decode(encode(values)) == values

    def test_flat_bytes_are_eight_each(self):
        assert flat_bytes([1.0, 2.0]) == 16


class TestMeasurements:
    def test_constants_cost_a_byte(self):
        assert mod.a_flat_gauge_costs_one_byte_per_repeat()

    def test_the_gauge_disappoints(self):
        assert mod.the_rounded_gauge_disappoints_and_the_reason_is_decimal()

    def test_noise_costs_more_than_flat(self):
        assert mod.uncorrelated_doubles_cost_more_than_flat()

    def test_round_trips_are_bit_exact(self):
        assert mod.round_trips_are_bit_exact()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4

    def test_the_shape_table_has_three_rows(self):
        rows = mod.compare_the_shapes()
        assert [row["shape"] for row in rows] == ["constant", "gauge", "noise"]

    def test_the_constant_row_is_cheapest(self):
        rows = mod.compare_the_shapes()
        costs = [row["bytes_per_sample"] for row in rows]
        assert costs == sorted(costs)

    def test_the_streams_are_cached(self):
        assert mod._gauge(100) is mod._gauge(100)
