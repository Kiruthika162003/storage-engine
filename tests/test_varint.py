from __future__ import annotations

import itertools

import pytest

from store import varint as mod
from store.errors import BadFormat, ConfigError
from store.varint import (
    decode,
    decode_all,
    decode_deltas,
    encode,
    encode_all,
    encode_deltas,
    flat_bytes,
)


class TestEncode:
    def test_zero_is_one_byte(self):
        assert encode(0) == b"\x00"

    def test_the_one_byte_ceiling(self):
        assert len(encode(127)) == 1

    def test_the_two_byte_floor(self):
        assert len(encode(128)) == 2

    def test_the_two_byte_ceiling(self):
        assert len(encode(16383)) == 2

    def test_the_64_bit_worst_case(self):
        assert len(encode(2**64 - 1)) == 10

    def test_a_negative_is_refused(self):
        with pytest.raises(ConfigError):
            encode(-1)

    def test_every_byte_but_the_last_continues(self):
        raw = encode(2**30)
        assert all(byte & 0x80 for byte in raw[:-1]) and not raw[-1] & 0x80


class TestDecode:
    def test_zero_round_trips(self):
        assert decode(encode(0)) == (0, 1)

    def test_boundaries_round_trip(self):
        for value in (1, 127, 128, 16383, 16384, 2**32, 2**63, 2**64 - 1):
            back, _ = decode(encode(value))
            assert back == value

    def test_decode_reports_where_it_ended(self):
        raw = encode(300) + encode(5)
        value, at = decode(raw)
        assert value == 300 and decode(raw, at) == (5, len(raw))

    def test_an_empty_buffer_is_refused(self):
        with pytest.raises(BadFormat):
            decode(b"")

    def test_a_truncated_varint_is_refused(self):
        with pytest.raises(BadFormat):
            decode(encode(2**40)[:-1])

    def test_an_overlong_varint_is_refused(self):
        with pytest.raises(BadFormat):
            decode(bytes([0x80] * 11))

    def test_a_run_round_trips(self):
        values = [0, 1, 127, 128, 5000, 2**40]
        assert decode_all(encode_all(values)) == values

    def test_an_empty_run_round_trips(self):
        assert decode_all(encode_all([])) == []


class TestDeltas:
    def test_a_sorted_run_round_trips(self):
        values = [3, 7, 7, 100, 4096]
        assert decode_deltas(encode_deltas(values)) == values

    def test_a_dense_run_costs_one_byte_each(self):
        values = list(range(1, 1001))
        assert len(encode_deltas(values)) == 1000

    def test_an_unsorted_run_is_refused(self):
        with pytest.raises(ConfigError):
            encode_deltas([5, 3])

    def test_equal_values_are_allowed(self):
        assert decode_deltas(encode_deltas([4, 4, 4])) == [4, 4, 4]

    def test_deltas_beat_plain_varints_on_dense_runs(self):
        values = [10**6 + at for at in range(500)]
        assert len(encode_deltas(values)) < len(encode_all(values))

    def test_the_flat_reference_is_eight_bytes_each(self):
        assert flat_bytes([1, 2, 3]) == 24


class TestMeasurements:
    def test_widths_break_at_powers_of_128(self):
        assert mod.small_integers_cost_one_byte_in_nine_less_at_the_top()

    def test_deltas_hit_the_one_byte_floor(self):
        assert mod.sequence_numbers_shrink_eightfold_as_deltas()

    def test_the_round_trip_is_exact(self):
        assert mod.the_round_trip_is_exact_over_the_whole_range()

    def test_truncation_is_refused(self):
        assert mod.a_truncated_varint_is_refused_not_misread()

    def test_overlong_is_refused(self):
        assert mod.an_overlong_varint_is_refused()

    def test_unsorted_is_refused(self):
        assert mod.an_unsorted_run_is_refused_by_the_delta_encoder()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_six_claims(self):
        assert len(mod.summarise()) == 6

    def test_the_encoding_table_has_three_rows(self):
        assert len(mod.compare_the_encodings(1000)) == 3

    def test_each_encoding_beats_the_one_before(self):
        rows = mod.compare_the_encodings(1000)
        sizes = [row["bytes"] for row in rows]
        assert sizes == sorted(sizes, reverse=True)

    def test_the_sequences_are_cached(self):
        assert mod._sequences(100) is mod._sequences(100)

    def test_the_sequences_are_increasing(self):
        values = mod._sequences(1000)
        assert all(a < b for a, b in itertools.pairwise(values))
