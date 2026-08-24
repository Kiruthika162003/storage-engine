from __future__ import annotations

import pytest

from store import bitpack as mod
from store.bitpack import ZERO_RUN, pack, pack_block, unpack, unpack_block
from store.errors import BadFormat, ConfigError


class TestPackBlock:
    def test_an_empty_block_is_refused(self):
        with pytest.raises(ConfigError):
            pack_block([])

    def test_a_negative_value_is_refused(self):
        with pytest.raises(ConfigError):
            pack_block([-1])

    def test_all_zeros_collapse_to_three_bytes(self):
        assert len(pack_block([0] * 5000)) == 3

    def test_the_zero_run_marker_is_written(self):
        assert pack_block([0, 0])[0] == ZERO_RUN

    def test_small_values_pack_tightly(self):
        raw = pack_block([1, 0, 1, 1, 0, 1, 0, 1])
        assert len(raw) == 4

    def test_the_width_follows_the_widest(self):
        assert pack_block([7])[0] == 3
        assert pack_block([8])[0] == 4

    def test_an_oversized_count_is_refused(self):
        with pytest.raises(ConfigError):
            pack_block([1] * 70000)


class TestUnpackBlock:
    def test_a_short_header_is_refused(self):
        with pytest.raises(BadFormat):
            unpack_block(b"\x01")

    def test_a_short_body_is_refused(self):
        raw = pack_block([255] * 10)
        with pytest.raises(BadFormat):
            unpack_block(raw[:-2])

    def test_a_zero_run_unpacks(self):
        values, used = unpack_block(pack_block([0] * 7))
        assert values == [0] * 7 and used == 3

    def test_a_packed_block_unpacks(self):
        values, _ = unpack_block(pack_block([5, 3, 1]))
        assert values == [5, 3, 1]


class TestStream:
    def test_a_stream_round_trips(self):
        values = [0, 1, 2, 0, 0, 9, 0]
        assert unpack(pack(values)) == values

    def test_blocks_split_at_the_size(self):
        values = list(range(300))
        assert unpack(pack(values, block=100)) == values

    def test_an_empty_stream_is_empty(self):
        assert unpack(pack([])) == []

    def test_zero_heavy_streams_shrink(self):
        values = [0] * 900 + [1] * 10
        assert len(pack(values, block=256)) < 40

    def test_wide_values_round_trip(self):
        values = [2**60, 0, 2**60 - 1]
        assert unpack(pack(values)) == values


class TestMeasurements:
    def test_the_metronome_pays_off(self):
        assert mod.the_metronome_finally_costs_fractions_of_a_byte()

    def test_jitter_packs_to_half(self):
        assert mod.the_jittery_scrape_packs_to_half_a_byte()

    def test_round_trips_hold(self):
        assert mod.round_trips_hold_across_widths_and_runs()

    def test_outliers_tax_their_block(self):
        assert mod.the_width_is_per_block_and_an_outlier_taxes_only_its_block()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4

    def test_the_second_differences_are_zig_zagged(self):
        wobbles = mod._second_differences([100, 110, 120, 131])
        assert all(value >= 0 for value in wobbles)
