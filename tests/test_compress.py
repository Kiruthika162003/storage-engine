from __future__ import annotations

import random

import pytest

from store import compress as mod
from store.compress import DEFLATED, RAW, measure, pack, unpack
from store.errors import BadFormat, ConfigError


class TestPack:
    def test_a_compressible_block_is_tagged_deflated(self):
        assert pack(bytes(4096))[0] == DEFLATED

    def test_an_incompressible_block_is_tagged_raw(self):
        assert pack(random.Random(1).randbytes(4096))[0] == RAW

    def test_an_empty_block_round_trips(self):
        assert unpack(pack(b"")) == b""

    def test_a_compressed_block_round_trips(self):
        block = b"abc" * 2000
        assert unpack(pack(block)) == block

    def test_a_raw_block_round_trips(self):
        block = random.Random(2).randbytes(2000)
        assert unpack(pack(block)) == block

    def test_the_raw_path_costs_one_byte(self):
        block = random.Random(3).randbytes(1000)
        assert len(pack(block)) == 1001

    def test_a_zero_threshold_compresses_marginal_wins(self):
        base = random.Random(4).randbytes(3000)
        block = base + bytes(200)
        eager = pack(block, threshold=0.0)
        cautious = pack(block, threshold=0.5)
        assert eager[0] == DEFLATED and cautious[0] == RAW


class TestUnpack:
    def test_an_empty_buffer_is_refused(self):
        with pytest.raises(BadFormat):
            unpack(b"")

    def test_an_unknown_tag_is_refused(self):
        with pytest.raises(BadFormat):
            unpack(bytes([7]) + b"payload")

    def test_a_damaged_deflate_stream_is_refused(self):
        packed = bytearray(pack(bytes(4096)))
        packed[5] ^= 0xFF
        with pytest.raises(BadFormat):
            unpack(bytes(packed))

    def test_a_raw_block_is_returned_verbatim(self):
        assert unpack(bytes([RAW]) + b"payload") == b"payload"


class TestMeasure:
    def test_an_empty_corpus_is_refused(self):
        with pytest.raises(ConfigError):
            measure("nothing", [])

    def test_the_outcome_counts_blocks(self):
        made = measure("x", [b"a" * 100, b"b" * 100])
        assert made.blocks == 2

    def test_the_ratio_divides_stored_by_raw(self):
        made = measure("x", [bytes(4096)])
        assert made.ratio < 0.1

    def test_kept_raw_plus_compressed_is_the_total(self):
        made = measure("x", list(mod._corpus("mixed", 20)))
        assert made.chose_raw + made.compressed_blocks == made.blocks

    def test_an_unknown_corpus_is_refused(self):
        with pytest.raises(ConfigError):
            mod._corpus("spiral")


class TestMeasurements:
    def test_text_yes_random_no(self):
        assert mod.text_compresses_and_random_does_not()

    def test_keys_beat_prose(self):
        assert mod.sorted_keys_compress_better_than_text()

    def test_the_threshold_holds_the_line(self):
        assert mod.the_threshold_keeps_marginal_wins_raw()

    def test_round_trips_hold(self):
        assert mod.every_block_round_trips_whichever_path_it_took()

    def test_damage_is_loud(self):
        assert mod.damage_to_a_compressed_block_is_loud()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5

    def test_the_corpus_table_has_five_rows(self):
        assert len(mod.compare_the_corpora()) == 5

    def test_zeros_compress_best(self):
        rows = {row["corpus"]: row["ratio"] for row in mod.compare_the_corpora()}
        assert rows["zeros"] == min(rows.values())

    def test_random_compresses_worst(self):
        rows = {row["corpus"]: row["ratio"] for row in mod.compare_the_corpora()}
        assert rows["random"] == max(rows.values())

    def test_the_corpora_are_cached(self):
        assert mod._corpus("text", 10) is mod._corpus("text", 10)
