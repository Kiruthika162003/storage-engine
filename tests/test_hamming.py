from __future__ import annotations

import pytest

from store import hamming as mod
from store.errors import BadFormat, ConfigError
from store.hamming import WORD_BITS, decode, encode


class TestCodec:
    def test_a_wrong_size_payload_is_refused(self):
        with pytest.raises(ConfigError):
            encode(b"short")

    def test_a_wrong_size_word_is_refused(self):
        with pytest.raises(BadFormat):
            decode(b"12345678")

    def test_a_clean_word_decodes_clean(self):
        payload = b"ABCDEFGH"
        back, verdict = decode(encode(payload))
        assert back == payload and verdict == "clean"

    def test_all_zero_payload_round_trips(self):
        back, verdict = decode(encode(bytes(8)))
        assert back == bytes(8) and verdict == "clean"

    def test_all_ones_payload_round_trips(self):
        payload = bytes([255] * 8)
        back, verdict = decode(encode(payload))
        assert back == payload and verdict == "clean"

    def test_the_coded_word_is_nine_bytes(self):
        assert len(encode(b"12345678")) == 9

    def test_distinct_payloads_encode_distinctly(self):
        assert encode(b"AAAAAAAA") != encode(b"AAAAAAAB")


class TestCorrection:
    def test_a_data_bit_flip_corrects(self):
        coded = bytearray(encode(b"ABCDEFGH"))
        coded[5] ^= 0x10
        back, verdict = decode(bytes(coded))
        assert back == b"ABCDEFGH" and verdict == "corrected"

    def test_a_parity_bit_flip_corrects(self):
        coded = bytearray(encode(b"ABCDEFGH"))
        coded[0] ^= 0x02
        back, verdict = decode(bytes(coded))
        assert back == b"ABCDEFGH" and verdict == "corrected"

    def test_the_overall_parity_flip_corrects(self):
        coded = bytearray(encode(b"ABCDEFGH"))
        coded[0] ^= 0x01
        back, verdict = decode(bytes(coded))
        assert back == b"ABCDEFGH" and verdict == "corrected"

    def test_a_double_flip_is_refused(self):
        coded = bytearray(encode(b"ABCDEFGH"))
        coded[2] ^= 0x01
        coded[6] ^= 0x40
        with pytest.raises(BadFormat):
            decode(bytes(coded))


class TestMeasurements:
    def test_singles_correct(self):
        assert mod.every_single_bit_flip_corrects_in_place()

    def test_doubles_refuse(self):
        assert mod.every_double_flip_is_refused_never_miscorrected()

    def test_triples_can_lie(self):
        assert mod.triple_flips_can_lie_which_is_the_codes_edge()

    def test_correction_costs_triple(self):
        assert mod.the_redundancy_is_an_eighth_against_the_checksums_twenty_fifth()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4

    def test_the_word_is_seventy_two_bits(self):
        assert WORD_BITS == 72
