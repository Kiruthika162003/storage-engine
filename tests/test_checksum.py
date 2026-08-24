from __future__ import annotations

import zlib

from store import checksum as mod
from store.checksum import CHECKSUMS, crc32, fletcher, plain_sum


class TestBasics:
    def test_an_empty_payload_has_a_tag(self):
        for check in CHECKSUMS.values():
            assert isinstance(check(b""), int)

    def test_the_tags_are_stable(self):
        raw = b"the same bytes"
        for check in CHECKSUMS.values():
            assert check(raw) == check(raw)

    def test_different_payloads_usually_differ(self):
        for check in CHECKSUMS.values():
            assert check(b"one payload") != check(b"another payload")

    def test_the_sum_is_the_sum(self):
        assert plain_sum(bytes([1, 2, 3])) == 6

    def test_the_crc_matches_zlib(self):
        raw = b"reference"
        assert crc32(raw) == zlib.crc32(raw) & 0xFFFFFFFF

    def test_fletcher_weights_position(self):
        assert fletcher(b"ab") != fletcher(b"ba")

    def test_the_sum_does_not_weight_position(self):
        assert plain_sum(b"ab") == plain_sum(b"ba")

    def test_three_candidates_are_registered(self):
        assert set(CHECKSUMS) == {"sum", "fletcher", "crc32"}


class TestBlindness:
    def test_a_swap_blinds_the_sum(self):
        assert mod.a_swap_is_invisible_to_the_sum_and_visible_to_the_others()

    def test_balance_blinds_the_sum(self):
        assert mod.a_balanced_pair_of_flips_is_invisible_to_the_sum()

    def test_single_bits_catch_everywhere(self):
        assert mod.every_single_bit_flip_is_caught_by_all_three()

    def test_the_sum_wastes_its_width(self):
        assert mod.the_sum_does_not_even_use_its_32_bits()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4


class TestBlindnessTable:
    def test_the_table_has_a_row_per_checksum(self):
        assert len(mod.blindness_table(500)) == 3

    def test_the_sum_misses_swaps(self):
        rows = {row["checksum"]: row for row in mod.blindness_table(2000)}
        assert rows["sum"]["swap"] > 0

    def test_the_sum_misses_balanced_damage(self):
        rows = {row["checksum"]: row for row in mod.blindness_table(2000)}
        assert rows["sum"]["balanced"] > 0

    def test_the_crc_misses_nothing_structured(self):
        rows = {row["checksum"]: row for row in mod.blindness_table(2000)}
        assert all(rows["crc32"][kind] == 0 for kind in ("swap", "balanced", "zero_run"))

    def test_fletcher_misses_nothing_structured(self):
        rows = {row["checksum"]: row for row in mod.blindness_table(2000)}
        assert all(rows["fletcher"][kind] == 0 for kind in ("swap", "balanced", "zero_run"))
