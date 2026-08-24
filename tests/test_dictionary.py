from __future__ import annotations

import pytest

from store import dictionary as mod
from store.dictionary import Encoded, plain_bytes
from store.errors import NotFound


class TestEncoded:
    def test_an_appended_value_reads_back(self):
        made = Encoded()
        made.append(b"red")
        assert made.read(0) == b"red"

    def test_repeats_share_a_code(self):
        made = Encoded()
        made.append(b"red")
        made.append(b"red")
        assert made.column == [0, 0] and len(made.values) == 1

    def test_distinct_values_get_distinct_codes(self):
        made = Encoded()
        made.append(b"red")
        made.append(b"blue")
        assert made.column == [0, 1]

    def test_a_read_past_the_end_raises(self):
        with pytest.raises(NotFound):
            Encoded().read(0)

    def test_order_is_preserved(self):
        made = Encoded()
        for value in (b"b", b"a", b"b", b"c"):
            made.append(value)
        assert [made.read(at) for at in range(4)] == [b"b", b"a", b"b", b"c"]


class TestCosts:
    def test_small_dictionaries_use_one_byte_codes(self):
        made = Encoded()
        for at in range(100):
            made.append(f"v{at % 10}".encode())
        assert made.code_bytes == 100

    def test_large_dictionaries_use_two_byte_codes(self):
        made = Encoded()
        for at in range(300):
            made.append(f"v{at:04d}".encode())
        assert made.code_bytes == 600

    def test_the_dictionary_holds_each_value_once(self):
        made = Encoded()
        for _ in range(50):
            made.append(b"abcdef")
        assert made.dictionary_bytes == 6

    def test_plain_bytes_sum_the_occurrences(self):
        assert plain_bytes([b"ab", b"cde"]) == 5


class TestScans:
    def test_a_match_is_found_everywhere_it_occurs(self):
        made = Encoded()
        for value in (b"a", b"b", b"a", b"c", b"a"):
            made.append(value)
        assert made.scan_equal(b"a") == [0, 2, 4]

    def test_an_absent_value_matches_nowhere(self):
        made = Encoded()
        made.append(b"a")
        assert made.scan_equal(b"zzz") == []

    def test_the_scan_agrees_with_a_plain_filter(self):
        column = list(mod._cities(3000))
        made = Encoded()
        for value in column:
            made.append(value)
        wanted = column[42]
        truth = [at for at, value in enumerate(column) if value == wanted]
        assert made.scan_equal(wanted) == truth


class TestMeasurements:
    def test_low_cardinality_compresses(self):
        assert mod.low_cardinality_compresses_by_the_width_over_the_code()

    def test_high_cardinality_overheads(self):
        assert mod.high_cardinality_makes_the_dictionary_pure_overhead()

    def test_reads_round_trip(self):
        assert mod.every_read_round_trips()

    def test_scans_compare_codes(self):
        assert mod.equality_scans_compare_codes_not_values()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4

    def test_the_columns_are_cached(self):
        assert mod._cities(100) is mod._cities(100)
