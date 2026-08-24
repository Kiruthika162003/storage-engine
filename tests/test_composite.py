from __future__ import annotations

import pytest

from store import composite as mod
from store.composite import FIELD_END, decode, encode, encode_field, naive
from store.errors import BadFormat, ConfigError


class TestEncode:
    def test_an_empty_tuple_is_refused(self):
        with pytest.raises(ConfigError):
            encode(())

    def test_one_field_gets_one_terminator(self):
        assert encode((b"a",)) == b"a" + FIELD_END

    def test_an_empty_field_is_just_the_terminator(self):
        assert encode((b"",)) == FIELD_END

    def test_zeros_are_escaped(self):
        assert encode((b"\x00",)) == b"\x00\xff" + FIELD_END

    def test_fields_concatenate_with_terminators(self):
        assert encode((b"a", b"b")) == b"a" + FIELD_END + b"b" + FIELD_END

    def test_the_terminator_cannot_appear_inside_a_field(self):
        made = encode_field(b"\x00\x01")
        assert FIELD_END not in made[:-2]


class TestDecode:
    def test_a_simple_pair_round_trips(self):
        assert decode(encode((b"a", b"bc"))) == (b"a", b"bc")

    def test_empties_round_trip(self):
        assert decode(encode((b"", b"", b""))) == (b"", b"", b"")

    def test_zeros_round_trip(self):
        assert decode(encode((b"\x00\x00", b"a\x00"))) == (b"\x00\x00", b"a\x00")

    def test_terminator_bytes_round_trip(self):
        assert decode(encode((b"\x00\x01", b"\x01"))) == (b"\x00\x01", b"\x01")

    def test_a_torn_escape_is_refused(self):
        with pytest.raises(BadFormat):
            decode(encode((b"a\x00b",))[:-1])

    def test_an_unknown_escape_is_refused(self):
        with pytest.raises(BadFormat):
            decode(b"a\x00\x7f" + FIELD_END)

    def test_a_trailing_fragment_is_refused(self):
        with pytest.raises(BadFormat):
            decode(encode((b"ab",)) + b"c")

    def test_an_empty_key_is_refused(self):
        with pytest.raises(ConfigError):
            decode(b"")


class TestOrdering:
    def test_prefix_pairs_sort_correctly(self):
        low, high = (b"a",), (b"a", b"")
        assert (low < high) == (encode(low) < encode(high))

    def test_short_against_long_fields_sort_correctly(self):
        low, high = (b"ab", b"z"), (b"abc", b"a")
        assert (low < high) == (encode(low) < encode(high))

    def test_zero_fields_sort_before_text(self):
        low, high = (b"\x00",), (b"a",)
        assert (low < high) == (encode(low) < encode(high))

    def test_the_corpus_sorts_identically(self):
        tuples = sorted(mod._tuples(1500))
        encoded = [encode(one) for one in tuples]
        assert encoded == sorted(encoded)

    def test_the_naive_encoding_collides(self):
        assert naive((b"ab", b"c")) == naive((b"a", b"bc"))

    def test_the_escaped_encoding_does_not(self):
        assert encode((b"ab", b"c")) != encode((b"a", b"bc"))


class TestSampler:
    def test_the_demand_over_the_space_is_refused(self):
        with pytest.raises(ConfigError):
            mod._tuples(2379)

    def test_the_corpus_is_distinct(self):
        made = mod._tuples(1000)
        assert len(set(made)) == 1000

    def test_the_corpus_is_cached(self):
        assert mod._tuples(500) is mod._tuples(500)


class TestMeasurements:
    def test_byte_order_is_tuple_order(self):
        assert mod.byte_order_agrees_with_tuple_order_everywhere()

    def test_concatenation_collides(self):
        assert mod.concatenation_confuses_field_boundaries()

    def test_round_trips_hold(self):
        assert mod.the_round_trip_is_exact_on_every_tuple()

    def test_prefix_scans_stop_at_the_field(self):
        assert mod.a_prefix_scan_matches_the_first_field_exactly()

    def test_torn_escapes_are_refused(self):
        assert mod.damage_inside_an_escape_is_refused()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5
