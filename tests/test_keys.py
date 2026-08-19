from __future__ import annotations

import struct

import pytest

from store import keys as encoding
from store.errors import ConfigError, TooLarge
from store.keys import (
    BLOB,
    FLOATS,
    INT,
    INTS,
    NULL,
    TAGS,
    TEXT,
    TEXTS,
    Ordering,
    check_order,
    decode_float,
    decode_int,
    encode_float,
    encode_int,
    encode_text,
    key_of,
    naive_float,
    naive_int,
    value_of,
)


def test_the_integer_encoding_is_order_preserving():
    assert encoding.a_sortable_integer_encoding_is_right_about_every_pair()[
        "it_is_order_preserving"
    ]


def test_the_integer_encoding_matches_a_sort():
    assert encoding.a_sortable_integer_encoding_is_right_about_every_pair()[
        "and_it_matches_sorted"
    ]


def test_the_integer_encoding_round_trips():
    assert encoding.a_sortable_integer_encoding_is_right_about_every_pair()["it_round_trips"]


def test_decimal_text_is_not_order_preserving():
    assert encoding.an_integer_written_as_text_puts_ten_before_nine()[
        "it_is_not_order_preserving"
    ]


def test_decimal_text_does_not_sort():
    assert encoding.an_integer_written_as_text_puts_ten_before_nine()["and_it_is_not_sorted"]


def test_decimal_text_has_backwards_pairs():
    made = encoding.an_integer_written_as_text_puts_ten_before_nine()
    assert made["backwards_pairs"] > 0


def test_a_raw_float_is_worse_than_a_coin():
    made = encoding.a_raw_float_is_wrong_more_often_than_it_is_right()
    assert made["it_is_worse_than_a_coin"]


def test_raw_float_negatives_sort_above():
    made = encoding.a_raw_float_is_wrong_more_often_than_it_is_right()
    assert made["the_negatives_sort_above"]


def test_raw_float_negatives_sort_backwards():
    assert encoding.a_raw_float_is_wrong_more_often_than_it_is_right()[
        "and_backwards_among_themselves"
    ]


def test_the_fixed_float_encoding_is_exact():
    assert encoding.a_raw_float_is_wrong_more_often_than_it_is_right()[
        "and_it_is_right_about_every_pair"
    ]


def test_utf_8_is_order_preserving():
    assert encoding.utf_8_needs_no_help_and_utf_16_would()["utf_8_is_order_preserving"]


def test_utf_8_handles_an_astral_character():
    assert encoding.utf_8_needs_no_help_and_utf_16_would()["with_an_astral_character"]


def test_utf_16_is_not():
    assert encoding.utf_8_needs_no_help_and_utf_16_would()["and_utf_16_is_not"]


def test_utf_16_gets_exactly_two_pairs_wrong():
    assert encoding.utf_8_needs_no_help_and_utf_16_would()["which_is_two_pairs_of_ninety"]


def test_the_tags_group_the_types():
    assert encoding.a_type_tag_keeps_the_types_apart()["they_are_grouped"]


def test_the_tag_decides_the_order():
    assert encoding.a_type_tag_keeps_the_types_apart()["and_the_tag_puts_it_below"]


def test_without_a_tag_nobody_chose_the_order():
    assert encoding.a_type_tag_keeps_the_types_apart()["which_nobody_chose"]


def test_an_oversized_integer_is_refused():
    assert encoding.a_key_too_large_to_frame_is_refused()


def test_an_unencodable_value_is_refused():
    assert encoding.a_value_with_no_encoding_is_refused()


def test_an_empty_key_is_refused():
    assert encoding.an_empty_key_has_no_value()


def test_the_encoding_table_covers_five():
    assert len(encoding.compare_the_encodings()) == 5


def test_the_decimal_encoding_is_mostly_right():
    assert encoding.the_wrong_encodings_are_wrong_in_different_ways()[
        "the_decimal_one_is_mostly_right"
    ]


def test_the_float_encoding_is_mostly_wrong():
    assert encoding.the_wrong_encodings_are_wrong_in_different_ways()[
        "and_the_float_one_is_mostly_wrong"
    ]


def test_the_mostly_right_one_is_the_dangerous_one():
    assert encoding.the_wrong_encodings_are_wrong_in_different_ways()[
        "so_the_first_is_the_dangerous_one"
    ]


def test_the_summary_says_the_integer_encoding_is_exact():
    assert encoding.summarise()["integer_encoding_is_exact"]


def test_the_summary_counts_the_types():
    assert encoding.summarise()["types"] == len(TAGS)


def test_an_integer_encodes_to_eight_bytes():
    assert len(encode_int(0)) == 8


def test_zero_encodes_to_the_midpoint():
    assert encode_int(0) == b"\x80" + b"\x00" * 7


def test_a_negative_sorts_below_zero():
    assert encode_int(-1) < encode_int(0)


def test_a_positive_sorts_above_zero():
    assert encode_int(1) > encode_int(0)


def test_the_smallest_integer_encodes_to_zeroes():
    assert encode_int(-(2**63)) == b"\x00" * 8


def test_the_largest_integer_encodes_to_ones():
    assert encode_int(2**63 - 1) == b"\xff" * 8


def test_an_integer_round_trips():
    assert decode_int(encode_int(-12345)) == -12345


def test_an_oversized_integer_raises():
    with pytest.raises(TooLarge):
        encode_int(2**63)


def test_a_short_integer_decode_raises():
    with pytest.raises(ConfigError):
        decode_int(b"\x00")


def test_a_float_encodes_to_eight_bytes():
    assert len(encode_float(1.5)) == 8


def test_a_negative_float_sorts_below_zero():
    assert encode_float(-1.0) < encode_float(0.0)


def test_a_larger_negative_sorts_lower():
    assert encode_float(-10.0) < encode_float(-1.0)


def test_a_float_round_trips():
    assert decode_float(encode_float(-2.75)) == -2.75


def test_a_float_infinity_round_trips():
    assert decode_float(encode_float(float("inf"))) == float("inf")


def test_infinity_sorts_above_everything():
    assert encode_float(float("inf")) > encode_float(1e308)


def test_a_short_float_decode_raises():
    with pytest.raises(ConfigError):
        decode_float(b"\x00" * 4)


def test_text_encodes_as_utf_8():
    assert encode_text("abc") == b"abc"


def test_text_sorts_by_code_point():
    assert encode_text("a") < encode_text("b")


def test_a_prefix_sorts_below_its_extension():
    assert encode_text("ab") < encode_text("abc")


def test_a_key_carries_its_tag():
    assert key_of(1)[0] == INT


def test_a_text_key_carries_the_text_tag():
    assert key_of("a")[0] == TEXT


def test_a_blob_key_carries_the_blob_tag():
    assert key_of(b"x")[0] == BLOB


def test_a_null_key_is_one_byte():
    assert key_of(None) == bytes([NULL])


def test_a_boolean_is_not_an_integer_key():
    assert key_of(True) != key_of(1)


def test_a_key_round_trips():
    for one in (None, True, False, -3, 0.5, "text", b"bytes"):
        assert value_of(key_of(one)) == one


def test_an_unknown_type_raises():
    with pytest.raises(ConfigError):
        key_of({1: 2})


def test_an_unknown_tag_raises():
    with pytest.raises(ConfigError):
        value_of(bytes([0xFE]) + b"junk")


def test_the_naive_integer_is_decimal():
    assert naive_int(-12) == b"-12"


def test_the_naive_float_is_raw():
    assert naive_float(1.5) == struct.pack(">d", 1.5)


def test_an_ordering_reports_its_share():
    made = Ordering(name="x", values=(1, 2), correct=3, total=4)
    assert made.share == 0.75


def test_a_perfect_ordering_is_truthy():
    assert Ordering(name="x", values=(1,), correct=4, total=4)


def test_an_imperfect_ordering_is_falsy():
    assert not Ordering(name="x", values=(1,), correct=3, total=4)


def test_an_ordering_summarises():
    made = Ordering(name="named", values=(1,), correct=1, total=1)
    assert made.as_dict()["encoding"] == "named"


def test_an_ordering_of_no_pairs_raises():
    with pytest.raises(ConfigError):
        Ordering(name="x", values=(), correct=0, total=0)


def test_checking_an_order_counts_every_pair():
    assert check_order("x", (1, 2, 3), encode_int).total == 6


def test_checking_a_correct_encoding_finds_no_fault():
    assert check_order("x", INTS, encode_int)


def test_checking_a_broken_encoding_finds_one():
    assert not check_order("x", INTS, naive_int)


def test_the_float_sample_spans_zero():
    assert min(FLOATS) < 0 < max(FLOATS)


def test_the_text_sample_holds_an_empty_string():
    assert "" in TEXTS
