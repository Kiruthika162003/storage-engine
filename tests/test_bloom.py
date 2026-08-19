from __future__ import annotations

import pytest

from store import bloom as filters
from store.bloom import (
    BITS_PER_KEY,
    HASHES,
    MAGIC,
    Filter,
    build,
    decode,
    expected_rate,
    measure_rate,
    optimal_hashes,
)
from store.errors import BadFormat, ConfigError


def test_there_are_no_false_negatives():
    assert filters.a_filter_never_says_no_to_a_key_it_holds()["there_are_no_false_negatives"]


def test_every_size_finds_its_keys():
    made = filters.a_filter_never_says_no_to_a_key_it_holds()
    assert all(made["found_at_every_size"].values())


def test_double_hashing_matches_the_formula():
    assert filters.double_hashing_matches_the_independence_formula()["they_agree_closely"]


def test_the_relative_error_is_small():
    made = filters.double_hashing_matches_the_independence_formula()
    assert made["worst_relative_error"] < 0.2


def test_one_digest_serves_every_hash():
    assert filters.double_hashing_matches_the_independence_formula()[
        "and_the_saving_is_the_hash_count"
    ]


def test_the_optimum_sets_half_the_bits():
    assert filters.the_optimum_sets_half_the_bits()["they_are_all_about_a_half"]


def test_the_fill_spread_is_small():
    assert filters.the_optimum_sets_half_the_bits()["and_it_is_small"]


def test_a_small_sample_resolves_to_one_hit():
    assert filters.a_small_probe_set_can_only_resolve_the_rate_to_one_hit()[
        "which_is_most_of_the_rate"
    ]


def test_the_three_readings_disagree():
    assert filters.a_small_probe_set_can_only_resolve_the_rate_to_one_hit()[
        "the_readings_disagree"
    ]


def test_the_largest_sample_agrees_with_the_theory():
    assert filters.a_small_probe_set_can_only_resolve_the_rate_to_one_hit()[
        "the_largest_sample_agrees"
    ]


def test_the_filter_did_not_change_between_readings():
    assert filters.a_small_probe_set_can_only_resolve_the_rate_to_one_hit()[
        "and_the_filter_did_not_change"
    ]


def test_both_ends_of_the_hash_count_are_worse():
    assert filters.the_wrong_hash_count_is_worse_in_both_directions()["both_ends_are_worse"]


def test_the_best_hash_count_is_the_formula():
    assert filters.the_wrong_hash_count_is_worse_in_both_directions()[
        "and_it_is_what_the_formula_says"
    ]


def test_one_hash_is_much_worse():
    made = filters.the_wrong_hash_count_is_worse_in_both_directions()
    assert made["one_hash_is_worse_by"] > 5


def test_the_middle_of_the_hash_range_is_shallow():
    assert filters.the_wrong_hash_count_is_worse_in_both_directions()["the_middle_is_shallow"]


def test_a_filter_round_trips():
    assert filters.a_filter_round_trips_through_its_encoding()["it_round_trips"]


def test_the_hash_count_survives_the_encoding():
    assert filters.a_filter_round_trips_through_its_encoding()["and_the_hash_count_did"]


def test_a_decoded_filter_still_finds_its_keys():
    assert filters.a_filter_round_trips_through_its_encoding()["and_it_still_finds_the_keys"]


def test_a_filter_over_no_keys_is_refused():
    assert filters.a_filter_over_no_keys_is_refused()


def test_a_zero_size_is_refused():
    assert filters.a_zero_size_is_refused()


def test_a_filter_with_no_hashes_is_refused():
    assert filters.a_filter_with_no_hashes_is_refused()


def test_something_that_is_not_a_filter_is_refused():
    assert filters.something_that_is_not_a_filter_is_refused()


def test_the_size_table_covers_seven():
    assert len(filters.compare_the_sizes()) == 7


def test_the_size_steps_are_alike():
    assert filters.each_extra_bit_per_key_costs_a_tenth_and_buys_a_third()[
        "and_the_steps_are_alike"
    ]


def test_there_is_no_knee_in_the_size_curve():
    assert filters.each_extra_bit_per_key_costs_a_tenth_and_buys_a_third()["there_is_no_knee"]


def test_doubling_the_space_cuts_the_rate_far_more():
    made = filters.each_extra_bit_per_key_costs_a_tenth_and_buys_a_third()
    assert made["and_the_rate_fell_by"] > made["space_from_eight_to_sixteen"]


def test_the_summary_says_there_are_no_false_negatives():
    assert filters.summarise()["no_false_negatives"]


def test_the_summary_reports_the_shipped_size():
    assert filters.summarise()["bits_per_key"] == BITS_PER_KEY


def test_the_optimal_hash_count_follows_the_formula():
    assert optimal_hashes(10) == 7


def test_a_tiny_filter_still_gets_one_hash():
    assert optimal_hashes(0.5) == 1


def test_a_zero_size_has_no_optimum():
    with pytest.raises(ConfigError):
        optimal_hashes(0)


def test_the_expected_rate_falls_with_the_size():
    assert expected_rate(16, 11) < expected_rate(8, 6)


def test_the_expected_rate_is_a_probability():
    assert 0 < expected_rate(10, 7) < 1


def test_an_expected_rate_needs_a_hash():
    with pytest.raises(ConfigError):
        expected_rate(10, 0)


def test_an_expected_rate_needs_a_size():
    with pytest.raises(ConfigError):
        expected_rate(0, 7)


def test_building_sizes_the_filter():
    made = build([b"a", b"b", b"c"], bits_per_key=8)
    assert made.size >= 24


def test_building_records_the_key_count():
    assert build([b"a", b"b"], bits_per_key=8).keys == 2


def test_building_uses_the_optimal_hash_count():
    assert build([b"a"], bits_per_key=10).hashes == optimal_hashes(10)


def test_building_accepts_an_explicit_hash_count():
    assert build([b"a"], bits_per_key=10, hashes=3).hashes == 3


def test_building_over_nothing_raises():
    with pytest.raises(ConfigError):
        build([])


def test_building_with_no_size_raises():
    with pytest.raises(ConfigError):
        build([b"a"], bits_per_key=0)


def test_a_filter_finds_what_it_holds():
    made = build([b"present"], bits_per_key=16)
    assert made.might_contain(b"present")


def test_a_filter_usually_misses_what_it_does_not():
    made = build([b"present"], bits_per_key=16)
    assert not made.might_contain(b"absent")


def test_a_filter_reports_its_fill():
    made = build([f"k{one}".encode() for one in range(100)], bits_per_key=10)
    assert 0 < made.fill < 1


def test_an_untouched_filter_has_no_bits_set():
    assert Filter(bits=bytearray(8), hashes=3).set_bits == 0


def test_a_filter_reports_its_bits_per_key():
    made = build([f"k{one}".encode() for one in range(1000)], bits_per_key=10)
    assert 9 < made.bits_per_key < 11


def test_a_filter_with_no_keys_has_no_bits_per_key():
    assert Filter(bits=bytearray(8), hashes=3).bits_per_key == 0.0


def test_a_filter_summarises():
    assert build([b"a"], bits_per_key=10).as_dict()["keys"] == 1


def test_a_filter_needs_bits():
    with pytest.raises(ConfigError):
        Filter(bits=bytearray(), hashes=3)


def test_a_filter_needs_a_hash():
    with pytest.raises(ConfigError):
        Filter(bits=bytearray(8), hashes=0)


def test_an_encoded_filter_starts_with_the_magic():
    raw = build([b"a"], bits_per_key=10).encode()
    assert int.from_bytes(raw[:4], "little") == MAGIC


def test_decoding_a_short_buffer_raises():
    with pytest.raises(BadFormat):
        decode(b"\x00" * 4)


def test_decoding_a_truncated_filter_raises():
    raw = build([b"a"], bits_per_key=10).encode()
    with pytest.raises(BadFormat):
        decode(raw[:20])


def test_decoding_a_foreign_buffer_raises():
    with pytest.raises(BadFormat):
        decode(b"\xff" * 64)


def test_measuring_a_rate_needs_probes():
    with pytest.raises(ConfigError):
        measure_rate(build([b"a"], bits_per_key=10), [])


def test_measuring_a_rate_returns_a_share():
    made = build([f"k{one}".encode() for one in range(100)], bits_per_key=10)
    rate = measure_rate(made, [f"z{one}".encode() for one in range(100)])
    assert 0.0 <= rate <= 1.0


def test_the_shipped_hash_count_matches_the_shipped_size():
    assert optimal_hashes(BITS_PER_KEY) == HASHES
