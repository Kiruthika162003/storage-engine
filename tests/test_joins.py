from __future__ import annotations

import pytest

from store import joins as mod
from store.errors import ConfigError
from store.joins import Meter, hash_join, merge_join


class TestHashJoin:
    def test_a_simple_match_joins(self):
        made = hash_join([(1, b"a")], [(1, b"x")], Meter())
        assert made == [(1, b"a", b"x")]

    def test_no_match_no_rows(self):
        assert hash_join([(1, b"a")], [(2, b"x")], Meter()) == []

    def test_many_to_one_multiplies(self):
        made = hash_join([(1, b"a")], [(1, b"x"), (1, b"y")], Meter())
        assert len(made) == 2

    def test_many_to_many_products(self):
        made = hash_join([(1, b"a"), (1, b"b")], [(1, b"x"), (1, b"y")], Meter())
        assert len(made) == 4

    def test_the_left_column_stays_left_when_flipped(self):
        made = hash_join([(1, b"a"), (2, b"b"), (3, b"c")], [(1, b"x")], Meter())
        assert made == [(1, b"a", b"x")]

    def test_the_build_side_is_the_smaller(self):
        meter = Meter()
        hash_join([(at, b"l") for at in range(100)], [(1, b"r")], meter)
        assert meter.held_rows == 1

    def test_empty_sides_join_to_nothing(self):
        assert hash_join([], [(1, b"x")], Meter()) == []


class TestMergeJoin:
    def test_a_simple_match_joins(self):
        made = merge_join([(1, b"a")], [(1, b"x")], Meter())
        assert made == [(1, b"a", b"x")]

    def test_unsorted_left_is_refused(self):
        with pytest.raises(ConfigError):
            merge_join([(2, b"a"), (1, b"b")], [(1, b"x")], Meter())

    def test_unsorted_right_is_refused(self):
        with pytest.raises(ConfigError):
            merge_join([(1, b"a")], [(2, b"x"), (1, b"y")], Meter())

    def test_many_to_many_products(self):
        made = merge_join([(1, b"a"), (1, b"b")], [(1, b"x"), (1, b"y")], Meter())
        assert len(made) == 4

    def test_gaps_on_either_side_are_skipped(self):
        made = merge_join([(1, b"a"), (5, b"e")], [(3, b"c"), (5, b"x")], Meter())
        assert made == [(5, b"e", b"x")]

    def test_the_window_is_the_group(self):
        meter = Meter()
        merge_join([(1, b"a")] * 3, [(1, b"x")] * 4, meter)
        assert meter.held_rows == 7

    def test_empty_sides_join_to_nothing(self):
        assert merge_join([], [(1, b"x")], Meter()) == []


class TestAgreement:
    def test_the_generated_sides_agree(self):
        left, right = mod._sides(3000, 300)
        assert hash_join(list(left), list(right), Meter()) == merge_join(
            list(left), list(right), Meter()
        )


class TestMeasurements:
    def test_the_joins_agree(self):
        assert mod.both_joins_produce_identical_output()

    def test_memory_is_the_separation(self):
        assert mod.the_hash_join_holds_the_build_side_and_the_merge_holds_a_group()

    def test_touches_are_a_tie(self):
        assert mod.both_joins_touch_each_row_about_once()

    def test_unsorted_is_refused(self):
        assert mod.the_merge_join_refuses_unsorted_input()

    def test_skew_bloats_the_window(self):
        assert mod.skew_bloats_the_merge_joins_window()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5

    def test_the_sides_are_cached(self):
        assert mod._sides(100, 10) is mod._sides(100, 10)
