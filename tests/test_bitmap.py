from __future__ import annotations

import random

from store import bitmap as mod
from store.bitmap import Bitmap, posting_bytes, posting_intersect


def of(rows) -> Bitmap:
    made = Bitmap()
    for at in rows:
        made.set_row(at)
    return made


class TestBitmap:
    def test_a_set_row_is_a_position(self):
        assert of([5]).positions() == [5]

    def test_positions_come_back_sorted(self):
        assert of([70, 3, 64]).positions() == [3, 64, 70]

    def test_word_boundaries_are_exact(self):
        assert of([63, 64]).positions() == [63, 64]

    def test_an_empty_bitmap_has_no_positions(self):
        assert Bitmap().positions() == []

    def test_double_setting_is_idempotent(self):
        made = of([5, 5])
        assert made.positions() == [5]

    def test_the_bytes_grow_by_words(self):
        assert of([0]).nbytes == 8 and of([64]).nbytes == 16

    def test_intersect_keeps_the_common(self):
        assert of([1, 2, 3]).intersect(of([2, 3, 4])).positions() == [2, 3]

    def test_intersect_of_disjoint_is_empty(self):
        assert of([1]).intersect(of([100])).positions() == []

    def test_intersect_of_different_lengths_works(self):
        assert of([1, 200]).intersect(of([1])).positions() == [1]

    def test_union_keeps_everything(self):
        assert of([1, 3]).union(of([2, 200])).positions() == [1, 2, 3, 200]

    def test_union_of_different_lengths_keeps_the_tail(self):
        assert of([1]).union(of([300])).positions() == [1, 300]


class TestPostings:
    def test_the_merge_intersects(self):
        assert posting_intersect([1, 2, 3], [2, 3, 4]) == [2, 3]

    def test_disjoint_lists_intersect_to_nothing(self):
        assert posting_intersect([1], [2]) == []

    def test_an_empty_list_intersects_to_nothing(self):
        assert posting_intersect([], [1, 2]) == []

    def test_posting_bytes_are_four_each(self):
        assert posting_bytes([1, 2, 3]) == 12


class TestAgreement:
    def test_random_sets_agree(self):
        source = random.Random(3)
        left = sorted(source.sample(range(2000), 300))
        right = sorted(source.sample(range(2000), 300))
        wanted = posting_intersect(left, right)
        assert of(left).intersect(of(right)).positions() == wanted


class TestMeasurements:
    def test_the_structures_agree(self):
        assert mod.bitmaps_and_postings_answer_identically()

    def test_dense_values_suit_bits(self):
        assert mod.common_values_cost_less_as_bits()

    def test_sparse_values_suit_postings(self):
        assert mod.rare_values_waste_bits_by_the_thousand()

    def test_the_and_is_word_shaped(self):
        assert mod.the_intersection_touches_words_not_rows()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4

    def test_the_table_is_cached(self):
        assert mod._table(100) is mod._table(100)
