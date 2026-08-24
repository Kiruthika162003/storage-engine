from __future__ import annotations

import pytest

from store.ngram import (
    Trigrams,
    an_absent_trigram_ends_the_query_early,
    brute,
    candidates_are_not_answers,
    grams,
    summarise,
    the_index_agrees_with_grep_on_every_probe,
    the_postings_are_smaller_than_the_text,
)


class TestGrams:
    def test_a_word_cuts_into_overlapping_trigrams(self):
        assert grams(b"abcd") == {b"abc", b"bcd"}

    def test_a_short_text_is_its_own_gram(self):
        assert grams(b"ab") == {b"ab"}

    def test_repeats_collapse(self):
        assert grams(b"aaaa") == {b"aaa"}


class TestTrigrams:
    def test_add_returns_ascending_numbers(self):
        index = Trigrams()
        assert index.add(b"one") == 0
        assert index.add(b"two") == 1

    def test_a_stored_string_is_found_whole(self):
        index = Trigrams()
        index.add(b"compaction")
        got, _ = index.search(b"compaction")
        assert got == [0]

    def test_an_inner_substring_is_found(self):
        index = Trigrams()
        index.add(b"compaction")
        got, _ = index.search(b"pact")
        assert got == [0]

    def test_a_false_candidate_is_rejected(self):
        index = Trigrams()
        index.add(b"abc xyz")
        index.add(b"abcxyz")
        got, checked = index.search(b"bcxy")
        assert got == [1]
        assert checked >= 1

    def test_an_unknown_needle_finds_nothing(self):
        index = Trigrams()
        index.add(b"compaction")
        got, checked = index.search(b"zzz")
        assert got == [] and checked == 0

    def test_memory_counts_posting_entries(self):
        index = Trigrams()
        index.add(b"abcd")
        assert index.memory() == 2

    def test_brute_is_the_reference(self):
        documents = [b"aa", b"ab", b"ba"]
        assert brute(documents, b"a") == [0, 1, 2]
        assert brute(documents, b"ab") == [1]


class TestClaims:
    @pytest.mark.parametrize(
        "claim",
        [
            the_index_agrees_with_grep_on_every_probe,
            candidates_are_not_answers,
            an_absent_trigram_ends_the_query_early,
            the_postings_are_smaller_than_the_text,
        ],
    )
    def test_claim_holds(self, claim):
        assert claim() is True

    def test_summary_is_all_true(self):
        told = summarise()
        assert all(value for name, value in told.items() if name != "module")
