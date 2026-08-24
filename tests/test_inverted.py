from __future__ import annotations

import pytest

from store import inverted as mod
from store.errors import ConfigError
from store.inverted import Index, grep_and, grep_phrase, tokenize


def small() -> Index:
    index = Index()
    index.add(1, b"the cat sat on the mat")
    index.add(2, b"the dog sat")
    index.add(3, b"a cat and a dog")
    return index


class TestTokenize:
    def test_words_split_on_whitespace(self):
        assert tokenize(b"a b  c") == [b"a", b"b", b"c"]

    def test_case_folds(self):
        assert tokenize(b"The CAT") == [b"the", b"cat"]

    def test_empty_text_has_no_tokens(self):
        assert tokenize(b"") == []


class TestIndex:
    def test_a_duplicate_document_is_refused(self):
        index = small()
        with pytest.raises(ConfigError):
            index.add(1, b"again")

    def test_docs_with_finds_the_holders(self):
        assert small().docs_with(b"cat") == [1, 3]

    def test_docs_with_is_case_blind(self):
        assert small().docs_with(b"CAT") == [1, 3]

    def test_an_unknown_token_holds_nothing(self):
        assert small().docs_with(b"bird") == []


class TestAnd:
    def test_a_single_token_and(self):
        assert small().search_and([b"sat"]) == [1, 2]

    def test_a_two_token_and(self):
        assert small().search_and([b"cat", b"dog"]) == [3]

    def test_an_empty_query_finds_nothing(self):
        assert small().search_and([]) == []

    def test_a_hopeless_and_short_circuits(self):
        assert small().search_and([b"bird", b"cat"]) == []

    def test_the_and_agrees_with_grep(self):
        index = small()
        assert index.search_and([b"the", b"sat"]) == grep_and(index.documents, [b"the", b"sat"])


class TestPhrase:
    def test_an_adjacent_phrase_is_found(self):
        assert small().search_phrase(b"cat sat") == [1]

    def test_a_reversed_phrase_is_not(self):
        assert small().search_phrase(b"sat cat") == []

    def test_a_gap_defeats_the_phrase(self):
        assert small().search_phrase(b"the sat") == []

    def test_the_phrase_agrees_with_grep(self):
        index = small()
        assert index.search_phrase(b"the mat") == grep_phrase(index.documents, b"the mat")

    def test_a_three_word_phrase_works(self):
        assert small().search_phrase(b"sat on the") == [1]


class TestMeasurements:
    def test_the_index_agrees_with_grep(self):
        assert mod.the_index_agrees_with_grep_on_ands_and_phrases()

    def test_the_rarest_term_drives(self):
        assert mod.intersection_work_tracks_the_rarest_term()

    def test_phrases_pay_for_positions(self):
        assert mod.phrases_pay_positions_on_top_of_the_and()

    def test_duplicates_are_refused(self):
        assert mod.duplicate_documents_are_refused()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4

    def test_the_corpus_is_cached(self):
        assert mod._corpus(100) is mod._corpus(100)
