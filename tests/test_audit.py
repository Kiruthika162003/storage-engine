from __future__ import annotations

import pytest

from store import audit as mod
from store.audit import GENESIS, Chain
from store.errors import BadFormat, ConfigError


class TestChain:
    def test_an_empty_entry_is_refused(self):
        with pytest.raises(ConfigError):
            Chain().append(b"")

    def test_an_empty_chain_has_the_genesis_head(self):
        assert Chain().head == GENESIS

    def test_an_append_moves_the_head(self):
        chain = Chain()
        before = chain.head
        chain.append(b"first")
        assert chain.head != before

    def test_the_same_entries_give_the_same_head(self):
        left, right = Chain(), Chain()
        for content in (b"a", b"b", b"c"):
            left.append(content)
            right.append(content)
        assert left.head == right.head

    def test_order_changes_the_head(self):
        left, right = Chain(), Chain()
        left.append(b"a")
        left.append(b"b")
        right.append(b"b")
        right.append(b"a")
        assert left.head != right.head

    def test_an_empty_chain_verifies(self):
        assert Chain().verify() == 0

    def test_a_clean_chain_verifies(self):
        chain = mod._grown(50)
        assert chain.verify() == 50

    def test_the_saved_head_is_checked(self):
        chain = mod._grown(50)
        with pytest.raises(BadFormat):
            chain.verify(bytes(32))


class TestTampering:
    def test_an_edit_is_caught_at_its_line(self):
        chain = mod._grown(50)
        _, stored = chain.entries[10]
        chain.entries[10] = (b"forged", stored)
        with pytest.raises(BadFormat) as caught:
            chain.verify()
        assert "entry 10" in str(caught.value)

    def test_a_deletion_is_caught(self):
        chain = mod._grown(50)
        del chain.entries[10]
        with pytest.raises(BadFormat):
            chain.verify()

    def test_an_insertion_is_caught(self):
        chain = mod._grown(50)
        chain.entries.insert(10, (b"forged", bytes(32)))
        with pytest.raises(BadFormat):
            chain.verify()

    def test_a_swap_is_caught(self):
        chain = mod._grown(50)
        chain.entries[10], chain.entries[11] = chain.entries[11], chain.entries[10]
        with pytest.raises(BadFormat):
            chain.verify()

    def test_a_rebuilt_suffix_passes_the_walk(self):
        chain = mod._grown(50)
        contents = [content for content, _ in chain.entries]
        contents[10] = b"forged"
        rebuilt = Chain()
        for content in contents:
            rebuilt.append(content)
        assert rebuilt.verify() == 50

    def test_the_rebuilt_suffix_fails_the_saved_head(self):
        chain = mod._grown(50)
        saved = chain.head
        contents = [content for content, _ in chain.entries]
        contents[10] = b"forged"
        rebuilt = Chain()
        for content in contents:
            rebuilt.append(content)
        with pytest.raises(BadFormat):
            rebuilt.verify(saved)


class TestMeasurements:
    def test_clean_chains_verify(self):
        assert mod.a_clean_chain_verifies_end_to_end()

    def test_edits_break_at_the_line(self):
        assert mod.editing_any_line_breaks_the_chain_there()

    def test_all_tamper_styles_break(self):
        assert mod.deletion_insertion_and_reorder_all_break()

    def test_suffix_rewrites_need_the_head(self):
        assert mod.a_full_suffix_rewrite_defeats_the_walk_and_meets_the_saved_head()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4
