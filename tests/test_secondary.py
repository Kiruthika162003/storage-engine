from __future__ import annotations

import pytest

from store import secondary as mod
from store.errors import ConfigError
from store.secondary import Indexed, Lazy, scan_find


class TestIndexed:
    def test_a_put_is_findable(self):
        made = Indexed()
        made.put(b"k", b"red")
        assert made.find(b"red") == {b"k"}

    def test_an_empty_key_is_refused(self):
        with pytest.raises(ConfigError):
            Indexed().put(b"", b"v")

    def test_an_update_leaves_the_old_value(self):
        made = Indexed()
        made.put(b"k", b"red")
        made.put(b"k", b"blue")
        assert made.find(b"red") == set() and made.find(b"blue") == {b"k"}

    def test_a_delete_leaves_the_index(self):
        made = Indexed()
        made.put(b"k", b"red")
        made.delete(b"k")
        assert made.find(b"red") == set()

    def test_two_keys_share_a_value(self):
        made = Indexed()
        made.put(b"a", b"red")
        made.put(b"b", b"red")
        assert made.find(b"red") == {b"a", b"b"}

    def test_a_rewrite_of_the_same_value_is_free(self):
        made = Indexed()
        made.put(b"k", b"red")
        before = made.index_writes
        made.put(b"k", b"red")
        assert made.index_writes == before

    def test_an_update_costs_two_index_writes(self):
        made = Indexed()
        made.put(b"k", b"red")
        before = made.index_writes
        made.put(b"k", b"blue")
        assert made.index_writes == before + 2

    def test_an_unknown_value_finds_nothing(self):
        assert Indexed().find(b"nope") == set()


class TestLazy:
    def test_a_put_is_findable(self):
        made = Lazy()
        made.put(b"k", b"red")
        assert made.find(b"red") == {b"k"}

    def test_an_update_does_not_return_the_old_value(self):
        made = Lazy()
        made.put(b"k", b"red")
        made.put(b"k", b"blue")
        assert made.find(b"red") == set()

    def test_the_old_entry_lingers_underneath(self):
        made = Lazy()
        made.put(b"k", b"red")
        made.put(b"k", b"blue")
        assert made.stale_entries() == 1

    def test_a_delete_does_not_return_the_key(self):
        made = Lazy()
        made.put(b"k", b"red")
        made.delete(b"k")
        assert made.find(b"red") == set()

    def test_queries_pay_checks(self):
        made = Lazy()
        made.put(b"k", b"red")
        made.find(b"red")
        assert made.checks == 1

    def test_a_scrub_removes_the_corpses(self):
        made = Lazy()
        made.put(b"k", b"red")
        made.put(b"k", b"blue")
        assert made.scrub() == 1 and made.stale_entries() == 0

    def test_a_scrub_keeps_the_living(self):
        made = Lazy()
        made.put(b"k", b"red")
        made.scrub()
        assert made.find(b"red") == {b"k"}

    def test_a_scrubbed_empty_value_disappears(self):
        made = Lazy()
        made.put(b"k", b"red")
        made.put(b"k", b"blue")
        made.scrub()
        assert b"red" not in made.index


class TestAgainstTheScan:
    def test_indexed_agrees_after_churn(self):
        made = mod._driven("indexed", 2000, 300, 3)
        for at in range(50):
            value = f"v{at:03d}".encode()
            assert made.find(value) == scan_find(made.primary, value)

    def test_lazy_agrees_after_churn(self):
        made = mod._driven("lazy", 2000, 300, 4)
        for at in range(50):
            value = f"v{at:03d}".encode()
            assert made.find(value) == scan_find(made.primary, value)

    def test_the_scan_finds_what_was_put(self):
        assert scan_find({b"a": b"x", b"b": b"y"}, b"x") == {b"a"}


class TestMeasurements:
    def test_both_agree_with_the_scan(self):
        assert mod.both_disciplines_agree_with_the_scan()

    def test_synchronous_doubles_writes(self):
        assert mod.the_synchronous_index_doubles_the_write_cost()

    def test_lazy_pays_per_query(self):
        assert mod.the_lazy_index_writes_less_and_pays_per_query()

    def test_corpses_accumulate(self):
        assert mod.stale_entries_accumulate_and_a_scrub_removes_them()

    def test_the_classic_bug_is_wrong_answers(self):
        assert mod.forgetting_the_leave_half_is_the_classic_bug()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5

    def test_the_discipline_table_has_two_rows(self):
        assert len(mod.compare_the_disciplines(2000)) == 2

    def test_lazy_writes_less(self):
        rows = {row["discipline"]: row for row in mod.compare_the_disciplines(2000)}
        assert rows["lazy"]["index_writes"] < rows["indexed"]["index_writes"]

    def test_only_lazy_checks(self):
        rows = {row["discipline"]: row for row in mod.compare_the_disciplines(2000)}
        assert rows["indexed"]["checks"] == 0 < rows["lazy"]["checks"]

    def test_the_driven_stores_are_cached(self):
        assert mod._driven("lazy", 100, 50, 1) is mod._driven("lazy", 100, 50, 1)
