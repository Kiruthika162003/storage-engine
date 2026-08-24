from __future__ import annotations

import pytest

from store import btreebulk as mod
from store.btreebulk import bulk_build, leaf_fills, measure
from store.errors import ConfigError


class TestBuild:
    def test_an_empty_build_is_refused(self):
        with pytest.raises(ConfigError):
            bulk_build([])

    def test_unsorted_pairs_are_refused(self):
        with pytest.raises(ConfigError):
            bulk_build([(b"b", b"2"), (b"a", b"1")])

    def test_duplicate_keys_are_refused(self):
        with pytest.raises(ConfigError):
            bulk_build([(b"a", b"1"), (b"a", b"2")])

    def test_an_impossible_fill_is_refused(self):
        with pytest.raises(ConfigError):
            bulk_build([(b"a", b"1")], fill=0.0)

    def test_one_pair_builds_one_leaf(self):
        tree = bulk_build([(b"a", b"1")])
        assert tree.pages == 1 and tree.get(b"a") == b"1"

    def test_every_key_reads_back(self):
        pairs = list(mod._pairs(3000))
        tree = bulk_build(pairs)
        assert all(tree.get(key) == value for key, value in pairs)

    def test_the_scan_is_sorted_and_complete(self):
        pairs = list(mod._pairs(3000))
        tree = bulk_build(pairs)
        assert tree.keys() == [key for key, _ in pairs]

    def test_absent_keys_read_absent(self):
        tree = bulk_build(list(mod._pairs(1000)))
        assert tree.get(b"zzz") is None

    def test_the_build_never_splits(self):
        tree = bulk_build(list(mod._pairs(5000)))
        assert tree.splits == 0

    def test_the_record_count_is_kept(self):
        assert bulk_build(list(mod._pairs(777))).records == 777

    def test_inserts_after_the_build_work(self):
        tree = bulk_build(list(mod._pairs(1000)))
        tree.put(b"zzz", b"new")
        assert tree.get(b"zzz") == b"new" and tree.records == 1001


class TestFills:
    def test_the_default_fill_lands_near_ninety(self):
        tree = bulk_build(list(mod._pairs(5000)))
        fills = leaf_fills(tree)
        assert 0.8 < sum(fills) / len(fills) < 0.95

    def test_a_half_fill_lands_near_fifty(self):
        tree = bulk_build(list(mod._pairs(5000)), fill=0.5)
        fills = leaf_fills(tree)
        assert 0.4 < sum(fills) / len(fills) < 0.6

    def test_a_full_fill_lands_at_one(self):
        tree = bulk_build(list(mod._pairs(6400)), fill=1.0)
        fills = leaf_fills(tree)
        assert sum(fills) / len(fills) == 1.0


class TestMeasure:
    def test_two_rows_come_back(self):
        rows = measure(2000)
        assert [row.method for row in rows] == ["insert", "bulk"]

    def test_the_bulk_row_is_denser(self):
        rows = {row.method: row for row in measure(2000)}
        assert rows["bulk"].mean_leaf_fill > rows["insert"].mean_leaf_fill

    def test_the_bulk_row_uses_fewer_pages(self):
        rows = {row.method: row for row in measure(2000)}
        assert rows["bulk"].pages < rows["insert"].pages


class TestMeasurements:
    def test_both_trees_answer_identically(self):
        assert mod.both_trees_answer_identically()

    def test_sorted_inserts_half_fill(self):
        assert mod.sorted_inserts_leave_half_empty_leaves()

    def test_the_build_writes_once(self):
        assert mod.the_bulk_build_never_splits_and_writes_once()

    def test_full_packs_split_immediately(self):
        assert mod.a_full_pack_splits_on_the_first_insert()

    def test_unsorted_is_refused(self):
        assert mod.unsorted_input_is_refused()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5
