from __future__ import annotations

import pytest

from store import btree as mod
from store.btree import (
    INTERIOR_KEYS,
    LEAF_RECORDS,
    Comparison,
    Interior,
    Leaf,
    Tree,
    compare,
)
from store.errors import ConfigError


def grown(count: int, start: int = 0) -> Tree:
    made = Tree()
    for at in range(start, start + count):
        made.put(f"k{at:08d}".encode(), at.to_bytes(4, "little"))
    return made


class TestLeaf:
    def test_a_fresh_leaf_is_empty(self):
        assert len(Leaf()) == 0

    def test_a_put_installs(self):
        leaf = Leaf()
        leaf.put(b"a", b"1")
        assert leaf.get(b"a") == b"1"

    def test_a_missing_key_is_nothing(self):
        assert Leaf().get(b"a") is None

    def test_puts_keep_the_keys_sorted(self):
        leaf = Leaf()
        for key in (b"c", b"a", b"b"):
            leaf.put(key, b"x")
        assert leaf.keys == [b"a", b"b", b"c"]

    def test_an_overwrite_does_not_grow(self):
        leaf = Leaf()
        assert leaf.put(b"a", b"1") and not leaf.put(b"a", b"2")

    def test_an_overwrite_changes_the_value(self):
        leaf = Leaf()
        leaf.put(b"a", b"1")
        leaf.put(b"a", b"2")
        assert leaf.get(b"a") == b"2"

    def test_a_remove_takes_the_key_out(self):
        leaf = Leaf()
        leaf.put(b"a", b"1")
        assert leaf.remove(b"a") and leaf.get(b"a") is None

    def test_removing_a_missing_key_reports_it(self):
        assert not Leaf().remove(b"a")

    def test_full_arrives_at_the_limit(self):
        leaf = Leaf()
        for at in range(LEAF_RECORDS):
            leaf.put(at.to_bytes(4, "big"), b"x")
        assert leaf.full

    def test_a_split_halves_the_leaf(self):
        leaf = Leaf()
        for at in range(LEAF_RECORDS):
            leaf.put(at.to_bytes(4, "big"), b"x")
        _, right = leaf.split()
        assert len(leaf) == len(right) == LEAF_RECORDS // 2

    def test_the_separator_is_the_right_halfs_first_key(self):
        leaf = Leaf()
        for at in range(LEAF_RECORDS):
            leaf.put(at.to_bytes(4, "big"), b"x")
        separator, right = leaf.split()
        assert separator == right.keys[0]

    def test_the_halves_do_not_overlap(self):
        leaf = Leaf()
        for at in range(LEAF_RECORDS):
            leaf.put(at.to_bytes(4, "big"), b"x")
        _, right = leaf.split()
        assert leaf.keys[-1] < right.keys[0]


class TestInterior:
    def test_child_for_routes_below_the_first_separator(self):
        node = Interior(separators=[b"m"], children=["left", "right"])
        assert node.child_for(b"a") == 0

    def test_child_for_routes_at_the_separator_rightwards(self):
        node = Interior(separators=[b"m"], children=["left", "right"])
        assert node.child_for(b"m") == 1

    def test_child_for_routes_above_the_separator(self):
        node = Interior(separators=[b"m"], children=["left", "right"])
        assert node.child_for(b"z") == 1

    def test_install_keeps_one_more_child_than_separators(self):
        node = Interior(separators=[b"m"], children=["a", "b"])
        node.install(1, b"t", "c")
        assert len(node.children) == len(node.separators) + 1

    def test_a_split_promotes_the_middle(self):
        node = Interior(
            separators=[bytes([at]) for at in range(INTERIOR_KEYS)],
            children=list(range(INTERIOR_KEYS + 1)),
        )
        promoted, right = node.split()
        assert promoted not in node.separators and promoted not in right.separators

    def test_a_split_keeps_the_child_balance(self):
        node = Interior(
            separators=[bytes([at]) for at in range(INTERIOR_KEYS)],
            children=list(range(INTERIOR_KEYS + 1)),
        )
        _, right = node.split()
        assert len(node.children) == len(node.separators) + 1
        assert len(right.children) == len(right.separators) + 1


class TestTree:
    def test_a_fresh_tree_is_a_single_leaf(self):
        assert Tree().height == 1

    def test_an_empty_key_is_refused(self):
        with pytest.raises(ConfigError):
            Tree().put(b"", b"x")

    def test_a_put_reads_back(self):
        tree = Tree()
        tree.put(b"a", b"1")
        assert tree.get(b"a") == b"1"

    def test_a_missing_key_is_nothing(self):
        assert grown(100).get(b"zzz") is None

    def test_every_key_reads_back(self):
        tree = grown(5000)
        assert all(
            tree.get(f"k{at:08d}".encode()) == at.to_bytes(4, "little") for at in range(5000)
        )

    def test_the_keys_come_back_sorted(self):
        tree = grown(3000)
        assert tree.keys() == sorted(tree.keys())

    def test_the_record_count_is_kept(self):
        assert grown(3000).records == 3000

    def test_an_overwrite_does_not_grow_the_count(self):
        tree = grown(10)
        tree.put(b"k00000005", b"new")
        assert tree.records == 10

    def test_an_overwrite_reads_back(self):
        tree = grown(10)
        tree.put(b"k00000005", b"new")
        assert tree.get(b"k00000005") == b"new"

    def test_a_remove_shrinks_the_count(self):
        tree = grown(100)
        tree.remove(b"k00000050")
        assert tree.records == 99

    def test_a_removed_key_is_gone(self):
        tree = grown(100)
        tree.remove(b"k00000050")
        assert tree.get(b"k00000050") is None

    def test_removing_a_missing_key_reports_it(self):
        assert not grown(10).remove(b"zzz")

    def test_the_height_grows_with_the_records(self):
        assert grown(20000).height > grown(10).height

    def test_a_scan_gives_everything(self):
        tree = grown(2000)
        assert len(list(tree.scan())) == 2000

    def test_a_scan_from_a_key_gives_the_tail(self):
        tree = grown(2000)
        found = [key for key, _ in tree.scan(b"k00001000")]
        assert found == [f"k{at:08d}".encode() for at in range(1000, 2000)]

    def test_a_scan_from_past_the_end_gives_nothing(self):
        assert list(grown(100).scan(b"z")) == []

    def test_reverse_insertion_still_reads_back(self):
        tree = Tree()
        for at in reversed(range(3000)):
            tree.put(f"k{at:08d}".encode(), b"x")
        assert tree.records == 3000 and tree.keys() == sorted(tree.keys())

    def test_shuffled_insertion_still_reads_back(self):
        tree = Tree()
        for key in mod._keys(3000):
            tree.put(key, b"x")
        assert all(tree.get(key) is not None for key in mod._keys(3000))

    def test_page_writes_count_every_put(self):
        tree = grown(1000)
        assert tree.page_writes >= 1000

    def test_page_reads_count_the_path(self):
        tree = grown(20000)
        before = tree.page_reads
        tree.get(b"k00000005")
        assert tree.page_reads - before == tree.height

    def test_as_dict_carries_the_shape(self):
        made = grown(1000).as_dict()
        assert {"records", "height", "pages", "splits"} <= set(made)


class TestBalance:
    def test_every_leaf_is_at_the_same_depth(self):
        tree = grown(20000)
        depths = set()

        def walk(node, depth):
            if isinstance(node, Leaf):
                depths.add(depth)
                return
            for child in node.children:
                walk(child, depth + 1)

        walk(tree.root, 1)
        assert len(depths) == 1

    def test_the_root_split_adds_the_level(self):
        tree = Tree()
        at = 0
        while tree.height == 1:
            tree.put(at.to_bytes(4, "big"), b"x")
            at += 1
        assert tree.height == 2 and isinstance(tree.root, Interior)

    def test_separators_route_correctly_after_growth(self):
        tree = grown(20000)
        assert tree.get(b"k00000000") is not None
        assert tree.get(b"k00019999") is not None


class TestComparison:
    def test_the_ratio_divides_the_bytes(self):
        made = Comparison(writes=10, tree_page_writes=10, lsm_records_written=10)
        assert made.ratio == round(made.tree_bytes / made.lsm_bytes, 2)

    def test_the_tree_bytes_count_pages(self):
        made = Comparison(writes=1, tree_page_writes=3, lsm_records_written=1)
        assert made.tree_bytes == 3 * mod.PAGE_BYTES

    def test_the_lsm_bytes_count_records(self):
        made = Comparison(writes=1, tree_page_writes=1, lsm_records_written=5)
        assert made.lsm_bytes == 5 * mod.RECORD_BYTES

    def test_as_dict_carries_the_ratio(self):
        assert "ratio" in compare(4000, 2000).as_dict()


class TestMeasurements:
    def test_the_tree_moves_more_bytes(self):
        assert mod.the_tree_moves_six_times_the_bytes_for_the_same_writes()

    def test_the_tree_reads_one_path(self):
        assert mod.the_tree_reads_one_path_and_the_lsm_reads_every_run()

    def test_splits_climb_rarely(self):
        assert mod.a_split_climbs_and_the_climb_is_rare()

    def test_balance_is_structural(self):
        assert mod.the_tree_stays_balanced_without_being_told_to()

    def test_overwrites_are_the_tree_case(self):
        assert mod.an_overwrite_is_free_in_a_tree_and_a_new_version_in_a_log()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5

    def test_the_record_size_table_has_six_rows(self):
        assert len(mod.compare_the_record_sizes()) == 6

    def test_the_ratio_falls_as_records_grow(self):
        rows = mod.compare_the_record_sizes()
        ratios = [row["tree_over_lsm"] for row in rows]
        assert ratios == sorted(ratios, reverse=True)

    def test_the_crossover_is_inside_the_table(self):
        rows = mod.compare_the_record_sizes()
        assert rows[0]["tree_over_lsm"] > 1.0 > rows[-1]["tree_over_lsm"]

    def test_the_cached_tree_is_shared(self):
        assert mod._tree(1000) is mod._tree(1000)

    def test_the_cached_keys_are_distinct(self):
        keys = mod._keys(5000)
        assert len(set(keys)) == 5000
