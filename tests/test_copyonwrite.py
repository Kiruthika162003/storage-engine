from __future__ import annotations

import pytest

from store.copyonwrite import (
    DEPTH,
    FANOUT,
    LEAVES,
    Cow,
    a_clustered_batch_pays_1_2_pages_per_edit,
    a_scattered_batch_shares_only_the_root,
    every_lone_update_writes_the_path,
    summarise,
    the_frozen_past_stays_readable_forever,
    the_tree_costs_4369_pages_and_the_snapshot_zero,
)


def small_tree() -> tuple[Cow, int]:
    cow = Cow()
    return cow, cow.build(list(range(LEAVES)))


class TestBuildAndRead:
    def test_every_leaf_reads_back(self):
        cow, root = small_tree()
        assert all(cow.read(root, leaf) == leaf for leaf in range(0, LEAVES, 313))

    def test_the_page_count_is_the_geometric_sum(self):
        cow, _ = small_tree()
        expected = sum(FANOUT**level for level in range(DEPTH))
        assert len(cow.pages) == expected

    def test_built_pages_are_frozen(self):
        cow, root = small_tree()
        assert cow.pages[root].frozen


class TestUpdate:
    def test_the_new_root_sees_the_edit(self):
        cow, root = small_tree()
        fresh = cow.update(root, 7, 700)
        assert cow.read(fresh, 7) == 700

    def test_the_old_root_does_not(self):
        cow, root = small_tree()
        cow.update(root, 7, 700)
        assert cow.read(root, 7) == 7

    def test_unrelated_leaves_are_shared_not_copied(self):
        cow, root = small_tree()
        fresh = cow.update(root, 7, 700)
        assert cow.read(fresh, 3000) == 3000

    def test_two_updates_write_two_paths(self):
        cow, root = small_tree()
        before = cow.written
        second = cow.update(root, 7, 700)
        cow.update(second, 8, 800)
        assert cow.written - before == 2 * DEPTH


class TestUpdateMany:
    def test_the_epoch_applies_every_edit(self):
        cow, root = small_tree()
        fresh = cow.update_many(root, {2: 20, 40: 400, 900: 9000})
        assert cow.read(fresh, 2) == 20
        assert cow.read(fresh, 40) == 400
        assert cow.read(fresh, 900) == 9000

    def test_the_epoch_leaves_the_past_alone(self):
        cow, root = small_tree()
        cow.update_many(root, {2: 20, 40: 400})
        assert cow.read(root, 2) == 2

    def test_the_epoch_refreezes_its_pages(self):
        cow, root = small_tree()
        fresh = cow.update_many(root, {2: 20})
        assert cow.pages[fresh].frozen

    def test_a_second_epoch_copies_again(self):
        cow, root = small_tree()
        second = cow.update_many(root, {2: 20})
        before = cow.written
        cow.update_many(second, {2: 21})
        assert cow.written - before == DEPTH

    def test_an_empty_epoch_writes_nothing(self):
        cow, root = small_tree()
        before = cow.written
        fresh = cow.update_many(root, {})
        assert cow.written == before and fresh == root


class TestClaims:
    @pytest.mark.parametrize(
        "claim",
        [
            the_tree_costs_4369_pages_and_the_snapshot_zero,
            every_lone_update_writes_the_path,
            a_clustered_batch_pays_1_2_pages_per_edit,
            a_scattered_batch_shares_only_the_root,
            the_frozen_past_stays_readable_forever,
        ],
    )
    def test_claim_holds(self, claim):
        assert claim() is True

    def test_summary_is_all_true(self):
        told = summarise()
        assert all(value for name, value in told.items() if name != "module")
