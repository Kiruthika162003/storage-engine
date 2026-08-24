from __future__ import annotations

import pytest

from store.merkle import (
    FANOUT,
    Tree,
    _replica,
    a_hundred_bad_keys_share_their_paths,
    diff,
    honest_replicas_agree_in_one_probe_pair,
    one_bad_key_costs_104_probes_not_20000,
    summarise,
    the_diff_names_exactly_the_guilty_keys,
)


def tiny(values: dict[bytes, bytes]) -> Tree:
    return Tree.over(values)


class TestTree:
    def test_one_key_is_its_own_root(self):
        tree = tiny({b"a": b"1"})
        assert len(tree.levels) == 1
        assert tree.root() == tree.levels[0][0]

    def test_levels_shrink_by_fanout(self):
        tree = tiny({f"k{n:03d}".encode(): b"v" for n in range(FANOUT * 2)})
        assert [len(level) for level in tree.levels] == [FANOUT * 2, 2, 1]

    def test_equal_data_equal_root(self):
        values = {b"a": b"1", b"b": b"2"}
        assert tiny(values).root() == tiny(dict(values)).root()

    def test_a_changed_value_changes_the_root(self):
        one = tiny({b"a": b"1", b"b": b"2"})
        two = tiny({b"a": b"1", b"b": b"X"})
        assert one.root() != two.root()

    def test_a_swapped_pair_changes_the_root(self):
        one = tiny({b"a": b"1", b"b": b"2"})
        two = tiny({b"a": b"2", b"b": b"1"})
        assert one.root() != two.root()


class TestDiff:
    def test_equal_trees_diff_empty(self):
        values = {f"k{n:03d}".encode(): b"v" for n in range(100)}
        differing, probes = diff(tiny(values), tiny(dict(values)))
        assert differing == [] and probes == 2

    def test_one_change_is_found(self):
        values = {f"k{n:03d}".encode(): b"v" for n in range(100)}
        changed = dict(values)
        changed[b"k042"] = b"other"
        differing, _ = diff(tiny(values), tiny(changed))
        assert differing == [b"k042"]

    def test_every_key_changed_is_every_key_reported(self):
        values = {f"k{n:03d}".encode(): b"v" for n in range(50)}
        changed = dict.fromkeys(values, b"other")
        differing, _ = diff(tiny(values), tiny(changed))
        assert differing == sorted(values)

    def test_the_replica_generator_is_deterministic(self):
        assert _replica(7) == _replica(7)


class TestClaims:
    @pytest.mark.parametrize(
        "claim",
        [
            honest_replicas_agree_in_one_probe_pair,
            one_bad_key_costs_104_probes_not_20000,
            a_hundred_bad_keys_share_their_paths,
            the_diff_names_exactly_the_guilty_keys,
        ],
    )
    def test_claim_holds(self, claim):
        assert claim() is True

    def test_summary_is_all_true(self):
        told = summarise()
        assert all(value for name, value in told.items() if name != "module")
