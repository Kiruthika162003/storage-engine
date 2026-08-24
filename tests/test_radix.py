from __future__ import annotations

import pytest

from store import radix as mod
from store.errors import ConfigError
from store.radix import Radix, stored_bytes
from store.record import Record


def one(key: bytes, sequence: int = 1, value: bytes = b"v") -> Record:
    return Record(key=key, sequence=sequence, value=value)


def grown(keys) -> Radix:
    made = Radix()
    for at, key in enumerate(keys):
        made.put(one(key, at + 1))
    return made


class TestPut:
    def test_an_empty_key_is_refused(self):
        with pytest.raises(ConfigError):
            Radix().put(one(b""))

    def test_a_put_reads_back(self):
        made = grown([b"hello"])
        assert made.get(b"hello").value == b"v"

    def test_two_keys_with_a_shared_prefix(self):
        made = grown([b"car", b"cart"])
        assert made.get(b"car") and made.get(b"cart")

    def test_a_key_that_is_a_prefix_of_another(self):
        made = grown([b"cart", b"car"])
        assert made.get(b"car") and made.get(b"cart")

    def test_diverging_keys_split(self):
        made = grown([b"apple", b"apply"])
        assert made.get(b"apple") and made.get(b"apply")

    def test_the_record_count_tracks_distinct_keys(self):
        made = grown([b"a", b"b", b"a"])
        assert made.records == 2

    def test_an_overwrite_takes_the_newer_sequence(self):
        made = Radix()
        made.put(one(b"k", 1, b"old"))
        made.put(one(b"k", 2, b"new"))
        assert made.get(b"k").value == b"new"

    def test_a_stale_overwrite_is_ignored(self):
        made = Radix()
        made.put(one(b"k", 2, b"new"))
        made.put(one(b"k", 1, b"old"))
        assert made.get(b"k").value == b"new"


class TestGet:
    def test_a_missing_key_is_none(self):
        assert grown([b"hello"]).get(b"world") is None

    def test_a_prefix_of_a_stored_key_is_none(self):
        assert grown([b"hello"]).get(b"hel") is None

    def test_an_extension_of_a_stored_key_is_none(self):
        assert grown([b"hel"]).get(b"hello") is None

    def test_every_key_of_a_large_set_reads_back(self):
        keys = mod._prefixed_keys(2000)
        made = grown(keys)
        assert all(made.get(key) is not None for key in keys)

    def test_binary_keys_work(self):
        keys = [bytes([a, b]) for a in range(0, 250, 50) for b in range(0, 250, 50)]
        made = grown(keys)
        assert all(made.get(key) is not None for key in keys)


class TestScan:
    def test_the_scan_is_sorted(self):
        keys = list(mod._random_keys(1000))
        made = grown(keys)
        assert [record.key for record in made.scan()] == sorted(keys)

    def test_the_scan_holds_everything(self):
        keys = list(mod._prefixed_keys(1000))
        made = grown(keys)
        assert len(list(made.scan())) == 1000

    def test_an_empty_tree_scans_to_nothing(self):
        assert list(Radix().scan()) == []


class TestShape:
    def test_the_node_count_is_at_least_the_keys(self):
        made = grown(list(mod._prefixed_keys(500)))
        assert made.nodes() >= 500

    def test_stored_bytes_are_less_than_raw_bytes_for_shared_keys(self):
        keys = mod._prefixed_keys(2000)
        made = grown(keys)
        assert stored_bytes(made) < sum(len(key) for key in keys)

    def test_lookups_count_steps(self):
        made = grown([b"abc"])
        before = made.steps
        made.get(b"abc")
        assert made.steps > before


class TestMeasurements:
    def test_agrees_with_the_skiplist(self):
        assert mod.both_structures_agree_on_contents_and_order()

    def test_prefixes_save_bytes_not_nodes(self):
        assert mod.prefixes_save_bytes_and_cost_nodes_which_was_half_expected()

    def test_lookups_cost_the_key(self):
        assert mod.lookup_cost_is_the_key_not_the_population()

    def test_overwrites_keep_the_newest(self):
        assert mod.an_overwrite_keeps_the_newest_sequence()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4

    def test_the_shape_table_has_two_rows(self):
        rows = mod.compare_the_shapes()
        assert [row["keys"] for row in rows] == ["prefixed", "random"]

    def test_random_keys_build_fewer_nodes(self):
        rows = {row["keys"]: row for row in mod.compare_the_shapes()}
        assert rows["random"]["nodes_per_key"] < rows["prefixed"]["nodes_per_key"]

    def test_the_key_sets_are_cached(self):
        assert mod._prefixed_keys(100) is mod._prefixed_keys(100)
