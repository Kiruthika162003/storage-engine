from __future__ import annotations

import pytest

from store import hashlog as mod
from store.errors import ConfigError, NotFound
from store.hashlog import HashLog


class TestPutGet:
    def test_a_put_reads_back(self):
        made = HashLog()
        made.put(b"k", b"value")
        assert made.get(b"k") == b"value"

    def test_an_empty_key_is_refused(self):
        with pytest.raises(ConfigError):
            HashLog().put(b"", b"v")

    def test_a_missing_key_raises(self):
        with pytest.raises(NotFound):
            HashLog().get(b"k")

    def test_an_overwrite_reads_the_new_value(self):
        made = HashLog()
        made.put(b"k", b"old")
        made.put(b"k", b"new")
        assert made.get(b"k") == b"new"

    def test_an_empty_value_round_trips(self):
        made = HashLog()
        made.put(b"k", b"")
        assert made.get(b"k") == b""

    def test_binary_values_round_trip(self):
        made = HashLog()
        made.put(b"k", bytes(range(256)))
        assert made.get(b"k") == bytes(range(256))

    def test_every_key_of_a_large_set_reads_back(self):
        made = HashLog()
        for at in range(5000):
            made.put(f"k{at:05d}".encode(), at.to_bytes(4, "big"))
        assert all(
            made.get(f"k{at:05d}".encode()) == at.to_bytes(4, "big") for at in range(5000)
        )

    def test_the_log_only_grows_on_put(self):
        made = HashLog()
        made.put(b"a", b"1")
        first = len(made.log)
        made.put(b"a", b"2")
        assert len(made.log) > first


class TestDelete:
    def test_a_deleted_key_raises(self):
        made = HashLog()
        made.put(b"k", b"v")
        made.delete(b"k")
        with pytest.raises(NotFound):
            made.get(b"k")

    def test_deleting_a_missing_key_raises(self):
        with pytest.raises(NotFound):
            HashLog().delete(b"k")

    def test_a_delete_leaves_the_bytes_as_garbage(self):
        made = HashLog()
        made.put(b"k", b"v")
        before = len(made.log)
        made.delete(b"k")
        assert len(made.log) == before and made.dead_bytes > 0

    def test_a_reinsert_after_delete_works(self):
        made = HashLog()
        made.put(b"k", b"v1")
        made.delete(b"k")
        made.put(b"k", b"v2")
        assert made.get(b"k") == b"v2"


class TestCompaction:
    def test_compaction_reclaims_dead_bytes(self):
        made = HashLog()
        for _ in range(10):
            made.put(b"k", b"x" * 50)
        assert made.compact() > 0 and made.dead_bytes == 0

    def test_compaction_keeps_every_live_key(self):
        made = HashLog()
        for at in range(100):
            made.put(f"k{at:03d}".encode(), at.to_bytes(2, "big"))
        made.compact()
        assert made.keys == 100

    def test_a_clean_log_compacts_to_itself(self):
        made = HashLog()
        made.put(b"a", b"1")
        made.put(b"b", b"2")
        assert made.compact() == 0

    def test_reads_work_after_compaction(self):
        made = HashLog()
        made.put(b"a", b"1")
        made.put(b"a", b"2")
        made.compact()
        assert made.get(b"a") == b"2"


class TestMeters:
    def test_log_reads_are_counted(self):
        made = HashLog()
        made.put(b"k", b"v")
        made.get(b"k")
        made.get(b"k")
        assert made.log_reads == 2

    def test_dead_bytes_track_overwrites(self):
        made = HashLog()
        made.put(b"k", b"v")
        assert made.dead_bytes == 0
        made.put(b"k", b"w")
        assert made.dead_bytes > 0

    def test_as_dict_carries_the_wall(self):
        made = HashLog()
        made.put(b"key", b"v")
        assert made.as_dict()["index_bytes"] == len(b"key") + 8


class TestMeasurements:
    def test_point_reads_are_flat(self):
        assert mod.a_point_read_costs_one_log_read_at_any_size()

    def test_the_index_is_the_wall(self):
        assert mod.the_index_holds_every_key_which_is_the_designs_wall()

    def test_overwrites_rot_the_log(self):
        assert mod.overwrites_rot_the_log_until_compaction()

    def test_compaction_changes_nothing(self):
        assert mod.compaction_changes_no_answer()

    def test_there_is_no_scan(self):
        assert mod.there_is_no_scan_and_that_is_the_price_of_the_flat_read()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5
