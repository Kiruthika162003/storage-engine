from __future__ import annotations

import pytest

from store.bitcask import (
    ENTRY,
    KEY,
    Cask,
    Keydir,
    absent_keys_cost_no_seek_at_all,
    every_present_read_is_one_seek,
    summarise,
    the_keydir_rent_scales_with_count_not_size,
    the_log_stores_five_times_the_live_data,
    the_merge_reads_the_garbage_to_evict_it,
)


class TestKeydir:
    def test_memory_charges_per_entry(self):
        keydir = Keydir()
        keydir.offsets[b"a"] = (0, 0)
        keydir.offsets[b"b"] = (0, 1)
        assert keydir.memory() == 2 * ENTRY


class TestCask:
    def test_put_then_get(self):
        cask = Cask()
        cask.put(b"k", b"v")
        assert cask.get(b"k") == b"v"

    def test_an_overwrite_serves_the_newest(self):
        cask = Cask()
        cask.put(b"k", b"old")
        cask.put(b"k", b"new")
        assert cask.get(b"k") == b"new"

    def test_a_delete_is_a_miss_thereafter(self):
        cask = Cask()
        cask.put(b"k", b"v")
        cask.delete(b"k")
        assert cask.get(b"k") is None

    def test_a_miss_costs_no_seek(self):
        cask = Cask()
        cask.get(b"ghost")
        assert cask.seeks == 0

    def test_a_hit_costs_one_seek(self):
        cask = Cask()
        cask.put(b"k", b"v")
        cask.get(b"k")
        assert cask.seeks == 1

    def test_gets_survive_a_roll(self):
        cask = Cask()
        cask.put(b"k", b"v")
        cask.roll()
        cask.put(b"other", b"w")
        assert cask.get(b"k") == b"v"

    def test_stored_grows_and_live_does_not(self):
        cask = Cask()
        for _ in range(5):
            cask.put(b"k", b"v")
        assert cask.stored_bytes() == 5 * (KEY + 1)
        assert cask.live_bytes() == KEY + 1

    def test_merge_keeps_every_live_answer(self):
        cask = Cask()
        for number in range(50):
            cask.put(f"k{number}".encode(), str(number).encode())
        cask.delete(b"k7")
        cask.merge()
        assert cask.get(b"k8") == b"8"
        assert cask.get(b"k7") is None

    def test_merge_drops_deleted_and_stale_bytes(self):
        cask = Cask()
        for _ in range(10):
            cask.put(b"k", b"vvvv")
        cask.merge()
        assert cask.stored_bytes() == cask.live_bytes() == KEY + 4

    def test_merge_resets_the_append_meter(self):
        cask = Cask()
        cask.put(b"k", b"v")
        cask.merge()
        assert cask.appended_bytes == 0


class TestClaims:
    @pytest.mark.parametrize(
        "claim",
        [
            every_present_read_is_one_seek,
            absent_keys_cost_no_seek_at_all,
            the_log_stores_five_times_the_live_data,
            the_merge_reads_the_garbage_to_evict_it,
            the_keydir_rent_scales_with_count_not_size,
        ],
    )
    def test_claim_holds(self, claim):
        assert claim() is True

    def test_summary_is_all_true(self):
        told = summarise()
        assert all(value for name, value in told.items() if name != "module")
