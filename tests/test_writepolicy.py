from __future__ import annotations

import pytest

from store.writepolicy import (
    CHECKPOINT,
    Cache,
    _storm,
    a_flush_before_the_crash_closes_the_window,
    recovery_reads_the_disk_not_the_memory,
    summarise,
    the_crash_bill_is_the_checkpoint_window,
    write_back_pays_a_fifth_of_the_io,
)


class TestWriteThrough:
    def test_every_write_reaches_the_disk(self):
        cache = Cache(policy="through")
        cache.write(1, 100)
        assert cache.disk[1] == 100 and cache.disk_writes == 1

    def test_a_crash_loses_nothing(self):
        cache = Cache(policy="through")
        for page in range(5):
            cache.write(page, page)
        assert cache.crash() == 0

    def test_rewrites_pay_every_time(self):
        cache = Cache(policy="through")
        for value in range(10):
            cache.write(1, value)
        assert cache.disk_writes == 10


class TestWriteBack:
    def test_a_write_only_dirties_the_page(self):
        cache = Cache(policy="back")
        cache.write(1, 100)
        assert cache.disk_writes == 0 and 1 in cache.dirty

    def test_rewrites_coalesce_into_one_flush(self):
        cache = Cache(policy="back")
        for value in range(10):
            cache.write(1, value)
        cache.flush()
        assert cache.disk_writes == 1 and cache.disk[1] == 9

    def test_a_crash_loses_the_dirty_pages(self):
        cache = Cache(policy="back")
        cache.write(1, 100)
        cache.write(2, 200)
        assert cache.crash() == 2

    def test_a_flushed_cache_survives_the_crash(self):
        cache = Cache(policy="back")
        cache.write(1, 100)
        cache.flush()
        assert cache.crash() == 0 and cache.held[1] == 100

    def test_the_checkpoint_fires_on_schedule(self):
        cache = Cache(policy="back")
        cache.write(1, 100)
        for _ in range(CHECKPOINT):
            cache.tick()
        assert cache.disk_writes == 1 and not cache.dirty

    def test_crash_resets_memory_to_the_disk_image(self):
        cache = Cache(policy="back")
        cache.write(1, 100)
        cache.flush()
        cache.write(1, 999)
        cache.crash()
        assert cache.held[1] == 100


class TestStorm:
    def test_the_storm_is_deterministic(self):
        assert _storm("back").disk_writes == _storm("back").disk_writes

    def test_both_policies_see_the_same_final_state(self):
        through = _storm("through")
        back = _storm("back")
        back.flush()
        assert through.disk == back.disk


class TestClaims:
    @pytest.mark.parametrize(
        "claim",
        [
            write_back_pays_a_fifth_of_the_io,
            the_crash_bill_is_the_checkpoint_window,
            a_flush_before_the_crash_closes_the_window,
            recovery_reads_the_disk_not_the_memory,
        ],
    )
    def test_claim_holds(self, claim):
        assert claim() is True

    def test_summary_is_all_true(self):
        told = summarise()
        assert all(value for name, value in told.items() if name != "module")
