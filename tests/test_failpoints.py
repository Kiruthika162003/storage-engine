from __future__ import annotations

import pytest

from store import failpoints as mod
from store.engine import Store
from store.failpoints import Fall, FlushWithFailpoints, survivor_after


class TestFlushSteps:
    def test_an_unfailed_flush_walks_every_step(self):
        store = Store(flush_at=10**9, fold_at=10**9)
        store.put(b"k", b"v")
        flusher = FlushWithFailpoints(store=store)
        flusher.run()
        assert flusher.steps == [
            "before_anything",
            "file_written",
            "table_installed",
            "manifest_synced",
            "log_dropped",
        ]

    def test_an_unfailed_flush_counts(self):
        store = Store(flush_at=10**9, fold_at=10**9)
        store.put(b"k", b"v")
        FlushWithFailpoints(store=store).run()
        assert store.flushes == 1

    def test_an_empty_memtable_flushes_nowhere(self):
        store = Store(flush_at=10**9, fold_at=10**9)
        flusher = FlushWithFailpoints(store=store)
        flusher.run()
        assert flusher.steps == []

    def test_a_failpoint_raises_at_its_step(self):
        store = Store(flush_at=10**9, fold_at=10**9)
        store.put(b"k", b"v")
        flusher = FlushWithFailpoints(store=store, fail_at="file_written")
        with pytest.raises(Fall):
            flusher.run()
        assert flusher.steps[-1] == "file_written"

    def test_a_failed_flush_does_not_count(self):
        store = Store(flush_at=10**9, fold_at=10**9)
        store.put(b"k", b"v")
        flusher = FlushWithFailpoints(store=store, fail_at="manifest_synced")
        with pytest.raises(Fall):
            flusher.run()
        assert store.flushes == 0


class TestSurvivors:
    def test_every_named_window_recovers_the_truth(self):
        for fail_at in (
            "before_anything",
            "file_written",
            "table_installed",
            "manifest_synced",
            "log_dropped",
        ):
            survivor, truth = survivor_after(fail_at, writes=100)
            assert all(survivor.get(key) == value for key, value in truth.items()), fail_at

    def test_the_orphan_file_never_surfaces(self):
        survivor, _ = survivor_after("file_written", writes=100)
        assert len(survivor.tables) == 0

    def test_the_synced_manifest_keeps_the_file(self):
        survivor, _ = survivor_after("manifest_synced", writes=100)
        assert len(survivor.tables) == 1

    def test_the_dropped_log_replays_nothing(self):
        survivor, _ = survivor_after("log_dropped", writes=100)
        assert len(survivor.memtable.records()) == 0


class TestMeasurements:
    def test_before_anything_replays(self):
        assert mod.a_crash_before_anything_replays_the_log()

    def test_orphan_files_are_harmless(self):
        assert mod.a_crash_after_the_file_is_written_is_harmless_duplication()

    def test_double_durability_is_boring(self):
        assert mod.a_crash_after_the_manifest_syncs_needs_no_log()

    def test_after_the_drop_the_file_carries(self):
        assert mod.a_crash_after_the_log_drops_still_answers()

    def test_all_windows_agree(self):
        assert mod.every_window_answers_identically()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5
