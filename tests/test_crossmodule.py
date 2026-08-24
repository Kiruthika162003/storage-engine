from __future__ import annotations

import random

from store.backup import library_of, restore, take
from store.bulkload import _sorted_records, bulk_load
from store.engine import Store, crash
from store.snapshotscan import pin
from store.verify.invariants import report
from store.writebatch import Batch, commit


class TestBulkloadMeetsTheRest:
    def test_a_bulk_loaded_store_checkpoints_and_restores(self):
        store = Store()
        bulk_load(store, _sorted_records(3000))
        saved = take(store)
        restored = restore(saved, library_of(store))
        assert restored.get(b"bulk:00001500") is not None

    def test_a_bulk_loaded_store_passes_the_invariants(self):
        store = Store()
        bulk_load(store, _sorted_records(3000))
        assert report(store)["clean"]

    def test_a_bulk_loaded_store_survives_a_crash_then_writes(self):
        store = Store()
        bulk_load(store, _sorted_records(2000))
        survivor = crash(store)
        survivor.put(b"zzz:new", b"v")
        assert survivor.get(b"zzz:new") == b"v"
        assert report(survivor)["clean"]


class TestBatchesMeetPinsAndCrashes:
    def test_a_pin_taken_before_a_batch_misses_it(self):
        store = Store(flush_at=10**9, fold_at=10**9)
        commit(store, Batch().put(b"a", b"1"))
        handle = pin(store)
        commit(store, Batch().put(b"b", b"2").delete(b"a"))
        held = dict(handle.items())
        assert held == {b"a": b"1"}

    def test_a_batched_store_crashes_clean(self):
        store = Store(flush_at=10**9, fold_at=10**9)
        for at in range(50):
            gone = f"k{at - 1:03d}".encode() if at else b"none"
            commit(store, Batch().put(f"k{at:03d}".encode(), b"v").delete(gone))
        survivor = crash(store)
        assert survivor.get(b"k049") == b"v"
        assert report(survivor)["clean"]


class TestMixedLifecycles:
    def test_flush_fold_pin_crash_in_one_story(self):
        source = random.Random(5)
        store = Store(flush_at=200, fold_at=3)
        truth = {}
        for _ in range(1500):
            key = f"k{source.randrange(300):04d}".encode()
            value = source.randbytes(8)
            store.put(key, value)
            truth[key] = value
        handle = pin(store)
        before = dict(handle.items())
        store.flush()
        store.fold()
        survivor = crash(store)
        assert dict(handle.items()) == before
        assert all(survivor.get(key) == value for key, value in truth.items())
        assert report(survivor)["clean"]
