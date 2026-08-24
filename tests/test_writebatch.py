from __future__ import annotations

import pytest

from store import writebatch as mod
from store.engine import Store, crash
from store.errors import Closed, ConfigError
from store.writebatch import Batch, commit, recover_batched


def fresh() -> Store:
    return Store(flush_at=10**9, fold_at=10**9)


class TestBatch:
    def test_puts_accumulate(self):
        made = Batch().put(b"a", b"1").put(b"b", b"2")
        assert made.operations == 2

    def test_deletes_accumulate(self):
        made = Batch().delete(b"a")
        assert made.operations == 1

    def test_an_empty_key_is_refused(self):
        with pytest.raises(ConfigError):
            Batch().put(b"", b"v")

    def test_the_builder_chains(self):
        made = Batch().put(b"a", b"1").delete(b"a").put(b"b", b"2")
        assert made.operations == 3

    def test_insertion_order_is_kept(self):
        made = Batch().put(b"a", b"1").delete(b"b").put(b"c", b"3")
        kinds = [kind for kind, _, _ in made.ops]
        assert kinds == ["put", "delete", "put"]


class TestCommit:
    def test_a_commit_applies_every_put(self):
        store = fresh()
        commit(store, Batch().put(b"a", b"1").put(b"b", b"2"))
        assert store.get(b"a") == b"1" and store.get(b"b") == b"2"

    def test_a_commit_applies_deletes(self):
        store = fresh()
        commit(store, Batch().put(b"a", b"1"))
        commit(store, Batch().delete(b"a"))
        assert store.get(b"a") is None

    def test_an_empty_batch_is_refused(self):
        with pytest.raises(ConfigError):
            commit(fresh(), Batch())

    def test_a_committed_batch_cannot_grow(self):
        store = fresh()
        batch = Batch().put(b"a", b"1")
        commit(store, batch)
        with pytest.raises(Closed):
            batch.put(b"b", b"2")

    def test_a_committed_batch_cannot_recommit(self):
        store = fresh()
        batch = Batch().put(b"a", b"1")
        commit(store, batch)
        with pytest.raises(Closed):
            commit(store, batch)

    def test_the_whole_batch_shares_one_frame(self):
        store = fresh()
        commit(store, Batch().put(b"a", b"1").put(b"b", b"2").put(b"c", b"3"))
        assert store.wal.disk.writes == 1

    def test_the_commit_syncs(self):
        store = fresh()
        commit(store, Batch().put(b"a", b"1"))
        assert store.wal.disk.at_risk == 0

    def test_sequences_are_consecutive_within_a_batch(self):
        store = fresh()
        commit(store, Batch().put(b"a", b"1").put(b"b", b"2"))
        held = {record.key: record.sequence for record in store.memtable.records()}
        assert held[b"b"] == held[b"a"] + 1

    def test_put_then_delete_deletes(self):
        store = fresh()
        commit(store, Batch().put(b"k", b"v").delete(b"k"))
        assert store.get(b"k") is None

    def test_delete_then_put_survives(self):
        store = fresh()
        batch = Batch()
        batch.delete(b"k")
        batch.put(b"k", b"v")
        commit(store, batch)
        assert store.get(b"k") == b"v"


class TestRecovery:
    def test_a_committed_batch_survives_a_crash(self):
        store = fresh()
        commit(store, Batch().put(b"a", b"1").put(b"b", b"2"))
        survivor = crash(store)
        assert survivor.get(b"a") == b"1" and survivor.get(b"b") == b"2"

    def test_recover_batched_reads_whole_frames(self):
        store = fresh()
        commit(store, Batch().put(b"a", b"1").put(b"b", b"2"))
        found = recover_batched(store.wal.disk.read())
        assert len(found) == 2

    def test_a_torn_frame_recovers_to_the_batch_before(self):
        store = fresh()
        commit(store, Batch().put(b"a", b"1"))
        commit(store, Batch().put(b"b", b"2").put(b"c", b"3"))
        raw = store.wal.disk.read()
        assert len(recover_batched(raw[:-4])) == 1

    def test_an_empty_log_recovers_to_nothing(self):
        assert recover_batched(b"") == []


class TestMeasurements:
    def test_no_crash_splits_a_batch(self):
        assert mod.no_crash_point_splits_a_batch()

    def test_commits_keep_batches_whole(self):
        assert mod.a_crash_after_commit_keeps_the_whole_batch()

    def test_order_within_is_preserved(self):
        assert mod.batch_order_within_is_preserved()

    def test_committed_batches_refuse_reuse(self):
        assert mod.a_committed_batch_refuses_reuse()

    def test_empty_batches_are_refused(self):
        assert mod.an_empty_batch_is_refused()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5
