from __future__ import annotations

import random

from store import backup as mod
from store.backup import library_of, restore, take
from store.engine import Store


def worked(writes: int = 2000, keys: int = 400, seed: int = 5) -> tuple[Store, dict]:
    source = random.Random(seed)
    store = Store(flush_at=300, fold_at=4)
    truth: dict[bytes, bytes] = {}
    for _ in range(writes):
        key = f"k{source.randrange(keys):05d}".encode()
        if source.random() < 0.1:
            store.delete(key)
            truth.pop(key, None)
        else:
            value = source.randbytes(10)
            store.put(key, value)
            truth[key] = value
    return store, truth


class TestTake:
    def test_a_checkpoint_names_the_live_files(self):
        store, _ = worked()
        made = take(store)
        assert made.files == tuple(table.number for table in store.tables)

    def test_a_checkpoint_copies_the_memtable(self):
        store, _ = worked()
        made = take(store)
        assert list(made.memtable) == store.memtable.records()

    def test_a_checkpoint_carries_the_sequence(self):
        store, _ = worked()
        assert take(store).sequence == store.sequence

    def test_the_cost_excludes_the_files(self):
        store, _ = worked()
        held = sum(record.nbytes for table in store.tables for record in table.records)
        assert take(store).cost < held

    def test_an_empty_store_checkpoints(self):
        made = take(Store())
        assert made.files == () and made.cost == 0


class TestRestore:
    def test_a_restore_answers_the_truth(self):
        store, truth = worked()
        restored = restore(take(store), library_of(store))
        assert all(restored.get(key) == value for key, value in truth.items())

    def test_a_restore_misses_what_the_store_missed(self):
        store, truth = worked()
        restored = restore(take(store), library_of(store))
        keys = [f"k{at:05d}".encode() for at in range(400)]
        absent = [key for key in keys if key not in truth]
        assert all(restored.get(key) is None for key in absent[:50])

    def test_a_restore_accepts_new_writes(self):
        store, _ = worked()
        restored = restore(take(store), library_of(store))
        restored.put(b"fresh", b"value")
        assert restored.get(b"fresh") == b"value"

    def test_a_restore_continues_the_sequence(self):
        store, _ = worked()
        restored = restore(take(store), library_of(store))
        assert restored.put(b"fresh", b"value") == store.sequence + 1

    def test_a_restore_does_not_see_later_writes(self):
        store, _ = worked()
        library = library_of(store)
        made = take(store)
        store.put(b"later", b"write")
        assert restore(made, library).get(b"later") is None

    def test_the_restored_file_numbers_do_not_collide(self):
        store, _ = worked()
        restored = restore(take(store), library_of(store))
        for _ in range(400):
            restored.put(random.Random(1).randbytes(6), b"v")
        numbers = [table.number for table in restored.tables]
        assert len(numbers) == len(set(numbers))


class TestMeasurements:
    def test_a_checkpoint_is_cheap(self):
        assert mod.a_checkpoint_costs_kilobytes_on_a_store_of_megabytes()

    def test_writes_do_not_reach_it(self):
        assert mod.writes_after_the_checkpoint_do_not_reach_it()

    def test_a_restore_agrees(self):
        assert mod.a_restore_agrees_with_the_original_everywhere()

    def test_it_survives_the_crash(self):
        assert mod.the_checkpoint_survives_a_crash_of_the_live_store()

    def test_sharing_ends_at_the_fold(self):
        assert mod.sharing_between_checkpoints_lasts_exactly_until_a_fold()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5

    def test_the_worked_store_is_cached(self):
        assert mod._worked(100, 50, 1) is mod._worked(100, 50, 1)
