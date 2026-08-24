from __future__ import annotations

import random

import pytest

from store import engine as mod
from store.disk import Disk
from store.engine import FLUSH_AT, Store, build_table, crash
from store.errors import Closed, ConfigError
from store.record import Record
from store.wal import EVERY_RECORD, NEVER, Log


def filled(writes: int, keys: int, seed: int = 1, **kwargs) -> tuple[Store, dict]:
    source = random.Random(seed)
    store = Store(**kwargs)
    truth: dict[bytes, bytes] = {}
    for _ in range(writes):
        key = f"k{source.randrange(keys):05d}".encode()
        if source.random() < 0.1:
            store.delete(key)
            truth.pop(key, None)
        else:
            value = source.randbytes(12)
            store.put(key, value)
            truth[key] = value
    return store, truth


class TestTable:
    def test_an_empty_table_is_refused(self):
        with pytest.raises(ConfigError):
            build_table(1, [])

    def test_the_range_is_the_ends(self):
        made = build_table(1, [Record(key=b"a", sequence=1), Record(key=b"c", sequence=2)])
        assert made.first == b"a" and made.last == b"c"

    def test_a_key_in_range_might_be_held(self):
        made = build_table(1, [Record(key=b"a", sequence=1), Record(key=b"c", sequence=2)])
        assert made.might_hold(b"a")

    def test_a_key_out_of_range_is_never_held(self):
        made = build_table(1, [Record(key=b"b", sequence=1)])
        assert not made.might_hold(b"a") and not made.might_hold(b"c")

    def test_get_finds_a_key(self):
        made = build_table(1, [Record(key=b"a", sequence=1, value=b"x")])
        assert made.get(b"a").value == b"x"

    def test_get_misses_an_absent_key(self):
        made = build_table(1, [Record(key=b"a", sequence=1), Record(key=b"c", sequence=2)])
        assert made.get(b"b") is None

    def test_the_source_is_the_records(self):
        records = [Record(key=b"a", sequence=1)]
        assert build_table(1, records).source().records == records


class TestWrites:
    def test_a_put_reads_back(self):
        store = Store()
        store.put(b"a", b"1")
        assert store.get(b"a") == b"1"

    def test_an_empty_key_is_refused(self):
        with pytest.raises(ConfigError):
            Store().put(b"", b"x")

    def test_an_overwrite_reads_back_the_new_value(self):
        store = Store()
        store.put(b"a", b"1")
        store.put(b"a", b"2")
        assert store.get(b"a") == b"2"

    def test_a_delete_hides_the_key(self):
        store = Store()
        store.put(b"a", b"1")
        store.delete(b"a")
        assert store.get(b"a") is None

    def test_a_missing_key_reads_as_nothing(self):
        assert Store().get(b"a") is None

    def test_sequences_climb(self):
        store = Store()
        assert store.put(b"a", b"1") < store.put(b"b", b"2")

    def test_every_write_reaches_the_log_first(self):
        store = Store()
        store.put(b"a", b"1")
        assert store.wal.appended == 1

    def test_the_log_syncs_per_batch_by_default(self):
        store = Store()
        store.put(b"a", b"1")
        assert store.wal.at_risk == 0


class TestFlush:
    def test_a_full_memtable_flushes(self):
        store, _ = filled(FLUSH_AT + 200, 10000, 3)
        assert store.flushes >= 1

    def test_a_flush_makes_a_table(self):
        store = Store(flush_at=10)
        for at in range(10):
            store.put(f"k{at}".encode(), b"x")
        assert len(store.tables) >= 1

    def test_a_flush_announces_in_the_manifest(self):
        store = Store(flush_at=10)
        for at in range(10):
            store.put(f"k{at}".encode(), b"x")
        assert store.manifest.edits >= 1

    def test_a_flush_starts_a_fresh_log(self):
        store = Store(flush_at=10)
        for at in range(10):
            store.put(f"k{at}".encode(), b"x")
        assert store.wal.appended == 0

    def test_a_flush_empties_the_memtable(self):
        store = Store(flush_at=10)
        for at in range(10):
            store.put(f"k{at}".encode(), b"x")
        assert len(store.memtable.records()) == 0

    def test_an_empty_flush_does_nothing(self):
        store = Store()
        assert store.flush() is None and store.flushes == 0

    def test_reads_span_the_flush(self):
        store = Store(flush_at=10)
        for at in range(25):
            store.put(f"k{at:03d}".encode(), at.to_bytes(1, "big"))
        assert all(
            store.get(f"k{at:03d}".encode()) == at.to_bytes(1, "big") for at in range(25)
        )

    def test_file_numbers_are_distinct(self):
        store = Store(flush_at=5, fold_at=100)
        for at in range(30):
            store.put(f"k{at:03d}".encode(), b"x")
        numbers = [table.number for table in store.tables]
        assert len(numbers) == len(set(numbers))


class TestFold:
    def test_enough_tables_fold(self):
        store, _ = filled(6000, 1500, 4, flush_at=500, fold_at=3)
        assert store.folds >= 1

    def test_a_fold_leaves_one_table(self):
        store, _ = filled(3000, 800, 5, flush_at=300, fold_at=100)
        store.flush()
        store.fold()
        assert len(store.tables) == 1

    def test_a_fold_retires_the_inputs_in_the_manifest(self):
        store, _ = filled(3000, 800, 5, flush_at=300, fold_at=100)
        store.flush()
        before = set(store.manifest.version.files)
        store.fold()
        assert set(store.manifest.version.files) != before

    def test_a_fold_changes_no_answer(self):
        store, truth = filled(3000, 800, 6, flush_at=300, fold_at=100)
        store.flush()
        store.fold()
        assert all(store.get(key) == value for key, value in truth.items())

    def test_a_fold_drops_shadowed_versions(self):
        store = Store(flush_at=100, fold_at=100)
        for at in range(400):
            store.put(b"same", at.to_bytes(2, "big"))
        store.flush()
        store.fold()
        assert len(store.tables[0].records) == 1

    def test_a_fold_drops_dead_tombstones(self):
        store = Store(flush_at=100, fold_at=100)
        for at in range(200):
            store.put(f"k{at:03d}".encode(), b"x")
        for at in range(200):
            store.delete(f"k{at:03d}".encode())
        store.flush()
        folded = store.fold()
        assert folded is None and store.tables == []

    def test_a_single_table_does_not_fold(self):
        store = Store()
        store.put(b"a", b"1")
        store.flush()
        assert store.fold() is None


class TestScan:
    def test_a_scan_sees_the_memtable(self):
        store = Store()
        store.put(b"a", b"1")
        assert store.items() == [(b"a", b"1")]

    def test_a_scan_sees_the_files(self):
        store = Store(flush_at=5)
        for at in range(12):
            store.put(f"k{at:02d}".encode(), b"x")
        assert len(store.items()) == 12

    def test_a_scan_is_sorted(self):
        store, _ = filled(2000, 600, 7)
        keys = [key for key, _ in store.items()]
        assert keys == sorted(keys)

    def test_a_scan_hides_deletes(self):
        store = Store()
        store.put(b"a", b"1")
        store.put(b"b", b"2")
        store.delete(b"a")
        assert store.items() == [(b"b", b"2")]

    def test_a_scan_from_a_key_starts_there(self):
        store = Store()
        for key in (b"a", b"b", b"c"):
            store.put(key, b"x")
        assert [record.key for record in store.scan(b"b")] == [b"b", b"c"]

    def test_a_scan_agrees_with_the_dictionary(self):
        store, truth = filled(4000, 900, 8)
        assert dict(store.items()) == truth


class TestCrash:
    def test_a_crash_keeps_synced_writes(self):
        store, truth = filled(3000, 700, 9)
        survivor = crash(store)
        assert all(survivor.get(key) == value for key, value in truth.items())

    def test_a_crash_loses_unsynced_writes(self):
        store = Store(wal=Log(disk=Disk(name="WAL"), policy=NEVER))
        store.put(b"a", b"1")
        survivor = crash(store)
        assert survivor.get(b"a") is None

    def test_a_crash_keeps_per_record_synced_writes(self):
        store = Store(wal=Log(disk=Disk(name="WAL"), policy=EVERY_RECORD))
        store.put(b"a", b"1")
        survivor = crash(store)
        assert survivor.get(b"a") == b"1"

    def test_the_survivor_keeps_the_sequence(self):
        store, _ = filled(3000, 700, 9)
        top = store.sequence
        assert crash(store).sequence == top

    def test_the_survivor_accepts_new_writes(self):
        store, _ = filled(1000, 300, 10)
        survivor = crash(store)
        survivor.put(b"new", b"value")
        assert survivor.get(b"new") == b"value"

    def test_the_survivor_only_holds_manifest_files(self):
        store, _ = filled(3000, 700, 11)
        survivor = crash(store)
        live = set(survivor.manifest.version.files)
        assert {table.number for table in survivor.tables} <= live

    def test_a_second_crash_agrees_with_the_first(self):
        store, truth = filled(2000, 500, 12)
        once = crash(store)
        twice = crash(once)
        assert all(twice.get(key) == value for key, value in truth.items())


class TestClose:
    def test_a_closed_store_refuses_reads(self):
        store = Store()
        store.close()
        with pytest.raises(Closed):
            store.get(b"a")

    def test_a_closed_store_refuses_writes(self):
        store = Store()
        store.close()
        with pytest.raises(Closed):
            store.put(b"a", b"1")

    def test_close_flushes_first(self):
        store = Store()
        store.put(b"a", b"1")
        store.close()
        assert store.flushes == 1

    def test_as_dict_carries_the_counters(self):
        store, _ = filled(2000, 500, 13)
        made = store.as_dict()
        assert {"sequence", "tables", "flushes", "folds"} <= set(made)


class TestMeasurements:
    def test_the_store_agrees_with_a_dictionary(self):
        assert mod.the_store_agrees_with_a_dictionary()

    def test_a_crash_loses_nothing_synced(self):
        assert mod.a_crash_loses_nothing_that_was_synced()

    def test_a_fold_changes_no_answer(self):
        assert mod.a_fold_changes_no_answer()

    def test_the_filter_absorbs_the_misses(self):
        assert mod.the_filter_absorbs_the_misses()

    def test_closed_means_closed(self):
        assert mod.a_closed_store_refuses_everything()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5

    def test_the_filled_store_is_cached(self):
        assert mod._filled(100, 50, 1) is mod._filled(100, 50, 1)
