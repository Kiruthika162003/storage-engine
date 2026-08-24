from __future__ import annotations

import pytest

from store import bulkload as mod
from store.bulkload import bulk_load
from store.engine import Store, crash
from store.errors import ConfigError, Conflict


class TestLoad:
    def test_an_empty_load_is_refused(self):
        with pytest.raises(ConfigError):
            bulk_load(Store(), [])

    def test_a_load_installs_files(self):
        store = Store()
        made = bulk_load(store, mod._sorted_records(5000))
        assert made.files == 3 and len(store.tables) == 3

    def test_the_file_size_is_respected(self):
        store = Store()
        bulk_load(store, mod._sorted_records(5000), file_records=1000)
        assert len(store.tables) == 5

    def test_loaded_keys_read_back(self):
        store = Store()
        bulk_load(store, mod._sorted_records(1000))
        assert store.get(b"bulk:00000500") is not None

    def test_unloaded_keys_stay_absent(self):
        store = Store()
        bulk_load(store, mod._sorted_records(1000))
        assert store.get(b"bulk:99999999") is None

    def test_the_manifest_records_every_file(self):
        store = Store()
        bulk_load(store, mod._sorted_records(5000))
        assert len(store.manifest.version.files) == 3

    def test_the_sequence_advances_past_the_load(self):
        store = Store()
        bulk_load(store, mod._sorted_records(1000, start_sequence=500))
        assert store.sequence >= 1499

    def test_writes_after_a_load_work(self):
        store = Store()
        bulk_load(store, mod._sorted_records(1000))
        store.put(b"zzz:after", b"v")
        assert store.get(b"zzz:after") == b"v"

    def test_the_log_stays_untouched(self):
        store = Store()
        made = bulk_load(store, mod._sorted_records(1000))
        assert made.log_bytes == 0


class TestRefusals:
    def test_unsorted_input_is_refused(self):
        records = mod._sorted_records(50)
        records.reverse()
        with pytest.raises(ConfigError):
            bulk_load(Store(), records)

    def test_duplicate_keys_are_refused(self):
        records = mod._sorted_records(50)
        records[10] = records[11]
        with pytest.raises(ConfigError):
            bulk_load(Store(), sorted(records, key=lambda one: one.key))

    def test_an_overlap_with_files_is_refused(self):
        store = Store(flush_at=10)
        for at in range(20):
            store.put(f"bulk:{at:08d}".encode(), b"live")
        with pytest.raises(Conflict):
            bulk_load(store, mod._sorted_records(100))

    def test_an_overlap_with_the_memtable_is_refused(self):
        store = Store()
        store.put(b"bulk:00000005", b"live")
        with pytest.raises(Conflict):
            bulk_load(store, mod._sorted_records(100))

    def test_a_refused_load_changes_nothing(self):
        store = Store()
        records = mod._sorted_records(50)
        records.reverse()
        with pytest.raises(ConfigError):
            bulk_load(store, records)
        assert not store.tables and store.manifest.edits == 0

    def test_a_disjoint_load_is_accepted_beside_live_keys(self):
        store = Store()
        store.put(b"live:1", b"v")
        made = bulk_load(store, mod._sorted_records(100))
        assert made.files == 1


class TestCrash:
    def test_loaded_data_survives(self):
        store = Store()
        bulk_load(store, mod._sorted_records(2000))
        survivor = crash(store)
        assert survivor.get(b"bulk:00001000") is not None

    def test_the_survivor_replays_nothing(self):
        store = Store()
        bulk_load(store, mod._sorted_records(2000))
        survivor = crash(store)
        assert len(survivor.memtable.records()) == 0


class TestMeasurements:
    def test_no_log_no_memtable(self):
        assert mod.a_load_writes_no_log_and_fills_no_memtable()

    def test_loads_survive_crashes(self):
        assert mod.loaded_records_read_back_and_survive_a_crash()

    def test_unsorted_is_refused(self):
        assert mod.unsorted_input_is_refused()

    def test_overlap_is_refused(self):
        assert mod.an_overlapping_load_is_refused()

    def test_disjoint_loads_coexist(self):
        assert mod.a_disjoint_load_lands_beside_live_data()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5
