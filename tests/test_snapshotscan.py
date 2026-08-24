from __future__ import annotations

from store import snapshotscan as mod
from store.snapshotscan import pin, sequence_only_items


class TestPin:
    def test_a_pin_reports_the_current_contents(self):
        store = mod._seeded(50)
        assert pin(store).items() == store.items()

    def test_a_pin_ignores_later_puts(self):
        store = mod._seeded(50)
        handle = pin(store)
        store.put(b"zzz", b"new")
        assert all(key != b"zzz" for key, _ in handle.items())

    def test_a_pin_ignores_later_deletes(self):
        store = mod._seeded(50)
        handle = pin(store)
        store.delete(b"k0010")
        assert any(key == b"k0010" for key, _ in handle.items())

    def test_a_pin_ignores_later_overwrites(self):
        store = mod._seeded(50)
        handle = pin(store)
        store.put(b"k0010", b"changed")
        held = dict(handle.items())
        assert held[b"k0010"] == b"v"

    def test_a_pin_of_an_empty_store_is_empty(self):
        from store.engine import Store

        assert pin(Store()).items() == []

    def test_two_reads_of_one_pin_agree(self):
        store = mod._seeded(50)
        handle = pin(store)
        store.put(b"k0001", b"changed")
        assert handle.items() == handle.items()

    def test_a_pin_sees_through_a_flush(self):
        store = mod._seeded(80)
        handle = pin(store)
        store.flush()
        assert len(handle.items()) == 80

    def test_a_pin_sees_files_and_memtable_together(self):
        store = mod._seeded(80)
        store.flush()
        for at in range(80, 120):
            store.put(f"k{at:04d}".encode(), b"v")
        assert len(pin(store).items()) == 120


class TestSequenceOnly:
    def test_it_agrees_when_nothing_moved(self):
        store = mod._seeded(50)
        assert sequence_only_items(store, store.sequence) == store.items()

    def test_it_loses_overwritten_keys(self):
        store = mod._seeded(50)
        pinned = store.sequence
        store.put(b"k0010", b"new")
        held = dict(sequence_only_items(store, pinned))
        assert b"k0010" not in held


class TestMeasurements:
    def test_the_accidental_snapshot(self):
        assert mod.the_engine_scan_is_a_snapshot_by_accident()

    def test_sequence_alone_is_unsound(self):
        assert mod.the_sequence_only_pin_is_unsound_here()

    def test_the_sound_pin_holds(self):
        assert mod.the_sound_pin_reports_one_moment_exactly()

    def test_two_pins_two_stories(self):
        assert mod.two_pins_tell_two_consistent_stories()

    def test_pins_survive_flushes(self):
        assert mod.a_pin_survives_a_flush_underneath()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5
