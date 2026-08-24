from __future__ import annotations

import random

import pytest

from store import vlog as mod
from store.errors import ConfigError, NotFound
from store.vlog import Separated, ValueLog


class TestValueLog:
    def test_an_append_returns_the_offset(self):
        log = ValueLog()
        assert log.append(b"first") == 0

    def test_offsets_advance_by_frame(self):
        log = ValueLog()
        log.append(b"12345")
        assert log.append(b"x") == 9

    def test_a_read_returns_the_value(self):
        log = ValueLog()
        at = log.append(b"hello")
        assert log.read(at) == b"hello"

    def test_an_empty_value_round_trips(self):
        log = ValueLog()
        at = log.append(b"")
        assert log.read(at) == b""

    def test_a_read_past_the_end_raises(self):
        with pytest.raises(NotFound):
            ValueLog().read(10)

    def test_retirement_counts_the_frame(self):
        log = ValueLog()
        at = log.append(b"12345")
        log.retire(at)
        assert log.dead_bytes == 9

    def test_reads_are_counted(self):
        log = ValueLog()
        at = log.append(b"v")
        log.read(at)
        log.read(at)
        assert log.reads == 2


class TestSeparated:
    def test_a_put_reads_back(self):
        made = Separated()
        made.put(b"k", b"value")
        assert made.get(b"k") == b"value"

    def test_an_empty_key_is_refused(self):
        with pytest.raises(ConfigError):
            Separated().put(b"", b"v")

    def test_a_missing_key_raises(self):
        with pytest.raises(NotFound):
            Separated().get(b"k")

    def test_an_overwrite_reads_the_new_value(self):
        made = Separated()
        made.put(b"k", b"old")
        made.put(b"k", b"new")
        assert made.get(b"k") == b"new"

    def test_an_overwrite_retires_the_old_value(self):
        made = Separated()
        made.put(b"k", b"12345")
        made.put(b"k", b"x")
        assert made.vlog.dead_bytes == 9

    def test_the_sorted_side_grows_by_key_and_pointer(self):
        made = Separated()
        made.put(b"key", bytes(1000))
        assert made.lsm_bytes_written == 3 + mod.POINTER_BYTES

    def test_the_log_grows_by_the_value(self):
        made = Separated()
        made.put(b"key", bytes(1000))
        assert made.log_bytes_written == 1004


class TestScan:
    def test_a_scan_returns_sorted_values(self):
        made = Separated()
        made.put(b"b", b"2")
        made.put(b"a", b"1")
        values, _ = made.scan([b"a", b"b"])
        assert values == [b"1", b"2"]

    def test_missing_keys_are_skipped(self):
        made = Separated()
        made.put(b"a", b"1")
        values, _ = made.scan([b"a", b"zz"])
        assert values == [b"1"]

    def test_in_order_writes_scan_without_jumps(self):
        made = Separated()
        keys = [f"k{at:03d}".encode() for at in range(50)]
        for key in keys:
            made.put(key, b"v")
        _, jumps = made.scan(keys)
        assert jumps == 0

    def test_reversed_writes_jump_everywhere(self):
        made = Separated()
        keys = [f"k{at:03d}".encode() for at in range(50)]
        for key in reversed(keys):
            made.put(key, b"v")
        _, jumps = made.scan(keys)
        assert jumps >= 48


class TestCollection:
    def test_collection_reclaims_the_dead(self):
        made = Separated()
        for _ in range(10):
            made.put(b"k", b"x" * 50)
        dead = made.vlog.dead_bytes
        assert made.collect() == dead

    def test_collection_resets_the_meter(self):
        made = Separated()
        for _ in range(10):
            made.put(b"k", b"x" * 50)
        made.collect()
        assert made.vlog.dead_bytes == 0

    def test_a_clean_log_collects_nothing(self):
        made = Separated()
        made.put(b"a", b"1")
        made.put(b"b", b"2")
        assert made.collect() == 0

    def test_reads_survive_collection(self):
        made = Separated()
        source = random.Random(5)
        truth = {}
        for _ in range(500):
            key = f"k{source.randrange(80):03d}".encode()
            value = source.randbytes(20)
            made.put(key, value)
            truth[key] = value
        made.collect()
        assert all(made.get(key) == value for key, value in truth.items())


class TestMeasurements:
    def test_separation_collapses_write_amp(self):
        assert mod.separation_collapses_write_amplification_for_large_values()

    def test_reads_hop_and_scans_seek(self):
        assert mod.reads_pay_one_extra_hop_and_scans_pay_seeks()

    def test_sequential_scans_are_free(self):
        assert mod.sequential_writes_scan_for_free()

    def test_the_log_rots_and_collects(self):
        assert mod.overwrites_rot_the_log_and_collection_reclaims_exactly()

    def test_collection_changes_nothing(self):
        assert mod.collection_changes_no_answer()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5
