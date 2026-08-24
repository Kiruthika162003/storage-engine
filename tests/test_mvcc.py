from __future__ import annotations

import pytest

from store import mvcc as mod
from store.errors import Closed, ConfigError
from store.mvcc import FIRST, History, Snapshot
from store.record import Record


class TestWrites:
    def test_a_put_reads_back(self):
        made = History()
        made.put(b"a", b"1")
        assert made.value(b"a") == b"1"

    def test_an_empty_key_is_refused(self):
        with pytest.raises(ConfigError):
            History().put(b"", b"x")

    def test_the_first_sequence_is_first(self):
        made = History()
        assert made.put(b"a", b"1") == FIRST

    def test_sequences_climb_by_one(self):
        made = History()
        first = made.put(b"a", b"1")
        assert made.put(b"b", b"2") == first + 1

    def test_an_overwrite_keeps_both_versions(self):
        made = History()
        made.put(b"a", b"1")
        made.put(b"a", b"2")
        assert made.records == 2

    def test_the_newest_version_wins_a_read(self):
        made = History()
        made.put(b"a", b"1")
        made.put(b"a", b"2")
        assert made.value(b"a") == b"2"

    def test_a_delete_hides_the_key(self):
        made = History()
        made.put(b"a", b"1")
        made.delete(b"a")
        assert made.value(b"a") is None

    def test_a_delete_is_a_version(self):
        made = History()
        made.put(b"a", b"1")
        made.delete(b"a")
        assert made.records == 2

    def test_a_delete_takes_a_sequence(self):
        made = History()
        made.put(b"a", b"1")
        assert made.delete(b"a") == 2

    def test_a_missing_key_reads_as_nothing(self):
        assert History().value(b"a") is None

    def test_a_put_after_a_delete_revives_the_key(self):
        made = History()
        made.put(b"a", b"1")
        made.delete(b"a")
        made.put(b"a", b"2")
        assert made.value(b"a") == b"2"

    def test_the_key_count_ignores_versions(self):
        made = History()
        made.put(b"a", b"1")
        made.put(b"a", b"2")
        made.put(b"b", b"3")
        assert made.keys == 2


class TestSnapshots:
    def test_a_snapshot_names_the_current_sequence(self):
        made = History()
        made.put(b"a", b"1")
        assert made.snapshot().sequence == 1

    def test_a_snapshot_sees_what_was_there(self):
        made = History()
        made.put(b"a", b"1")
        held = made.snapshot()
        made.put(b"a", b"2")
        assert made.value(b"a", held) == b"1"

    def test_a_snapshot_does_not_see_later_deletes(self):
        made = History()
        made.put(b"a", b"1")
        held = made.snapshot()
        made.delete(b"a")
        assert made.value(b"a", held) == b"1"

    def test_a_snapshot_does_not_see_later_keys(self):
        made = History()
        held = made.snapshot()
        made.put(b"a", b"1")
        assert made.value(b"a", held) is None

    def test_a_snapshot_before_a_delete_sees_through_it(self):
        made = History()
        made.put(b"a", b"1")
        made.delete(b"a")
        held = made.snapshot()
        assert made.value(b"a", held) is None

    def test_two_snapshots_see_their_own_moments(self):
        made = History()
        made.put(b"a", b"1")
        early = made.snapshot()
        made.put(b"a", b"2")
        late = made.snapshot()
        assert made.value(b"a", early) == b"1" and made.value(b"a", late) == b"2"

    def test_snapshot_numbers_are_distinct(self):
        made = History()
        assert made.snapshot().number != made.snapshot().number

    def test_sees_is_a_sequence_comparison(self):
        held = Snapshot(sequence=5, number=1)
        assert held.sees(Record(key=b"a", sequence=5))
        assert not held.sees(Record(key=b"a", sequence=6))

    def test_release_closes_the_snapshot(self):
        made = History()
        held = made.snapshot()
        made.release(held)
        assert not made.open_snapshots

    def test_a_double_release_is_refused(self):
        made = History()
        held = made.snapshot()
        made.release(held)
        with pytest.raises(Closed):
            made.release(held)


class TestHorizon:
    def test_with_nothing_open_the_horizon_is_the_present(self):
        made = History()
        made.put(b"a", b"1")
        assert made.horizon == made.sequence

    def test_an_open_snapshot_pins_the_horizon(self):
        made = History()
        made.put(b"a", b"1")
        held = made.snapshot()
        made.put(b"a", b"2")
        assert made.horizon == held.sequence

    def test_the_oldest_snapshot_wins(self):
        made = History()
        made.put(b"a", b"1")
        oldest = made.snapshot()
        made.put(b"a", b"2")
        made.snapshot()
        assert made.horizon == oldest.sequence

    def test_releasing_the_oldest_moves_the_horizon(self):
        made = History()
        made.put(b"a", b"1")
        oldest = made.snapshot()
        made.put(b"a", b"2")
        newer = made.snapshot()
        made.release(oldest)
        assert made.horizon == newer.sequence


class TestCollect:
    def test_a_single_version_survives(self):
        made = History()
        made.put(b"a", b"1")
        assert made.collect() == 0 and made.records == 1

    def test_a_shadowed_version_goes(self):
        made = History()
        made.put(b"a", b"1")
        made.put(b"a", b"2")
        assert made.collect() == 1 and made.records == 1

    def test_the_surviving_version_is_the_newest(self):
        made = History()
        made.put(b"a", b"1")
        made.put(b"a", b"2")
        made.collect()
        assert made.value(b"a") == b"2"

    def test_a_bottom_tombstone_goes_with_its_key(self):
        made = History()
        made.put(b"a", b"1")
        made.delete(b"a")
        made.collect()
        assert made.records == 0 and made.keys == 0

    def test_a_pinned_version_stays(self):
        made = History()
        made.put(b"a", b"1")
        held = made.snapshot()
        made.put(b"a", b"2")
        made.collect()
        assert made.value(b"a", held) == b"1"

    def test_release_then_collect_frees_the_pinned_version(self):
        made = History()
        made.put(b"a", b"1")
        held = made.snapshot()
        made.put(b"a", b"2")
        made.release(held)
        made.collect()
        assert made.records == 1

    def test_collection_reports_what_went(self):
        made = History()
        for at in range(10):
            made.put(b"a", at.to_bytes(1, "big"))
        assert made.collect() == 9

    def test_a_tombstone_above_the_horizon_stays(self):
        made = History()
        made.put(b"a", b"1")
        held = made.snapshot()
        made.delete(b"a")
        made.collect()
        assert made.value(b"a", held) == b"1"
        made.release(held)

    def test_collection_leaves_reads_correct(self):
        made = mod._worked(2000, 100)
        wanted = {key: made.value(key) for key in list(made.versions)}
        made.collect()
        assert all(made.value(key) == value for key, value in wanted.items())


class TestCounters:
    def test_reads_are_counted(self):
        made = History()
        made.put(b"a", b"1")
        made.get(b"a")
        assert made.reads == 1

    def test_skips_count_versions_walked_past(self):
        made = History()
        made.put(b"a", b"1")
        held = made.snapshot()
        made.put(b"a", b"2")
        made.get(b"a", held)
        assert made.skipped == 1

    def test_a_present_read_skips_nothing(self):
        made = History()
        made.put(b"a", b"1")
        made.put(b"a", b"2")
        made.get(b"a")
        assert made.skipped == 0

    def test_as_dict_carries_the_horizon(self):
        made = History()
        made.put(b"a", b"1")
        assert made.as_dict()["horizon"] == 1

    def test_as_dict_counts_open_snapshots(self):
        made = History()
        made.snapshot()
        assert made.as_dict()["open"] == 1


class TestMeasurements:
    def test_a_snapshot_sees_the_moment(self):
        assert mod.a_snapshot_sees_the_store_as_it_stood()

    def test_open_snapshots_hold_collection(self):
        assert mod.an_open_snapshot_holds_back_collection()

    def test_the_horizon_is_the_minimum(self):
        assert mod.the_horizon_is_the_oldest_snapshot_not_the_average()

    def test_deep_reads_walk_the_versions(self):
        assert mod.a_read_of_the_present_skips_nothing_and_a_deep_read_skips_everything()

    def test_double_release_is_refused(self):
        assert mod.releasing_a_snapshot_twice_is_refused()

    def test_collection_is_idempotent(self):
        assert mod.collection_is_idempotent()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_six_claims(self):
        assert len(mod.summarise()) == 6

    def test_the_depth_table_deepens(self):
        rows = mod.compare_the_snapshot_depths()
        skips = [row["skipped"] for row in rows]
        assert skips == sorted(skips)

    def test_the_depth_table_has_four_rows(self):
        assert len(mod.compare_the_snapshot_depths()) == 4

    def test_the_worked_history_is_cached(self):
        assert mod._worked(100, 10) is mod._worked(100, 10)
