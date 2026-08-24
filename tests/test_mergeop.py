from __future__ import annotations

import pytest

from store import mergeop as mod
from store.errors import BadFormat
from store.mergeop import Counters, fold, pack, unpack


class TestPacking:
    def test_a_counter_round_trips(self):
        assert unpack(pack(42)) == 42

    def test_a_negative_round_trips(self):
        assert unpack(pack(-7)) == -7

    def test_the_wrong_width_is_refused(self):
        with pytest.raises(BadFormat):
            unpack(b"\x01\x02")

    def test_fold_with_no_base_starts_at_zero(self):
        assert unpack(fold(None, [pack(3), pack(4)])) == 7

    def test_fold_with_a_base_adds_to_it(self):
        assert unpack(fold(pack(10), [pack(5)])) == 15

    def test_fold_of_nothing_is_the_base(self):
        assert unpack(fold(pack(9), [])) == 9


class TestCounters:
    def test_a_put_reads_back(self):
        made = Counters()
        made.put(b"k", 5)
        assert made.get(b"k") == 5

    def test_a_missing_counter_is_zero(self):
        assert Counters().get(b"k") == 0

    def test_an_add_accumulates(self):
        made = Counters()
        made.add(b"k", 3)
        made.add(b"k", 4)
        assert made.get(b"k") == 7

    def test_a_negative_add_subtracts(self):
        made = Counters()
        made.put(b"k", 10)
        made.add(b"k", -4)
        assert made.get(b"k") == 6

    def test_the_read_path_agrees_with_the_merge_path(self):
        reader, merger = Counters(), Counters()
        for delta in (5, -2, 9, 1):
            reader.add_by_reading(b"k", delta)
            merger.add(b"k", delta)
        assert reader.get(b"k") == merger.get(b"k") == 13

    def test_the_merge_path_does_not_read(self):
        made = Counters()
        for _ in range(50):
            made.add(b"k", 1)
        assert made.reads == 0

    def test_the_read_path_reads_every_time(self):
        made = Counters()
        for _ in range(50):
            made.add_by_reading(b"k", 1)
        assert made.reads == 50

    def test_the_depth_counts_pending_merges(self):
        made = Counters()
        made.put(b"k", 1)
        made.add(b"k", 1)
        made.add(b"k", 1)
        assert made.depth(b"k") == 3

    def test_compaction_flattens_the_depth(self):
        made = Counters()
        for _ in range(100):
            made.add(b"k", 1)
        made.compact(b"k")
        assert made.depth(b"k") == 1

    def test_compaction_keeps_the_value(self):
        made = Counters()
        made.put(b"k", 5)
        for _ in range(20):
            made.add(b"k", 2)
        made.compact(b"k")
        assert made.get(b"k") == 45

    def test_a_put_hides_older_merges(self):
        made = Counters()
        made.add(b"k", 100)
        made.put(b"k", 1)
        assert made.get(b"k") == 1

    def test_adds_after_a_put_stack_on_it(self):
        made = Counters()
        made.add(b"k", 100)
        made.put(b"k", 1)
        made.add(b"k", 2)
        assert made.get(b"k") == 3

    def test_as_dict_counts_the_folds(self):
        made = Counters()
        made.add(b"k", 1)
        made.add(b"k", 1)
        made.get(b"k")
        assert made.as_dict()["folds"] == 1


class TestMeasurements:
    def test_the_read_moves_not_shrinks(self):
        assert mod.the_merge_path_writes_what_the_read_path_reads()

    def test_reads_cancel_the_saving(self):
        assert mod.a_read_between_every_write_cancels_the_saving()

    def test_unread_counters_grow(self):
        assert mod.an_unread_counter_grows_without_bound_until_compaction()

    def test_a_put_is_a_wall(self):
        assert mod.a_put_cuts_the_fold_short()

    def test_folding_is_associative(self):
        assert mod.folding_is_associative_so_partial_folds_are_safe()

    def test_read_folding_is_quadratic(self):
        assert mod.read_time_folding_without_write_back_is_quadratic()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_six_claims(self):
        assert len(mod.summarise()) == 6

    def test_the_ratio_table_has_four_rows(self):
        assert len(mod.compare_the_ratios(500)) == 4

    def test_more_reads_fold_more_in_total(self):
        rows = mod.compare_the_ratios(500)
        folded = [row["records_folded"] for row in rows]
        assert folded == sorted(folded)
