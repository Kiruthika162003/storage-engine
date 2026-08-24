from __future__ import annotations

import itertools

import pytest

from store import iterator as mod
from store.errors import ConfigError
from store.iterator import Compacting, Merge, Source, naive, resurrects
from store.record import DELETE, MERGE, PUT, Record


def one(key: bytes, sequence: int = 1, value: bytes = b"v", kind: int = PUT) -> Record:
    return Record(key=key, sequence=sequence, kind=kind, value=value)


def source(name: str, records: list[Record]) -> Source:
    return Source(name=name, records=sorted(records, key=lambda record: record.order))


def keys(records) -> list[bytes]:
    return [record.key for record in records]


class TestSource:
    def test_a_sorted_source_is_accepted(self):
        assert len(Source(name="s", records=[one(b"a"), one(b"b")])) == 2

    def test_an_unsorted_source_is_refused(self):
        with pytest.raises(ConfigError):
            Source(name="s", records=[one(b"b"), one(b"a")])

    def test_an_empty_source_is_accepted(self):
        assert len(Source(name="s", records=[])) == 0

    def test_a_source_out_of_sequence_order_is_refused(self):
        with pytest.raises(ConfigError):
            Source(name="s", records=[one(b"a", 1), one(b"a", 2)])

    def test_the_same_key_newest_first_is_accepted(self):
        assert len(Source(name="s", records=[one(b"a", 2), one(b"a", 1)])) == 2

    def test_a_scan_gives_everything(self):
        made = Source(name="s", records=[one(b"a"), one(b"b")])
        assert len(list(made.scan())) == 2

    def test_a_scan_from_a_key_skips_below_it(self):
        made = Source(name="s", records=[one(b"a"), one(b"b"), one(b"c")])
        assert keys(made.scan(b"b")) == [b"b", b"c"]

    def test_a_scan_past_the_end_gives_nothing(self):
        made = Source(name="s", records=[one(b"a")])
        assert list(made.scan(b"z")) == []

    def test_the_name_is_kept(self):
        assert Source(name="level-0", records=[]).name == "level-0"


class TestMerge:
    def test_a_merge_needs_a_source(self):
        with pytest.raises(ConfigError):
            Merge(sources=[])

    def test_one_source_merges_to_itself(self):
        made = source("s", [one(b"a"), one(b"b")])
        assert list(Merge(sources=[made]).newest()) == made.records

    def test_two_disjoint_sources_interleave(self):
        left = source("l", [one(b"a"), one(b"c")])
        right = source("r", [one(b"b"), one(b"d")])
        assert keys(Merge(sources=[left, right]).newest()) == [b"a", b"b", b"c", b"d"]

    def test_the_newest_version_wins(self):
        old = source("o", [one(b"a", 1, b"old")])
        new = source("n", [one(b"a", 9, b"new")])
        assert next(iter(Merge(sources=[old, new]).newest())).value == b"new"

    def test_the_order_of_the_sources_does_not_matter(self):
        old = source("o", [one(b"a", 1, b"old")])
        new = source("n", [one(b"a", 9, b"new")])
        first = list(Merge(sources=[old, new]).newest())
        second = list(Merge(sources=[new, old]).newest())
        assert first == second

    def test_an_empty_source_contributes_nothing(self):
        made = source("s", [one(b"a")])
        empty = Source(name="e", records=[])
        assert keys(Merge(sources=[made, empty]).newest()) == [b"a"]

    def test_every_source_empty_gives_nothing(self):
        empty = Source(name="e", records=[])
        assert list(Merge(sources=[empty, empty]).newest()) == []

    def test_the_output_is_sorted(self):
        made = list(mod._sources(4, 500))
        assert keys(Merge(sources=made).newest()) == sorted(keys(Merge(sources=made).newest()))

    def test_the_output_has_no_repeated_key(self):
        made = list(mod._sources(4, 500))
        found = keys(Merge(sources=made).newest())
        assert len(found) == len(set(found))

    def test_the_dropped_count_is_what_is_missing(self):
        made = list(mod._sources(4, 500))
        merge = Merge(sources=made)
        kept = len(list(merge.newest()))
        assert kept + merge.dropped == merge.total

    def test_the_total_is_the_sum_of_the_sources(self):
        made = list(mod._sources(3, 400))
        assert Merge(sources=made).total == sum(len(one) for one in made)

    def test_a_scan_from_a_key_skips_below_it(self):
        made = list(mod._sources(3, 400))
        found = keys(Merge(sources=made).newest(b"k000000500"))
        assert all(key >= b"k000000500" for key in found)

    def test_as_dict_carries_the_counters(self):
        made = list(mod._sources(2, 200))
        merge = Merge(sources=made)
        list(merge.newest())
        assert merge.as_dict()["dropped"] == merge.dropped


class TestAgainstTheReference:
    def reference(self, sources):
        made = []
        seen = None
        for record in naive(sources):
            if record.key != seen:
                made.append(record)
                seen = record.key
        return made

    def test_two_sources_agree(self):
        made = list(mod._sources(2, 1000))
        assert list(Merge(sources=made).newest()) == self.reference(made)

    def test_four_sources_agree(self):
        made = list(mod._sources(4, 1000))
        assert list(Merge(sources=made).newest()) == self.reference(made)

    def test_sixteen_sources_agree(self):
        made = list(mod._sources(16, 300))
        assert list(Merge(sources=made).newest()) == self.reference(made)

    def test_disjoint_sources_agree(self):
        made = list(mod._sources(4, 500, overlap=0.02))
        assert list(Merge(sources=made).newest()) == self.reference(made)

    def test_fully_overlapping_sources_agree(self):
        made = list(mod._sources(4, 500, overlap=1.0))
        assert list(Merge(sources=made).newest()) == self.reference(made)

    def test_the_raw_stream_holds_every_record(self):
        made = list(mod._sources(4, 500))
        merge = Merge(sources=made)
        assert len(list(merge.raw())) == merge.total

    def test_the_raw_stream_is_ordered(self):
        made = list(mod._sources(4, 500))
        found = [record.order for _, record in Merge(sources=made).raw()]
        assert found == sorted(found)

    def test_the_raw_stream_names_a_real_source(self):
        made = list(mod._sources(4, 200))
        assert all(0 <= at < 4 for at, _ in Merge(sources=made).raw())

    def test_a_scan_agrees_with_the_reference(self):
        made = list(mod._sources(4, 500))
        start = b"k000000900"
        wanted = [record for record in self.reference(made) if record.key >= start]
        assert list(Merge(sources=made).newest(start)) == wanted


class TestLive:
    def test_a_tombstone_is_hidden(self):
        made = source("s", [one(b"a", 2, b"", DELETE), one(b"a", 1)])
        assert list(Merge(sources=[made]).live()) == []

    def test_a_put_above_a_tombstone_is_visible(self):
        made = source("s", [one(b"a", 3), one(b"a", 2, b"", DELETE), one(b"a", 1)])
        assert len(list(Merge(sources=[made]).live())) == 1

    def test_a_tombstone_hides_an_older_source(self):
        old = source("o", [one(b"a", 1)])
        new = source("n", [one(b"a", 2, b"", DELETE)])
        assert list(Merge(sources=[old, new]).live()) == []

    def test_other_keys_are_not_hidden(self):
        made = source("s", [one(b"a", 2, b"", DELETE), one(b"b", 1)])
        assert keys(Merge(sources=[made]).live()) == [b"b"]

    def test_get_finds_a_live_key(self):
        made = source("s", [one(b"a", 1, b"x")])
        assert Merge(sources=[made]).get(b"a").value == b"x"

    def test_get_misses_a_deleted_key(self):
        made = source("s", [one(b"a", 2, b"", DELETE), one(b"a", 1)])
        assert Merge(sources=[made]).get(b"a") is None

    def test_get_misses_an_absent_key(self):
        made = source("s", [one(b"a"), one(b"c")])
        assert Merge(sources=[made]).get(b"b") is None

    def test_get_misses_past_the_end(self):
        made = source("s", [one(b"a")])
        assert Merge(sources=[made]).get(b"z") is None

    def test_get_takes_the_newest_across_sources(self):
        old = source("o", [one(b"a", 1, b"old")])
        new = source("n", [one(b"a", 5, b"new")])
        assert Merge(sources=[old, new]).get(b"a").value == b"new"

    def test_a_merge_record_is_not_hidden(self):
        made = source("s", [one(b"a", 1, b"1", MERGE)])
        assert len(list(Merge(sources=[made]).live())) == 1


class TestCompacting:
    def test_a_non_bottom_compaction_keeps_the_tombstone(self):
        made = source("s", [one(b"a", 2, b"", DELETE)])
        out = Compacting(merge=Merge(sources=[made]), bottom=False)
        assert len(list(out.records())) == 1

    def test_a_bottom_compaction_drops_the_tombstone(self):
        made = source("s", [one(b"a", 2, b"", DELETE)])
        out = Compacting(merge=Merge(sources=[made]), bottom=True, horizon=1 << 62)
        assert list(out.records()) == []

    def test_a_bottom_compaction_drops_the_put_underneath_too(self):
        made = source("s", [one(b"a", 2, b"", DELETE), one(b"a", 1)])
        out = Compacting(merge=Merge(sources=[made]), bottom=True, horizon=1 << 62)
        assert list(out.records()) == []

    def test_a_horizon_below_the_tombstone_keeps_it(self):
        made = source("s", [one(b"a", 9, b"", DELETE), one(b"a", 1)])
        out = Compacting(merge=Merge(sources=[made]), bottom=True, horizon=5)
        assert len(list(out.records())) == 1

    def test_a_horizon_above_the_tombstone_drops_it(self):
        made = source("s", [one(b"a", 3, b"", DELETE)])
        out = Compacting(merge=Merge(sources=[made]), bottom=True, horizon=9)
        assert list(out.records()) == []

    def test_a_put_is_never_dropped(self):
        made = source("s", [one(b"a", 3)])
        out = Compacting(merge=Merge(sources=[made]), bottom=True, horizon=1 << 62)
        assert len(list(out.records())) == 1

    def test_the_kept_count_matches_the_output(self):
        made = list(mod._sources(3, 400))
        out = Compacting(merge=Merge(sources=made), bottom=True, horizon=1 << 62)
        found = len(list(out.records()))
        assert out.kept == found

    def test_the_removed_count_matches_the_tombstones(self):
        made = source("s", [one(b"a", 2, b"", DELETE), one(b"b", 3, b"", DELETE)])
        out = Compacting(merge=Merge(sources=[made]), bottom=True, horizon=1 << 62)
        list(out.records())
        assert out.removed == 2

    def test_as_dict_carries_the_bottom_flag(self):
        out = Compacting(merge=Merge(sources=[source("s", [one(b"a")])]), bottom=True)
        assert out.as_dict()["bottom"] is True

    def test_as_dict_carries_the_horizon(self):
        out = Compacting(merge=Merge(sources=[source("s", [one(b"a")])]), horizon=7)
        assert out.as_dict()["horizon"] == 7

    def test_the_compacted_output_is_sorted(self):
        made = list(mod._sources(4, 400))
        out = Compacting(merge=Merge(sources=made), bottom=True, horizon=1 << 62)
        found = keys(out.records())
        assert found == sorted(found)


class TestResurrection:
    def test_dropping_a_tombstone_too_early_brings_the_key_back(self):
        older = source("o", [one(b"a", 1, b"x")])
        newer = source("n", [one(b"a", 2, b"", DELETE)])
        assert resurrects(older, newer)

    def test_nothing_comes_back_when_the_older_source_is_empty(self):
        older = Source(name="o", records=[])
        newer = source("n", [one(b"a", 2, b"", DELETE)])
        assert not resurrects(older, newer)

    def test_nothing_comes_back_when_the_key_differs(self):
        older = source("o", [one(b"b", 1)])
        newer = source("n", [one(b"a", 2, b"", DELETE)])
        assert not resurrects(older, newer)

    def test_nothing_comes_back_without_a_tombstone(self):
        older = source("o", [one(b"a", 1)])
        newer = source("n", [one(b"a", 2)])
        assert not resurrects(older, newer)

    def test_several_keys_come_back(self):
        older = source("o", [one(b"a", 1), one(b"b", 2)])
        newer = source("n", [one(b"a", 3, b"", DELETE), one(b"b", 4, b"", DELETE)])
        assert resurrects(older, newer)


class TestMeasurements:
    def test_the_heap_agrees_with_the_sort(self):
        assert mod.the_heap_merge_agrees_with_sorting_everything()

    def test_the_newest_wins(self):
        assert mod.the_newest_version_wins_and_the_older_one_is_never_looked_at()

    def test_a_compaction_may_not_drop_the_tombstone(self):
        assert mod.a_read_hides_a_tombstone_and_a_compaction_may_not()

    def test_the_bottom_may_drop_it(self):
        assert mod.a_bottom_compaction_can_drop_the_tombstone_safely()

    def test_the_horizon_holds_it_back(self):
        assert mod.a_horizon_keeps_a_tombstone_a_reader_still_needs()

    def test_the_cost_is_logarithmic(self):
        assert mod.the_heap_cost_grows_with_the_log_of_the_source_count()

    def test_overlap_makes_the_work(self):
        assert mod.the_merge_drops_more_when_the_sources_overlap_more()

    def test_an_unsorted_source_is_refused(self):
        assert mod.a_source_that_is_not_sorted_is_refused_when_it_is_built()

    def test_an_empty_merge_is_refused(self):
        assert mod.a_merge_with_no_sources_is_refused()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_nine_claims(self):
        assert len(mod.summarise()) == 9

    def test_the_source_count_table_has_six_rows(self):
        assert len(mod.compare_the_source_counts(2400)) == 6

    def test_the_source_count_table_grows_the_comparisons(self):
        rows = mod.compare_the_source_counts(2400)
        found = [row["comparisons_per_record"] for row in rows]
        assert found == sorted(found)

    def test_the_source_count_table_grows_the_drops(self):
        rows = mod.compare_the_source_counts(2400)
        assert rows[-1]["dropped"] > rows[0]["dropped"]

    def test_one_source_drops_nothing(self):
        assert mod.compare_the_source_counts(2400)[0]["dropped"] == 0

    def test_the_overlap_table_has_five_rows(self):
        assert len(mod.compare_the_overlaps(3, 400)) == 5

    def test_the_overlap_table_wastes_more_as_it_overlaps(self):
        rows = mod.compare_the_overlaps(3, 400)
        found = [row["wasted"] for row in rows]
        assert found == sorted(found)

    def test_the_cached_sources_are_shared(self):
        assert mod._sources(2, 100) is mod._sources(2, 100)

    def test_the_cached_sources_are_sorted(self):
        for made in mod._sources(3, 200):
            assert keys(made.records) == sorted(keys(made.records))

    def test_the_naive_merge_holds_everything(self):
        made = list(mod._sources(3, 300))
        assert len(list(naive(made))) == sum(len(one) for one in made)

    def test_the_naive_merge_from_a_key_skips_below_it(self):
        made = list(mod._sources(3, 300))
        assert all(record.key >= b"k000000500" for record in naive(made, b"k000000500"))

    def test_grouping_the_naive_merge_gives_the_distinct_keys(self):
        made = list(mod._sources(3, 300))
        grouped = [key for key, _ in itertools.groupby(naive(made), lambda one: one.key)]
        assert grouped == keys(Merge(sources=made).newest())
