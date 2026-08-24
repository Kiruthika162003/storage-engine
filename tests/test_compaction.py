from __future__ import annotations

import pytest

from store import compaction as mod
from store.compaction import (
    FAN_OUT,
    FLUSH_RECORDS,
    RUNS_PER_LEVEL,
    Levelled,
    Load,
    Run,
    Tiered,
    Work,
    amplification,
    batches,
    read_cost,
    run_load,
    stale,
)
from store.errors import ConfigError
from store.iterator import Merge
from store.record import DELETE, PUT, Record


def one(key: bytes, sequence: int = 1, value: bytes = b"v", kind: int = PUT) -> Record:
    return Record(key=key, sequence=sequence, kind=kind, value=value)


def span(low: int, high: int, base: int = 0) -> Run:
    return Run(
        records=[
            one(f"k{at:09d}".encode(), base + at + 1) for at in range(low, high)
        ]
    )


def live_of(policy) -> dict:
    sources = [
        run.source(f"{at}-{one}")
        for at, level in enumerate(policy.levels)
        for one, run in enumerate(level)
    ]
    if not sources:
        return {}
    return {record.key: record for record in Merge(sources=sources).live()}


def expected(records: list[Record]) -> dict:
    held: dict[bytes, Record] = {}
    for record in records:
        if record.kind == DELETE:
            held.pop(record.key, None)
        else:
            held[record.key] = record
    return held


class TestRun:
    def test_a_sorted_run_is_accepted(self):
        assert len(span(0, 10)) == 10

    def test_an_unsorted_run_is_refused(self):
        with pytest.raises(ConfigError):
            Run(records=[one(b"b"), one(b"a")])

    def test_a_negative_level_is_refused(self):
        with pytest.raises(ConfigError):
            Run(records=[], level=-1)

    def test_the_first_key_is_the_lowest(self):
        assert span(3, 10).first == b"k000000003"

    def test_the_last_key_is_the_highest(self):
        assert span(3, 10).last == b"k000000009"

    def test_nbytes_adds_the_records(self):
        made = span(0, 10)
        assert made.nbytes == sum(record.nbytes for record in made.records)

    def test_overlapping_runs_overlap(self):
        assert span(0, 10).overlaps(span(5, 15))

    def test_disjoint_runs_do_not_overlap(self):
        assert not span(0, 10).overlaps(span(20, 30))

    def test_touching_runs_overlap(self):
        assert span(0, 10).overlaps(span(9, 20))

    def test_adjacent_runs_do_not_overlap(self):
        assert not span(0, 10).overlaps(span(10, 20))

    def test_an_empty_run_overlaps_nothing(self):
        assert not Run(records=[]).overlaps(span(0, 10))

    def test_overlap_is_symmetric(self):
        left, right = span(0, 10), span(5, 15)
        assert left.overlaps(right) == right.overlaps(left)

    def test_a_run_becomes_a_source(self):
        assert len(span(0, 10).source("s")) == 10

    def test_as_dict_carries_the_level(self):
        assert Run(records=[one(b"a")], level=3).as_dict()["level"] == 3

    def test_as_dict_carries_the_range(self):
        made = span(2, 8).as_dict()
        assert made["first"] == "k000000002" and made["last"] == "k000000007"


class TestLevelled:
    def test_a_fan_out_below_two_is_refused(self):
        with pytest.raises(ConfigError):
            Levelled(fan_out=1)

    def test_a_fresh_store_holds_nothing(self):
        assert Levelled().records == 0

    def test_one_flush_lands_at_level_zero(self):
        store = Levelled()
        store.flush(span(0, 10).records)
        assert len(store.levels[0]) == 1

    def test_a_second_flush_pushes_the_first_down(self):
        store = Levelled()
        store.flush(span(0, 10).records)
        store.flush(span(20, 30).records)
        assert len(store.levels[0]) == 1

    def test_level_zero_never_holds_two_runs(self):
        store = run_load(Levelled(), Load(keys=2000, writes=8000))
        assert len(store.levels[0]) <= 1

    def test_a_level_below_zero_holds_disjoint_runs(self):
        store = run_load(Levelled(), Load(keys=5000, writes=20000))
        for level in store.levels[1:]:
            for at in range(len(level) - 1):
                assert not level[at].overlaps(level[at + 1])

    def test_the_runs_at_a_level_are_ordered(self):
        store = run_load(Levelled(), Load(keys=5000, writes=20000))
        for level in store.levels[1:]:
            firsts = [run.first for run in level]
            assert firsts == sorted(firsts)

    def test_the_capacity_grows_by_the_fan_out(self):
        store = Levelled(fan_out=10)
        assert store.capacity(2) == store.capacity(1) * 10

    def test_the_capacity_at_zero_is_the_flush_size(self):
        assert Levelled().capacity(0) == FLUSH_RECORDS

    def test_every_live_key_reads_back(self):
        load = Load(keys=2000, writes=8000)
        store = run_load(Levelled(), load)
        wanted = expected(load.records())
        assert all(store.get(key) is not None for key in wanted)

    def test_every_live_value_reads_back(self):
        load = Load(keys=2000, writes=8000)
        store = run_load(Levelled(), load)
        wanted = expected(load.records())
        assert all(store.get(key).value == record.value for key, record in wanted.items())

    def test_a_deleted_key_does_not_read_back(self):
        load = Load(keys=500, writes=4000, deletes=0.5)
        store = run_load(Levelled(), load)
        wanted = expected(load.records())
        gone = {record.key for record in load.records()} - set(wanted)
        assert all(store.get(key) is None for key in gone)

    def test_an_absent_key_does_not_read_back(self):
        store = run_load(Levelled(), Load(keys=2000, writes=8000))
        assert store.get(b"nothing") is None

    def test_the_written_count_exceeds_the_writes(self):
        load = Load(keys=2000, writes=8000)
        store = run_load(Levelled(), load)
        assert store.written > load.writes

    def test_the_compaction_count_grows_with_the_load(self):
        small = run_load(Levelled(), Load(keys=2000, writes=4000))
        large = run_load(Levelled(), Load(keys=2000, writes=16000))
        assert large.compactions > small.compactions

    def test_the_history_matches_the_compaction_count(self):
        store = run_load(Levelled(), Load(keys=2000, writes=8000))
        assert len(store.history) == store.compactions

    def test_as_dict_names_the_policy(self):
        assert Levelled().as_dict()["policy"] == "levelled"

    def test_as_dict_carries_the_fan_out(self):
        assert Levelled(fan_out=7).as_dict()["fan_out"] == 7


class TestTiered:
    def test_a_run_count_below_two_is_refused(self):
        with pytest.raises(ConfigError):
            Tiered(runs_per_level=1)

    def test_a_fresh_store_holds_nothing(self):
        assert Tiered().records == 0

    def test_a_level_never_holds_its_quota(self):
        store = run_load(Tiered(), Load(keys=5000, writes=20000))
        assert all(len(level) < RUNS_PER_LEVEL for level in store.levels)

    def test_a_level_may_hold_overlapping_runs(self):
        store = Tiered()
        store.flush(span(0, 100).records)
        store.flush(span(50, 150, base=100).records)
        assert store.levels[0][0].overlaps(store.levels[0][1])

    def test_every_live_key_reads_back(self):
        load = Load(keys=2000, writes=8000)
        store = run_load(Tiered(), load)
        wanted = expected(load.records())
        assert all(store.get(key) is not None for key in wanted)

    def test_every_live_value_reads_back(self):
        load = Load(keys=2000, writes=8000)
        store = run_load(Tiered(), load)
        wanted = expected(load.records())
        assert all(store.get(key).value == record.value for key, record in wanted.items())

    def test_a_deleted_key_does_not_read_back(self):
        load = Load(keys=500, writes=4000, deletes=0.5)
        store = run_load(Tiered(), load)
        wanted = expected(load.records())
        gone = {record.key for record in load.records()} - set(wanted)
        assert all(store.get(key) is None for key in gone)

    def test_a_wider_tier_writes_less(self):
        load = Load(keys=5000, writes=20000)
        narrow = run_load(Tiered(runs_per_level=2), load)
        wide = run_load(Tiered(runs_per_level=8), load)
        assert wide.written < narrow.written

    def test_a_wider_tier_compacts_less_often(self):
        load = Load(keys=5000, writes=20000)
        narrow = run_load(Tiered(runs_per_level=2), load)
        wide = run_load(Tiered(runs_per_level=8), load)
        assert wide.compactions < narrow.compactions

    def test_as_dict_names_the_policy(self):
        assert Tiered().as_dict()["policy"] == "tiered"

    def test_as_dict_carries_the_run_count(self):
        assert Tiered(runs_per_level=6).as_dict()["runs_per_level"] == 6

    def test_the_history_matches_the_compaction_count(self):
        store = run_load(Tiered(), Load(keys=2000, writes=8000))
        assert len(store.history) == store.compactions


class TestBothPoliciesAgree:
    def test_they_hold_the_same_live_set(self):
        load = Load(keys=2000, writes=8000)
        assert set(live_of(run_load(Levelled(), load))) == set(
            live_of(run_load(Tiered(), load))
        )

    def test_they_hold_the_same_live_values(self):
        load = Load(keys=2000, writes=8000)
        left, right = live_of(run_load(Levelled(), load)), live_of(run_load(Tiered(), load))
        assert all(left[key].value == right[key].value for key in left)

    def test_they_agree_with_a_dictionary(self):
        load = Load(keys=2000, writes=8000)
        wanted = expected(load.records())
        assert set(live_of(run_load(Levelled(), load))) == set(wanted)

    def test_they_agree_under_deletes(self):
        load = Load(keys=500, writes=4000, deletes=0.5)
        wanted = expected(load.records())
        assert set(live_of(run_load(Tiered(), load))) == set(wanted)

    def test_they_agree_on_a_sequential_stream(self):
        load = Load(keys=2000, writes=8000, shape="sequential")
        wanted = expected(load.records())
        assert set(live_of(run_load(Levelled(), load))) == set(wanted)

    def test_they_agree_on_a_hot_stream(self):
        load = Load(keys=2000, writes=8000, shape="hot")
        wanted = expected(load.records())
        assert set(live_of(run_load(Tiered(), load))) == set(wanted)

    def test_levelled_writes_more(self):
        load = Load(keys=5000, writes=20000)
        assert run_load(Levelled(), load).written > run_load(Tiered(), load).written

    def test_levelled_holds_fewer_records(self):
        load = Load(keys=5000, writes=20000)
        assert run_load(Levelled(), load).records < run_load(Tiered(), load).records


class TestLoad:
    def test_a_uniform_load_gives_the_write_count(self):
        assert len(Load(keys=100, writes=500).records()) == 500

    def test_a_sequential_load_walks_the_key_space(self):
        made = Load(keys=10, writes=10, shape="sequential").records()
        assert len({record.key for record in made}) == 10

    def test_a_hot_load_concentrates(self):
        made = Load(keys=10000, writes=5000, shape="hot").records()
        assert len({record.key for record in made}) < 2000

    def test_an_unknown_shape_is_refused(self):
        with pytest.raises(ConfigError):
            Load(keys=10, writes=10, shape="spiral").records()

    def test_no_deletes_means_no_tombstones(self):
        made = Load(keys=100, writes=500).records()
        assert not any(record.kind == DELETE for record in made)

    def test_every_delete_means_every_tombstone(self):
        made = Load(keys=100, writes=500, deletes=1.0).records()
        assert all(record.kind == DELETE for record in made)

    def test_half_deletes_means_about_half(self):
        made = Load(keys=100, writes=4000, deletes=0.5).records()
        found = sum(1 for record in made if record.kind == DELETE)
        assert 1800 < found < 2200

    def test_the_sequences_are_in_order(self):
        made = Load(keys=100, writes=500).records()
        assert [record.sequence for record in made] == list(range(1, 501))

    def test_the_same_seed_gives_the_same_stream(self):
        assert Load(keys=100, writes=500).records() == Load(keys=100, writes=500).records()

    def test_a_different_seed_gives_a_different_stream(self):
        left = Load(keys=1000, writes=500, seed=1).records()
        right = Load(keys=1000, writes=500, seed=2).records()
        assert left != right

    def test_as_dict_carries_the_shape(self):
        assert Load(keys=1, writes=1, shape="hot").as_dict()["shape"] == "hot"


class TestBatches:
    def test_a_short_stream_makes_one_batch(self):
        assert len(list(batches(Load(keys=50, writes=50).records()))) == 1

    def test_a_long_stream_makes_many_batches(self):
        made = Load(keys=5000, writes=5000).records()
        assert len(list(batches(made))) == 5

    def test_a_batch_is_sorted(self):
        for batch in batches(Load(keys=5000, writes=3000).records()):
            found = [record.order for record in batch]
            assert found == sorted(found)

    def test_a_batch_holds_one_record_per_key(self):
        for batch in batches(Load(keys=100, writes=3000).records()):
            found = [record.key for record in batch]
            assert len(found) == len(set(found))

    def test_a_batch_keeps_the_newest_version(self):
        made = [one(b"a", 1, b"old"), one(b"a", 2, b"new")]
        assert next(iter(batches(made)))[0].value == b"new"

    def test_the_batch_size_is_respected(self):
        made = Load(keys=5000, writes=3000).records()
        assert all(len(batch) <= 500 for batch in batches(made, size=500))


class TestCost:
    def test_amplification_is_one_when_nothing_is_rewritten(self):
        load = Load(keys=100, writes=100)
        store = Levelled()
        store.flush(sorted(load.records(), key=lambda record: record.order))
        assert amplification(store, load) == 1.0

    def test_amplification_grows_with_the_load(self):
        small = Load(keys=2000, writes=4000)
        large = Load(keys=2000, writes=16000)
        assert amplification(run_load(Levelled(), large), large) > amplification(
            run_load(Levelled(), small), small
        )

    def test_read_cost_is_zero_on_an_empty_store(self):
        assert read_cost(Levelled(), [b"a"]) == 0.0

    def test_read_cost_never_exceeds_the_run_count(self):
        store = run_load(Levelled(), Load(keys=2000, writes=8000))
        probes = [f"k{at:09d}".encode() for at in range(0, 2000, 17)]
        assert read_cost(store, probes) <= store.runs

    def test_read_cost_survives_no_keys(self):
        assert read_cost(Levelled(), []) == 0.0

    def test_stale_is_zero_on_an_empty_store(self):
        assert stale(Levelled()) == 0.0

    def test_stale_is_a_fraction(self):
        store = run_load(Levelled(), Load(keys=2000, writes=8000))
        assert 0.0 <= stale(store) < 1.0

    def test_stale_grows_with_the_overwrite_rate(self):
        few = run_load(Levelled(), Load(keys=8000, writes=8000))
        many = run_load(Levelled(), Load(keys=500, writes=8000))
        assert stale(many) > stale(few)

    def test_the_work_waste_is_a_fraction(self):
        made = Work(level=0, inputs=2, read=100, written=60, dropped=40)
        assert made.waste == 0.4

    def test_the_work_waste_survives_an_empty_read(self):
        assert Work(level=0, inputs=1, read=0, written=0, dropped=0).waste == 0.0

    def test_the_work_as_dict_carries_every_field(self):
        made = Work(level=1, inputs=2, read=3, written=4, dropped=5).as_dict()
        assert set(made) == {"level", "inputs", "read", "written", "dropped", "waste"}


class TestMeasurements:
    def test_levelled_writes_three_times_what_tiered_does(self):
        assert mod.levelled_writes_three_times_what_tiered_does()

    def test_tiered_holds_more_stale_records(self):
        assert mod.tiered_holds_more_stale_records_than_levelled()

    def test_a_sequential_stream_is_cheap(self):
        assert mod.a_sequential_write_stream_compacts_almost_for_free()

    def test_a_tiered_read_looks_in_more_runs(self):
        assert mod.a_tiered_read_looks_in_more_runs_than_a_levelled_one()

    def test_a_larger_fan_out_writes_more(self):
        assert mod.a_larger_fan_out_writes_more_and_not_less()

    def test_more_runs_per_tier_writes_less(self):
        assert mod.more_runs_per_tier_writes_less_and_reads_more()

    def test_waste_is_the_real_work(self):
        assert mod.a_compaction_reads_more_than_it_writes_only_when_there_is_overlap()

    def test_deletes_are_cheap_and_leave_the_store_stale(self):
        assert mod.deletes_make_compaction_cheaper_and_the_store_staler()

    def test_the_curve_is_fan_out_over_its_log(self):
        assert mod.the_write_cost_of_a_level_is_the_fan_out_and_the_count_of_them_is_the_log()

    def test_the_range_rules_runs_out(self):
        assert mod.a_run_that_does_not_overlap_is_ruled_out_by_two_comparisons()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_ten_claims(self):
        assert len(mod.summarise()) == 10

    def test_the_policy_table_has_two_rows(self):
        assert len(mod.compare_the_policies(2000, 6000)) == 2

    def test_the_policy_table_shows_levelled_writing_more(self):
        rows = mod.compare_the_policies(2000, 6000)
        assert rows[0]["amplification"] > rows[1]["amplification"]

    def test_the_policy_table_shows_tiered_reading_more(self):
        rows = mod.compare_the_policies(2000, 6000)
        assert rows[1]["read_cost"] >= rows[0]["read_cost"]

    def test_the_fan_out_table_has_five_rows(self):
        assert len(mod.compare_the_fan_outs(2000, 6000)) == 5

    def test_the_fan_out_table_shrinks_the_levels(self):
        rows = mod.compare_the_fan_outs(2000, 6000)
        found = [row["levels"] for row in rows]
        assert found == sorted(found, reverse=True)

    def test_the_fan_out_table_ends_higher_than_it_starts(self):
        rows = mod.compare_the_fan_outs(20000, 40000)
        assert rows[-1]["amplification"] > rows[0]["amplification"]

    def test_the_tier_width_table_has_five_rows(self):
        assert len(mod.compare_the_tier_widths(2000, 6000)) == 5

    def test_a_wider_tier_writes_less(self):
        rows = mod.compare_the_tier_widths(20000, 40000)
        assert rows[-1]["amplification"] < rows[0]["amplification"]

    def test_the_shape_table_has_three_rows(self):
        assert len(mod.compare_the_shapes(2000, 6000)) == 3

    def test_a_sequential_shape_costs_least(self):
        rows = mod.compare_the_shapes(20000, 40000)
        assert rows[0]["amplification"] < rows[-1]["amplification"]

    def test_the_delete_table_has_four_rows(self):
        assert len(mod.compare_the_delete_rates(1000, 4000)) == 4

    def test_more_deletes_writes_less(self):
        rows = mod.compare_the_delete_rates(5000, 20000)
        assert rows[-1]["written"] < rows[0]["written"]

    def test_more_deletes_leaves_more_stale(self):
        rows = mod.compare_the_delete_rates(5000, 20000)
        assert rows[-1]["stale"] > rows[0]["stale"]

    def test_the_cached_load_is_shared(self):
        assert mod._load(100, 100) is mod._load(100, 100)

    def test_the_cached_levelled_store_is_shared(self):
        assert mod._levelled(1000, 2000) is mod._levelled(1000, 2000)

    def test_the_cached_tiered_store_is_shared(self):
        assert mod._tiered(1000, 2000) is mod._tiered(1000, 2000)

    def test_the_default_fan_out_is_ten(self):
        assert FAN_OUT == 10
