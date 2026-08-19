from __future__ import annotations

import pytest

from store import memtable as tables
from store.errors import ConfigError
from store.memtable import (
    FLUSH_BYTES,
    MAX_LEVEL,
    PROMOTION,
    Memtable,
    Node,
    SortedList,
)
from store.record import Record


def test_a_memtable_keeps_its_order():
    assert tables.a_memtable_holds_its_writes_in_order()["keys_are_sorted"]


def test_a_memtable_holds_every_write():
    assert tables.a_memtable_holds_its_writes_in_order()["they_are_all_there"]


def test_a_read_finds_a_written_key():
    assert tables.a_memtable_holds_its_writes_in_order()["a_read_finds_one"]


def test_a_read_misses_an_absent_key():
    assert tables.a_memtable_holds_its_writes_in_order()["and_misses_what_is_absent"]


def test_overwriting_leaves_one_entry():
    assert tables.overwriting_one_key_fills_the_table_as_fast_as_writing_many()[
        "the_narrow_one_holds_one"
    ]


def test_overwriting_keeps_the_table_small():
    assert tables.overwriting_one_key_fills_the_table_as_fast_as_writing_many()[
        "and_it_is_far_smaller"
    ]


def test_every_write_but_the_first_was_an_overwrite():
    assert tables.overwriting_one_key_fills_the_table_as_fast_as_writing_many()[
        "which_is_every_write_but_the_first"
    ]


def test_the_surprise_moves_to_recovery():
    assert tables.overwriting_one_key_fills_the_table_as_fast_as_writing_many()[
        "so_the_surprise_moves_to_recovery"
    ]


def test_the_sorted_list_wins_every_comparison_count():
    assert tables.the_skiplist_loses_on_comparisons_at_every_size()["the_list_wins_everywhere"]


def test_the_comparison_gap_does_not_shrink():
    assert tables.the_skiplist_loses_on_comparisons_at_every_size()["and_it_does_not_shrink"]


def test_a_comparison_benchmark_picks_the_list():
    assert tables.the_skiplist_loses_on_comparisons_at_every_size()[
        "so_a_comparison_benchmark_picks_the_list"
    ]


def test_the_moves_are_quadratic():
    assert tables.the_skiplist_wins_on_the_thing_nobody_counts()["so_the_moves_are_quadratic"]


def test_the_growth_is_about_a_hundredfold():
    assert tables.the_skiplist_wins_on_the_thing_nobody_counts()["and_it_is_about_a_hundred"]


def test_the_moves_dwarf_the_comparisons():
    made = tables.the_skiplist_wins_on_the_thing_nobody_counts()
    assert made["which_is_this_many_times_more"] > 10


def test_ordered_keys_move_nothing():
    assert tables.keys_that_arrive_in_order_cost_the_sorted_list_nothing()["it_moved_nothing"]


def test_shuffled_keys_move_millions():
    assert tables.keys_that_arrive_in_order_cost_the_sorted_list_nothing()[
        "and_shuffled_moved_millions"
    ]


def test_the_list_wins_the_ordered_workload_outright():
    assert tables.keys_that_arrive_in_order_cost_the_sorted_list_nothing()[
        "so_the_list_wins_this_workload_outright"
    ]


def test_the_promotion_curve_is_not_flat():
    assert tables.the_promotion_chance_trades_height_against_comparisons()[
        "so_the_curve_is_not_flat"
    ]


def test_both_ends_of_the_promotion_range_are_worse():
    assert tables.the_promotion_chance_trades_height_against_comparisons()[
        "and_both_ends_are_worse"
    ]


def test_the_best_promotion_is_the_shipped_one():
    assert tables.the_promotion_chance_trades_height_against_comparisons()[
        "and_the_best_is_the_shipped_one"
    ]


def test_the_height_grows_with_the_promotion_chance():
    assert tables.the_promotion_chance_trades_height_against_comparisons()[
        "the_height_grows_with_the_chance"
    ]


def test_the_same_seed_builds_the_same_shape():
    assert tables.a_deterministic_skiplist_has_the_same_shape_every_time()["they_are_identical"]


def test_a_different_seed_builds_another():
    assert tables.a_deterministic_skiplist_has_the_same_shape_every_time()[
        "and_a_different_seed_differs"
    ]


def test_the_contents_do_not_depend_on_the_seed():
    assert tables.a_deterministic_skiplist_has_the_same_shape_every_time()[
        "but_the_contents_are_the_same"
    ]


def test_the_byte_counts_are_close_across_value_sizes():
    assert tables.a_flush_threshold_in_bytes_is_the_only_one_that_means_anything()[
        "and_the_byte_counts_are_close"
    ]


def test_an_entry_threshold_would_split_them():
    assert tables.a_flush_threshold_in_bytes_is_the_only_one_that_means_anything()[
        "so_an_entry_threshold_would_split_them"
    ]


def test_an_impossible_promotion_chance_is_refused():
    assert tables.an_impossible_promotion_chance_is_refused()


def test_a_table_with_no_levels_is_refused():
    assert tables.a_table_with_no_levels_is_refused()


def test_a_scan_is_half_open():
    assert tables.a_scan_stops_where_it_is_told()["which_is_the_half_open_range"]


def test_an_open_ended_scan_reaches_the_end():
    assert tables.a_scan_stops_where_it_is_told()["an_open_ended_scan_runs_to_the_end"]


def test_the_structure_table_covers_four():
    assert len(tables.compare_the_structures()) == 4


def test_comparisons_pick_the_list_in_both_orders():
    assert tables.the_answer_depends_on_which_operation_you_count()[
        "comparisons_pick_the_list_twice"
    ]


def test_total_work_splits_the_two_orders():
    assert tables.the_answer_depends_on_which_operation_you_count()[
        "and_total_work_splits_them"
    ]


def test_the_measure_decides_the_shuffled_row():
    assert tables.the_answer_depends_on_which_operation_you_count()[
        "the_shuffled_winner_changes"
    ]


def test_the_summary_says_the_shape_repeats():
    assert tables.summarise()["the_shape_repeats"]


def test_the_summary_reports_the_promotion():
    assert tables.summarise()["promotion"] == PROMOTION


def test_a_new_memtable_is_empty():
    assert Memtable().entries == 0


def test_putting_a_record_adds_an_entry():
    made = Memtable(seed=1)
    made.put(Record(key=b"k", sequence=1, value=b"v"))
    assert made.entries == 1


def test_putting_the_same_key_twice_adds_one():
    made = Memtable(seed=1)
    made.put(Record(key=b"k", sequence=1, value=b"v"))
    made.put(Record(key=b"k", sequence=2, value=b"w"))
    assert made.entries == 1


def test_an_overwrite_keeps_the_newer_record():
    made = Memtable(seed=1)
    made.put(Record(key=b"k", sequence=1, value=b"old"))
    made.put(Record(key=b"k", sequence=2, value=b"new"))
    assert made.get(b"k").value == b"new"


def test_an_overwrite_is_counted():
    made = Memtable(seed=1)
    made.put(Record(key=b"k", sequence=1, value=b"v"))
    made.put(Record(key=b"k", sequence=2, value=b"w"))
    assert made.overwrites == 1


def test_an_overwrite_adjusts_the_byte_count():
    made = Memtable(seed=1)
    made.put(Record(key=b"k", sequence=1, value=b"x" * 100))
    before = made.nbytes
    made.put(Record(key=b"k", sequence=2, value=b"x"))
    assert made.nbytes < before


def test_a_missing_key_reads_as_nothing():
    assert Memtable(seed=1).get(b"absent") is None


def test_a_tombstone_is_stored_like_anything_else():
    made = Memtable(seed=1)
    made.put(Record(key=b"k", sequence=1, value=b"v"))
    made.put(Record(key=b"k", sequence=2, kind=1))
    assert made.get(b"k").tombstone


def test_records_come_back_sorted():
    made = Memtable(seed=1)
    for one in (b"c", b"a", b"b"):
        made.put(Record(key=one, sequence=1, value=b"v"))
    assert [one.key for one in made.records()] == [b"a", b"b", b"c"]


def test_a_scan_from_a_key_skips_what_is_below():
    made = Memtable(seed=1)
    for one in (b"a", b"b", b"c"):
        made.put(Record(key=one, sequence=1, value=b"v"))
    assert [one.key for one in made.scan(b"b")] == [b"b", b"c"]


def test_a_scan_with_a_stop_excludes_it():
    made = Memtable(seed=1)
    for one in (b"a", b"b", b"c"):
        made.put(Record(key=one, sequence=1, value=b"v"))
    assert [one.key for one in made.scan(b"a", b"c")] == [b"a", b"b"]


def test_a_scan_of_an_empty_table_is_empty():
    assert list(Memtable(seed=1).scan()) == []


def test_a_table_reports_when_it_is_full():
    made = Memtable(seed=1)
    made.nbytes = FLUSH_BYTES
    assert made.full


def test_a_small_table_is_not_full():
    assert not Memtable(seed=1).full


def test_a_table_summarises():
    assert Memtable(seed=1).as_dict()["entries"] == 0


def test_the_height_starts_at_one():
    assert Memtable(seed=1).height == 1


def test_the_height_grows_with_the_entries():
    made = Memtable(seed=1)
    for one in range(500):
        made.put(Record(key=f"k{one:04d}".encode(), sequence=one, value=b"v"))
    assert made.height > 1


def test_an_impossible_promotion_raises():
    with pytest.raises(ConfigError):
        Memtable(promotion=0.0)


def test_a_promotion_of_one_raises():
    with pytest.raises(ConfigError):
        Memtable(promotion=1.0)


def test_zero_levels_raises():
    with pytest.raises(ConfigError):
        Memtable(levels=0)


def test_a_node_reports_its_key():
    made = Node(record=Record(key=b"k", sequence=1), forward=[None])
    assert made.key == b"k"


def test_the_head_node_has_an_empty_key():
    assert Node(record=None, forward=[None]).key == b""


def test_a_node_reports_its_level():
    assert Node(record=None, forward=[None, None]).level == 2


def test_a_sorted_list_holds_its_writes():
    made = SortedList()
    made.put(Record(key=b"b", sequence=1, value=b"v"))
    made.put(Record(key=b"a", sequence=2, value=b"w"))
    assert [one.key for one in made.records()] == [b"a", b"b"]


def test_a_sorted_list_replaces_a_key():
    made = SortedList()
    made.put(Record(key=b"a", sequence=1, value=b"old"))
    made.put(Record(key=b"a", sequence=2, value=b"new"))
    assert made.get(b"a").value == b"new" and len(made.keys) == 1


def test_a_sorted_list_misses_an_absent_key():
    assert SortedList().get(b"a") is None


def test_a_sorted_list_counts_its_moves():
    made = SortedList()
    made.put(Record(key=b"b", sequence=1, value=b"v"))
    made.put(Record(key=b"a", sequence=2, value=b"w"))
    assert made.moves == 1


def test_an_appended_key_moves_nothing():
    made = SortedList()
    made.put(Record(key=b"a", sequence=1, value=b"v"))
    made.put(Record(key=b"b", sequence=2, value=b"w"))
    assert made.moves == 0


def test_a_sorted_list_summarises():
    assert SortedList().as_dict()["entries"] == 0


def test_the_level_cap_is_reachable():
    assert MAX_LEVEL >= 8


def test_the_flush_threshold_is_a_real_size():
    assert FLUSH_BYTES >= 1 << 16
