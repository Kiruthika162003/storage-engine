from __future__ import annotations

import pytest

from store import disk as files
from store.disk import SECTOR, Disk
from store.errors import ConfigError


def test_a_write_is_all_at_risk():
    assert files.a_write_reaches_the_buffer_and_a_sync_reaches_the_medium()["it_is_all_at_risk"]


def test_a_sync_makes_it_durable():
    assert files.a_write_reaches_the_buffer_and_a_sync_reaches_the_medium()[
        "and_it_is_all_durable"
    ]


def test_the_size_does_not_change_on_a_sync():
    assert files.a_write_reaches_the_buffer_and_a_sync_reaches_the_medium()[
        "the_size_did_not_change"
    ]


def test_a_reader_sees_the_buffered_bytes():
    assert files.a_reader_cannot_tell_the_difference_and_a_crash_can()["it_was_there"]


def test_a_crash_takes_them_away():
    assert files.a_reader_cannot_tell_the_difference_and_a_crash_can()["and_it_is_gone"]


def test_the_durable_view_agreed_all_along():
    assert files.a_reader_cannot_tell_the_difference_and_a_crash_can()[
        "the_durable_view_agreed_all_along"
    ]


def test_only_a_crash_separates_the_two_states():
    assert files.a_reader_cannot_tell_the_difference_and_a_crash_can()[
        "so_only_a_crash_separates_them"
    ]


def test_two_sync_policies_move_the_same_bytes():
    assert files.syncing_costs_a_call_and_not_a_byte()["the_same_bytes"]


def test_the_eager_policy_makes_ten_times_the_calls():
    assert files.syncing_costs_a_call_and_not_a_byte()["and_ten_times_the_calls"]


def test_the_two_policies_leave_the_same_content():
    assert files.syncing_costs_a_call_and_not_a_byte()["the_same_durable_content"]


def test_truncation_shrinks_the_file():
    assert files.truncating_removes_durable_bytes_too()["it_shrank"]


def test_truncation_keeps_the_good_part():
    assert files.truncating_removes_durable_bytes_too()["and_the_good_part_survived"]


def test_truncation_leaves_nothing_pending():
    assert files.truncating_removes_durable_bytes_too()["and_nothing_is_pending"]


def test_a_truncation_past_the_end_is_refused():
    assert files.a_truncation_past_the_end_is_refused()


def test_a_negative_truncation_is_refused():
    assert files.a_negative_truncation_is_refused()


def test_a_file_without_a_name_is_refused():
    assert files.a_file_without_a_name_is_refused()


def test_the_summary_says_a_write_is_not_durable():
    assert files.summarise()["a_write_is_not_durable"]


def test_the_summary_reports_the_sector_size():
    assert files.summarise()["sector"] == SECTOR


def test_a_new_file_is_empty():
    assert Disk(name="x").size == 0


def test_appending_grows_the_file():
    made = Disk(name="x")
    made.append(b"abc")
    assert made.size == 3


def test_appending_returns_the_length():
    assert Disk(name="x").append(b"abcd") == 4


def test_appending_counts_the_write():
    made = Disk(name="x")
    made.append(b"a")
    made.append(b"b")
    assert made.writes == 2


def test_syncing_returns_what_moved():
    made = Disk(name="x")
    made.append(b"abcde")
    assert made.sync() == 5


def test_syncing_an_empty_buffer_moves_nothing():
    assert Disk(name="x").sync() == 0


def test_syncing_counts_the_call():
    made = Disk(name="x")
    made.sync()
    made.sync()
    assert made.syncs == 2


def test_crashing_returns_what_was_lost():
    made = Disk(name="x")
    made.append(b"lost")
    assert made.crash() == 4


def test_crashing_keeps_the_durable_part():
    made = Disk(name="x")
    made.append(b"kept")
    made.sync()
    made.append(b"lost")
    made.crash()
    assert made.read() == b"kept"


def test_crashing_twice_loses_nothing_more():
    made = Disk(name="x")
    made.append(b"lost")
    made.crash()
    assert made.crash() == 0


def test_reading_includes_the_buffer():
    made = Disk(name="x")
    made.append(b"buffered")
    assert made.read() == b"buffered"


def test_reading_durable_does_not():
    made = Disk(name="x")
    made.append(b"buffered")
    assert made.read_durable() == b""


def test_the_at_risk_count_is_the_buffer():
    made = Disk(name="x")
    made.append(b"abc")
    assert made.at_risk == 3


def test_truncating_to_zero_empties_the_file():
    made = Disk(name="x")
    made.append(b"abc")
    made.sync()
    made.truncate(0)
    assert made.size == 0


def test_truncating_to_the_size_changes_nothing():
    made = Disk(name="x")
    made.append(b"abc")
    made.sync()
    made.truncate(3)
    assert made.read() == b"abc"


def test_truncating_past_the_end_raises():
    with pytest.raises(ConfigError):
        Disk(name="x").truncate(1)


def test_truncating_below_zero_raises():
    with pytest.raises(ConfigError):
        Disk(name="x").truncate(-1)


def test_an_unnamed_file_raises():
    with pytest.raises(ConfigError):
        Disk(name="")


def test_a_file_summarises():
    assert Disk(name="named").as_dict()["file"] == "named"


def test_the_byte_counters_add_up():
    made = Disk(name="x")
    made.append(b"abc")
    made.append(b"de")
    made.sync()
    assert made.bytes_written == made.bytes_synced == 5


def test_the_sector_is_a_power_of_two():
    assert SECTOR & (SECTOR - 1) == 0
