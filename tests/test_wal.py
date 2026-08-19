from __future__ import annotations

import zlib

import pytest

from store import wal as logging
from store.disk import Disk
from store.errors import BadChecksum, ConfigError, TornWrite
from store.record import Record
from store.wal import (
    EVERY_BATCH,
    EVERY_RECORD,
    FRAME,
    NEVER,
    POLICIES,
    Log,
    Recovery,
    frame,
    recover,
    unframe,
    weak_frame,
)


def test_a_log_replays_everything():
    assert logging.a_log_replays_every_record_it_acknowledged()["they_all_came_back"]


def test_a_replay_keeps_the_order():
    assert logging.a_log_replays_every_record_it_acknowledged()["the_order_is_kept"]


def test_a_clean_replay_loses_nothing():
    assert logging.a_log_replays_every_record_it_acknowledged()["and_nothing_was_lost"]


def test_a_torn_tail_loses_one_record():
    assert logging.a_torn_tail_stops_recovery_and_keeps_everything_before_it()["it_lost_one"]


def test_a_torn_tail_keeps_the_rest():
    assert logging.a_torn_tail_stops_recovery_and_keeps_everything_before_it()[
        "and_kept_the_rest"
    ]


def test_a_torn_tail_is_reported_as_torn():
    assert logging.a_torn_tail_stops_recovery_and_keeps_everything_before_it()[
        "and_it_is_a_torn_write"
    ]


def test_a_bad_checksum_stops_recovery_early():
    assert logging.a_flipped_bit_in_the_middle_stops_recovery_there()["it_stopped_early"]


def test_a_bad_checksum_keeps_what_came_before():
    assert logging.a_flipped_bit_in_the_middle_stops_recovery_there()[
        "and_kept_what_came_before"
    ]


def test_a_bad_checksum_is_reported_as_one():
    assert logging.a_flipped_bit_in_the_middle_stops_recovery_there()[
        "and_it_is_a_bad_checksum"
    ]


def test_the_bytes_after_a_stop_are_not_read():
    assert logging.a_flipped_bit_in_the_middle_stops_recovery_there()["which_are_not_read"]


def test_the_strong_frame_catches_a_corrupt_length():
    assert logging.a_checksum_that_skips_the_length_believes_a_corrupt_one()[
        "the_strong_frame_caught_it"
    ]


def test_the_corrupt_length_is_larger_than_the_frame():
    assert logging.a_checksum_that_skips_the_length_believes_a_corrupt_one()[
        "which_is_more_than_there_are"
    ]


def test_never_syncing_loses_everything():
    assert logging.the_sync_policy_decides_what_a_crash_costs()["never_loses_everything"]


def test_syncing_per_record_loses_nothing():
    assert logging.the_sync_policy_decides_what_a_crash_costs()["per_record_loses_nothing"]


def test_syncing_per_batch_costs_a_tenth():
    assert logging.the_sync_policy_decides_what_a_crash_costs()[
        "the_batch_policy_costs_a_tenth"
    ]


def test_a_crash_after_the_sync_keeps_the_batch():
    assert logging.a_crash_between_the_write_and_the_sync_loses_the_batch()[
        "which_is_all_of_them"
    ]


def test_a_crash_before_it_loses_the_batch():
    assert logging.a_crash_between_the_write_and_the_sync_loses_the_batch()["which_is_none"]


def test_the_window_is_a_batch_and_not_a_record():
    assert logging.a_crash_between_the_write_and_the_sync_loses_the_batch()[
        "the_window_is_a_batch_not_a_record"
    ]


def test_an_empty_batch_is_refused():
    assert logging.an_empty_batch_is_refused()


def test_an_unknown_policy_is_refused():
    assert logging.an_unknown_sync_policy_is_refused()


def test_an_empty_log_recovers_cleanly():
    assert logging.an_empty_log_recovers_to_nothing()["and_it_is_not_damage"]


def test_an_empty_log_has_no_records():
    assert logging.an_empty_log_recovers_to_nothing()["it_is_empty"]


def test_the_policy_table_covers_three():
    assert len(logging.compare_the_policies()) == 3


def test_every_policy_writes_the_same_bytes():
    assert logging.every_policy_writes_the_same_bytes_and_survives_differently()[
        "they_wrote_the_same_bytes"
    ]


def test_the_policies_differ_in_syncs():
    assert logging.every_policy_writes_the_same_bytes_and_survives_differently()[
        "and_the_syncs_differ"
    ]


def test_the_policies_differ_in_what_they_lose():
    assert logging.every_policy_writes_the_same_bytes_and_survives_differently()[
        "and_so_does_what_they_lose"
    ]


def test_the_summary_says_the_checksum_covers_the_length():
    assert logging.summarise()["the_checksum_covers_the_length"]


def test_the_summary_lists_the_policies():
    assert logging.summarise()["policies"] == list(POLICIES)


def test_a_frame_round_trips():
    assert unframe(frame(b"payload"))[0] == b"payload"


def test_a_frame_reports_where_it_ended():
    made = frame(b"payload")
    assert unframe(made)[1] == len(made)


def test_a_frame_of_nothing_round_trips():
    assert unframe(frame(b""))[0] == b""


def test_a_frame_costs_a_header_and_a_length():
    assert len(frame(b"abc")) == FRAME.size + 3


def test_a_frame_checksum_covers_the_length():
    made = bytearray(frame(b"abc"))
    made[4] ^= 0x01
    with pytest.raises((BadChecksum, TornWrite)):
        unframe(bytes(made))


def test_a_frame_checksum_covers_the_payload():
    made = bytearray(frame(b"abc"))
    made[-1] ^= 0x01
    with pytest.raises(BadChecksum):
        unframe(bytes(made))


def test_a_short_frame_header_is_torn():
    with pytest.raises(TornWrite):
        unframe(b"\x00\x00")


def test_a_short_frame_payload_is_torn():
    made = frame(b"payload")
    with pytest.raises(TornWrite):
        unframe(made[:-2])


def test_the_weak_frame_leaves_its_length_unprotected():
    made = weak_frame(b"abc")
    checksum = int.from_bytes(made[:4], "little")
    assert checksum == zlib.crc32(b"abc") & 0xFFFFFFFF


def test_two_frames_read_back_in_order():
    raw = frame(b"one") + frame(b"two")
    first, at = unframe(raw)
    second, end = unframe(raw, at)
    assert (first, second, end) == (b"one", b"two", len(raw))


def test_a_log_appends_a_batch():
    made = Log(disk=Disk(name="x"))
    assert made.append([Record(key=b"k", sequence=1)]) > 0


def test_a_log_counts_its_records():
    made = Log(disk=Disk(name="x"))
    made.append([Record(key=b"a", sequence=1), Record(key=b"b", sequence=2)])
    assert made.appended == 2


def test_a_log_counts_its_batches():
    made = Log(disk=Disk(name="x"))
    made.append([Record(key=b"a", sequence=1)])
    made.append([Record(key=b"b", sequence=2)])
    assert made.batches == 2


def test_the_never_policy_syncs_nothing():
    made = Log(disk=Disk(name="x"), policy=NEVER)
    made.append([Record(key=b"k", sequence=1)])
    assert made.disk.syncs == 0


def test_the_batch_policy_syncs_once_per_batch():
    made = Log(disk=Disk(name="x"), policy=EVERY_BATCH)
    made.append([Record(key=b"a", sequence=1), Record(key=b"b", sequence=2)])
    assert made.disk.syncs == 1


def test_the_record_policy_syncs_once_per_record():
    made = Log(disk=Disk(name="x"), policy=EVERY_RECORD)
    made.append([Record(key=b"a", sequence=1), Record(key=b"b", sequence=2)])
    assert made.disk.syncs == 2


def test_a_log_reports_what_is_at_risk():
    made = Log(disk=Disk(name="x"), policy=NEVER)
    made.append([Record(key=b"k", sequence=1)])
    assert made.at_risk > 0


def test_a_synced_log_risks_nothing():
    made = Log(disk=Disk(name="x"), policy=EVERY_BATCH)
    made.append([Record(key=b"k", sequence=1)])
    assert made.at_risk == 0


def test_a_log_summarises():
    assert Log(disk=Disk(name="x")).as_dict()["policy"] == EVERY_BATCH


def test_an_empty_batch_raises():
    with pytest.raises(ConfigError):
        Log(disk=Disk(name="x")).append([])


def test_an_unknown_policy_raises():
    with pytest.raises(ConfigError):
        Log(disk=Disk(name="x"), policy="hopefully")


def test_recovery_returns_the_records():
    made = Log(disk=Disk(name="x"))
    written = [Record(key=b"a", sequence=1, value=b"v")]
    made.append(written)
    assert recover(made.disk.read()).records == written


def test_a_clean_recovery_is_truthy():
    assert recover(b"")


def test_a_damaged_recovery_is_falsy():
    assert not recover(b"\x00" * 20)


def test_a_recovery_reports_what_was_lost():
    made = Recovery(bytes_read=100, stopped_at=40, reason="TornWrite")
    assert made.lost == 60


def test_a_clean_recovery_lost_nothing():
    made = Recovery(bytes_read=40, stopped_at=40)
    assert made.lost == 0


def test_a_recovery_summarises():
    assert recover(b"").as_dict()["complete"]


def test_an_empty_recovery_says_end_of_log():
    assert recover(b"").as_dict()["reason"] == "end of log"


def test_the_frame_header_is_eight_bytes():
    assert FRAME.size == 8


def test_there_are_three_policies():
    assert len(POLICIES) == 3
