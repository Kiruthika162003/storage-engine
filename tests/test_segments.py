from __future__ import annotations

import pytest

from store import segments as mod
from store.errors import ConfigError
from store.segments import Segments


def grown(count: int, segment_bytes: int = 4096) -> Segments:
    log = Segments(segment_bytes=segment_bytes)
    for sequence in range(1, count + 1):
        log.append(mod._record(sequence))
    return log


class TestChain:
    def test_a_tiny_segment_size_is_refused(self):
        with pytest.raises(ConfigError):
            Segments(segment_bytes=10)

    def test_a_fresh_log_has_one_open_segment(self):
        assert Segments().segments == 1

    def test_records_fill_the_tail(self):
        log = grown(10)
        assert log.segments == 1 and log.bytes_held > 0

    def test_the_boundary_opens_a_new_segment(self):
        log = grown(200)
        assert log.segments > 1

    def test_the_high_mark_tracks_per_segment(self):
        log = grown(300)
        assert log.highest_in == sorted(log.highest_in)

    def test_replay_holds_everything_in_order(self):
        log = grown(250)
        assert [record.sequence for record in log.replay()] == list(range(1, 251))


class TestTruncate:
    def test_a_zero_mark_frees_nothing(self):
        log = grown(300)
        assert log.truncate(0) == 0

    def test_a_full_mark_frees_all_but_the_tail(self):
        log = grown(300)
        log.truncate(10**9)
        assert log.segments == 1

    def test_a_mid_mark_frees_the_prefix(self):
        log = grown(1000)
        before = log.segments
        freed = log.truncate(500)
        assert 0 < freed < before

    def test_freed_records_are_gone_from_replay(self):
        log = grown(1000)
        log.truncate(500)
        assert all(record.sequence > 400 for record in log.replay())

    def test_unfreed_records_remain(self):
        log = grown(1000)
        log.truncate(500)
        assert any(record.sequence == 1000 for record in log.replay())

    def test_truncation_is_idempotent(self):
        log = grown(1000)
        log.truncate(500)
        assert log.truncate(500) == 0

    def test_deletions_are_counted(self):
        log = grown(1000)
        freed = log.truncate(800)
        assert log.deleted == freed


class TestMeasurements:
    def test_whole_segments_only(self):
        assert mod.truncation_frees_whole_segments_and_only_those()

    def test_nothing_flushed_nothing_freed(self):
        assert mod.nothing_flushed_nothing_freed()

    def test_the_tail_stays(self):
        assert mod.the_tail_segment_never_goes()

    def test_the_delay_is_one_segment(self):
        assert mod.the_space_delay_is_one_segments_writes()

    def test_replay_is_whole(self):
        assert mod.replay_equals_the_unsegmented_log()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5
