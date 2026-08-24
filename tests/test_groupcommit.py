from __future__ import annotations

import pytest

from store.groupcommit import (
    TIMER,
    Log,
    _arrivals,
    a_busy_group_fills_before_anyone_waits,
    a_quiet_log_pays_the_timer_not_the_group,
    percentile,
    raising_the_group_on_a_quiet_log_changes_nothing,
    run,
    summarise,
    the_discount_scales_with_the_crowd,
)


class TestLog:
    def test_a_full_group_syncs_at_once(self):
        log = Log(group_size=2)
        log.append(5)
        log.append(5)
        assert log.syncs == 1 and log.waits == [0, 0]

    def test_a_lone_write_waits_for_the_timer(self):
        log = Log(group_size=8)
        log.append(0)
        for now in range(1, TIMER + 1):
            log.tick(now)
        assert log.syncs == 1 and log.waits == [TIMER]

    def test_the_timer_starts_with_the_first_pending_write(self):
        log = Log(group_size=8)
        log.append(0)
        log.append(5)
        for now in range(1, TIMER + 1):
            log.tick(now)
        assert log.waits == [TIMER, TIMER - 5]

    def test_drain_flushes_the_stragglers(self):
        log = Log(group_size=8)
        log.append(3)
        log.drain(4)
        assert log.syncs == 1 and log.waits == [1]

    def test_an_empty_drain_does_nothing(self):
        log = Log(group_size=8)
        log.drain(10)
        assert log.syncs == 0

    def test_group_size_one_never_waits(self):
        log = run(1, 0.5)
        assert set(log.waits) == {0}


class TestHarness:
    def test_arrivals_are_deterministic(self):
        assert _arrivals(17, 0.5) == _arrivals(17, 0.5)

    def test_every_write_is_acknowledged(self):
        log = run(8, 0.5)
        assert len(log.waits) == len(_arrivals(17, 0.5))

    def test_percentile_picks_from_the_sorted_order(self):
        assert percentile([5, 1, 9], 0.0) == 1
        assert percentile([5, 1, 9], 0.5) == 5
        assert percentile([5, 1, 9], 0.99) == 9

    def test_no_wait_exceeds_the_timer(self):
        for group in (2, 8, 32):
            assert max(run(group, 0.3).waits) <= TIMER


class TestClaims:
    @pytest.mark.parametrize(
        "claim",
        [
            a_busy_group_fills_before_anyone_waits,
            the_discount_scales_with_the_crowd,
            a_quiet_log_pays_the_timer_not_the_group,
            raising_the_group_on_a_quiet_log_changes_nothing,
        ],
    )
    def test_claim_holds(self, claim):
        assert claim() is True

    def test_summary_is_all_true(self):
        told = summarise()
        assert all(value for name, value in told.items() if name != "module")
