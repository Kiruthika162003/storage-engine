from __future__ import annotations

import pytest

from store.errorbudget import (
    OBJECTIVE,
    RATE,
    WINDOW,
    Month,
    Policy,
    background_noise_spends_a_fifth_of_the_budget,
    fast,
    slow,
    summarise,
    the_fast_window_pages_in_minutes,
    the_fast_window_sleeps_through_a_leak,
    the_slow_window_catches_the_leak,
    the_slow_window_pages_long_after_recovery,
)


class TestMonth:
    def test_quiet_is_deterministic(self):
        assert Month.quiet(3).failed == Month.quiet(3).failed

    def test_quiet_has_a_full_window_of_ticks(self):
        assert len(Month.quiet(3).failed) == WINDOW

    def test_the_budget_matches_the_objective(self):
        assert Month.quiet(3).budget() == int(RATE * WINDOW * (1 - OBJECTIVE))

    def test_an_outage_overwrites_its_ticks(self):
        month = Month.quiet(3).with_outage(100, 10, 0.5)
        assert month.failed[100] == RATE // 2
        assert month.failed[110] == Month.quiet(3).failed[110]

    def test_an_outage_does_not_mutate_the_source(self):
        month = Month.quiet(3)
        month.with_outage(0, 10, 1.0)
        assert month.failed[0] == Month.quiet(3).failed[0]

    def test_an_outage_clips_at_the_window_end(self):
        month = Month.quiet(3).with_outage(WINDOW - 5, 100, 1.0)
        assert len(month.failed) == WINDOW


class TestPolicy:
    def test_a_perfect_month_never_pages(self):
        month = Month(failed=[0] * 1000)
        assert Policy(window=60, multiplier=1.0).replay(month) == []

    def test_a_total_outage_pages_immediately(self):
        month = Month(failed=[RATE] * 100)
        pages = Policy(window=60, multiplier=14.4).replay(month)
        assert pages[0] == 0

    def test_the_rolling_window_forgets(self):
        failed = [RATE] * 60 + [0] * 200
        pages = Policy(window=60, multiplier=14.4).replay(Month(failed=failed))
        assert pages[-1] < 130

    def test_replay_resets_between_calls(self):
        policy = fast()
        policy.replay(Month(failed=[RATE] * 100))
        assert policy.replay(Month(failed=[0] * 100)) == []

    def test_the_stock_policies_have_their_shapes(self):
        assert fast().window < slow().window
        assert fast().multiplier > slow().multiplier


class TestClaims:
    @pytest.mark.parametrize(
        "claim",
        [
            the_fast_window_pages_in_minutes,
            the_fast_window_sleeps_through_a_leak,
            the_slow_window_catches_the_leak,
            the_slow_window_pages_long_after_recovery,
            background_noise_spends_a_fifth_of_the_budget,
        ],
    )
    def test_claim_holds(self, claim):
        assert claim() is True

    def test_summary_is_all_true(self):
        told = summarise()
        assert all(value for name, value in told.items() if name != "module")
