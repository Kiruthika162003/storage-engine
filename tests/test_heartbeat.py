from __future__ import annotations

import pytest

from store.heartbeat import (
    INTERVAL,
    Adaptive,
    Fixed,
    _jittery_gaps,
    _shifted_gaps,
    _watch,
    patience_stretches_with_the_observed_spread,
    summarise,
    the_fixed_dial_trades_alarms_for_delay,
    the_learned_detector_survives_the_regime_change,
    the_learned_patience_is_a_number_it_picked,
)


class TestFixed:
    def test_a_recent_beat_clears_suspicion(self):
        detector = Fixed(timeout=10)
        detector.beat(100)
        assert not detector.suspects(105)

    def test_silence_past_the_timeout_is_suspect(self):
        detector = Fixed(timeout=10)
        detector.beat(100)
        assert detector.suspects(111)


class TestAdaptive:
    def test_the_cold_start_patience_is_generous(self):
        assert Adaptive().patience() == INTERVAL * 4.0

    def test_beats_record_their_gaps(self):
        detector = Adaptive()
        for now in (10, 20, 30):
            detector.beat(now)
        assert detector.gaps == [10, 10, 10]

    def test_the_window_is_bounded(self):
        detector = Adaptive()
        for now in range(10, 2000, 10):
            detector.beat(now)
        assert len(detector.gaps) <= 100

    def test_steady_gaps_give_tight_patience(self):
        detector = Adaptive()
        for now in range(10, 500, 10):
            detector.beat(now)
        assert detector.patience() == 10.0

    def test_suspicion_follows_the_patience(self):
        detector = Adaptive()
        for now in range(10, 500, 10):
            detector.beat(now)
        assert not detector.suspects(495)
        assert detector.suspects(510)


class TestWatch:
    def test_a_steady_line_never_alarms(self):
        gaps = [INTERVAL] * 200
        alarms, delay = _watch(Fixed(timeout=15), gaps, None)
        assert alarms == 0 and delay is None

    def test_a_death_is_detected(self):
        gaps = [INTERVAL] * 200
        die = sum(gaps[:100])
        _, delay = _watch(Fixed(timeout=15), gaps[:100], die)
        assert delay == 16

    def test_the_gap_makers_are_deterministic(self):
        assert _jittery_gaps(7) == _jittery_gaps(7)
        assert _shifted_gaps() == _shifted_gaps()


class TestClaims:
    @pytest.mark.parametrize(
        "claim",
        [
            the_fixed_dial_trades_alarms_for_delay,
            the_learned_patience_is_a_number_it_picked,
            the_learned_detector_survives_the_regime_change,
            patience_stretches_with_the_observed_spread,
        ],
    )
    def test_claim_holds(self, claim):
        assert claim() is True

    def test_summary_is_all_true(self):
        told = summarise()
        assert all(value for name, value in told.items() if name != "module")
