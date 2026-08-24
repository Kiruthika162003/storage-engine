from __future__ import annotations

import pytest

from store import hedged as mod
from store.errors import ConfigError
from store.hedged import Trial, run


class TestTrial:
    def test_an_empty_trial_refuses_to_rank(self):
        with pytest.raises(ConfigError):
            Trial().percentile(50)

    def test_the_median_of_a_run_is_the_middle(self):
        trial = Trial(latencies=[1.0, 2.0, 3.0])
        assert trial.percentile(50) == 2.0


class TestRun:
    def test_the_unhedged_run_sends_one_request_per_call(self):
        trial = run(None, calls=500)
        assert trial.requests_sent == 500

    def test_the_eager_run_sends_two_per_call(self):
        trial = run(0.0, calls=500)
        assert trial.requests_sent == 1000

    def test_a_thresholded_run_sends_between(self):
        plain = run(None, calls=2000)
        hedged = run(plain.percentile(95), calls=2000)
        assert 2000 < hedged.requests_sent < 2300

    def test_every_call_yields_a_latency(self):
        assert len(run(None, calls=300).latencies) == 300

    def test_runs_are_deterministic(self):
        assert run(None, calls=300).latencies == run(None, calls=300).latencies

    def test_hedging_never_worsens_a_latency(self):
        plain = run(None, calls=3000)
        hedged = run(plain.percentile(95), calls=3000)
        assert hedged.percentile(99) <= plain.percentile(99)


class TestMeasurements:
    def test_p95_hedging_buys_the_tail(self):
        assert mod.hedging_at_the_p95_collapses_the_p99_for_five_percent_load()

    def test_eager_hedging_overpays(self):
        assert mod.hedging_immediately_doubles_the_load_for_little_more_tail()

    def test_thresholds_spare_the_median(self):
        assert mod.thresholded_hedging_leaves_the_median_and_eager_moves_it()

    def test_the_body_is_the_floor(self):
        assert mod.hedging_cannot_beat_the_distributions_body()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4
