from __future__ import annotations

import pytest

from store import retry as mod
from store.errors import ConfigError
from store.retry import Service, peak_after_recovery, run_outage


class TestService:
    def test_a_downed_service_refuses(self):
        service = Service(capacity=10, down_until=5)
        assert not service.offer(0)

    def test_a_recovered_service_admits_to_capacity(self):
        service = Service(capacity=2, down_until=0)
        assert service.offer(1) and service.offer(1) and not service.offer(1)

    def test_capacity_resets_each_tick(self):
        service = Service(capacity=1, down_until=0)
        assert service.offer(1) and service.offer(2)

    def test_the_peak_tracks_the_busiest_tick(self):
        service = Service(capacity=10, down_until=0)
        service.offer(1)
        service.offer(2)
        service.offer(2)
        assert service.peak == 2

    def test_peak_after_recovery_ignores_the_outage(self):
        service = Service(capacity=10, down_until=5)
        service.offer(1)
        service.offer(1)
        service.offer(6)
        assert peak_after_recovery(service) == 1


class TestRunOutage:
    def test_an_unknown_discipline_is_refused(self):
        with pytest.raises(ConfigError):
            run_outage("psychic")

    def test_the_fixed_run_finishes(self):
        _, finished = run_outage("fixed")
        assert finished < 400

    def test_the_jittered_run_finishes_fastest(self):
        _, fixed = run_outage("fixed")
        _, jittered = run_outage("jittered")
        assert jittered < fixed

    def test_the_backoff_run_is_slowest(self):
        _, fixed = run_outage("fixed")
        _, backoff = run_outage("backoff")
        assert backoff > fixed

    def test_runs_are_deterministic(self):
        first = run_outage("jittered")[0].arrivals
        second = run_outage("jittered")[0].arrivals
        assert first == second


class TestMeasurements:
    def test_fixed_delays_hammer(self):
        assert mod.fixed_delays_keep_the_herd_and_hammer_the_recovery()

    def test_bare_backoff_never_breaks_the_herd(self):
        assert mod.backoff_without_jitter_spaces_the_herd_but_never_breaks_it()

    def test_jitter_wins_both_meters(self):
        assert mod.jitter_breaks_the_herd_and_wins_both_meters()

    def test_down_means_down(self):
        assert mod.the_downed_service_admits_nobody()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4
