from __future__ import annotations

import pytest

from store import circuit as mod
from store.circuit import Breaker, Dependency
from store.errors import ConfigError


def wired(healthy: bool = True, threshold: int = 3, cooldown: int = 5) -> Breaker:
    return Breaker(
        dependency=Dependency(healthy=healthy), threshold=threshold, cooldown=cooldown
    )


class TestClosed:
    def test_bad_settings_are_refused(self):
        with pytest.raises(ConfigError):
            Breaker(dependency=Dependency(), threshold=0)

    def test_a_healthy_call_succeeds(self):
        breaker = wired()
        assert breaker.call() and breaker.state == "closed"

    def test_a_success_resets_the_failure_count(self):
        breaker = wired(healthy=False)
        breaker.call()
        breaker.dependency.healthy = True
        breaker.call()
        assert breaker.recent_failures == 0

    def test_failures_below_the_threshold_stay_closed(self):
        breaker = wired(healthy=False, threshold=3)
        breaker.call()
        breaker.call()
        assert breaker.state == "closed"

    def test_the_threshold_opens_the_breaker(self):
        breaker = wired(healthy=False, threshold=3)
        for _ in range(3):
            breaker.call()
        assert breaker.state == "open"


class TestOpen:
    def test_open_calls_fail_fast(self):
        breaker = wired(healthy=False, threshold=1)
        breaker.call()
        calls_before = breaker.dependency.calls
        assert not breaker.call()
        assert breaker.dependency.calls == calls_before
        assert breaker.fast_failures == 1

    def test_the_cooldown_admits_a_probe(self):
        breaker = wired(healthy=False, threshold=1, cooldown=2)
        breaker.call()
        for _ in range(3):
            breaker.tick()
        breaker.call()
        assert breaker.probes == 1

    def test_a_failed_probe_reopens(self):
        breaker = wired(healthy=False, threshold=1, cooldown=2)
        breaker.call()
        for _ in range(3):
            breaker.tick()
        breaker.call()
        assert breaker.state == "open"

    def test_a_successful_probe_closes(self):
        breaker = wired(healthy=False, threshold=1, cooldown=2)
        breaker.call()
        breaker.dependency.healthy = True
        for _ in range(3):
            breaker.tick()
        breaker.call()
        assert breaker.state == "closed"

    def test_a_closed_breaker_flows_normally(self):
        breaker = wired(healthy=False, threshold=1, cooldown=1)
        breaker.call()
        breaker.dependency.healthy = True
        breaker.tick()
        breaker.call()
        assert breaker.call() and breaker.state == "closed"


class TestMeasurements:
    def test_timeouts_become_fast_failures(self):
        assert mod.the_breaker_converts_timeouts_into_fast_failures()

    def test_the_dependency_gets_quiet(self):
        assert mod.the_open_breaker_spares_the_dependency()

    def test_one_probe_per_cooldown(self):
        assert mod.exactly_one_probe_tests_each_cooldown()

    def test_recovery_is_noticed(self):
        assert mod.recovery_is_noticed_within_one_cooldown()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4
