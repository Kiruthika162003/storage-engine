from __future__ import annotations

import functools
from dataclasses import dataclass, field

from store.errors import ConfigError

# The circuit breaker: failing fast on purpose, and the probe that ends it.
#
# The retry module protected the dependency from the herd; the breaker protects the caller
# from the dependency. While a dependency is down, every call to it costs a full timeout,
# and a service whose threads are all waiting on a dead dependency is itself down, the
# cascade. The breaker counts recent failures, opens past a threshold, and answers open
# circuit calls instantly with a refusal, converting timeout latency into an error the
# caller planned for. The half-open state is the subtle part: after a cooldown, exactly
# one probe call goes through, success closing the breaker and failure reopening it,
# because a thundering herd of probes at the first cooldown is the retry storm again
# through a different door.


@dataclass
class Dependency:
    """The thing behind the breaker, scriptable."""

    healthy: bool = field(default=True)
    calls: int = field(default=0)
    timeout_cost: int = field(default=30)

    def call(self) -> bool:
        self.calls += 1
        return self.healthy


@dataclass
class Breaker:
    """Closed, open, half-open."""

    dependency: Dependency
    threshold: int = field(default=5)
    cooldown: int = field(default=20)
    state: str = field(default="closed")
    recent_failures: int = field(default=0)
    opened_at: int = field(default=-1)
    now: int = field(default=0)
    fast_failures: int = field(default=0)
    latency_spent: int = field(default=0)
    probes: int = field(default=0)

    def __post_init__(self) -> None:
        if self.threshold < 1 or self.cooldown < 1:
            raise ConfigError("the breaker needs positive settings")

    def tick(self) -> None:
        self.now += 1

    def call(self) -> bool:
        """One caller's attempt through the breaker."""
        if self.state == "open":
            if self.now - self.opened_at >= self.cooldown:
                self.state = "half_open"
            else:
                self.fast_failures += 1
                return False
        if self.state == "half_open":
            self.probes += 1
            ok = self.dependency.call()
            self.latency_spent += 1 if ok else self.dependency.timeout_cost
            if ok:
                self.state = "closed"
                self.recent_failures = 0
                return True
            self.state = "open"
            self.opened_at = self.now
            return False
        ok = self.dependency.call()
        self.latency_spent += 1 if ok else self.dependency.timeout_cost
        if ok:
            self.recent_failures = 0
            return True
        self.recent_failures += 1
        if self.recent_failures >= self.threshold:
            self.state = "open"
            self.opened_at = self.now
        return False


def _outage(with_breaker: bool, down_ticks: int = 100, calls_per_tick: int = 10):
    """An outage and recovery; latency accounted, outage-window calls counted."""
    dependency = Dependency(healthy=False)
    breaker = Breaker(dependency=dependency)
    naive_latency = 0
    calls_during_outage = 0
    for tick in range(down_ticks + 40):
        if tick == down_ticks:
            dependency.healthy = True
            calls_during_outage = dependency.calls
        breaker.tick()
        for _ in range(calls_per_tick):
            if with_breaker:
                breaker.call()
            else:
                ok = dependency.call()
                naive_latency += 1 if ok else dependency.timeout_cost
    return breaker, naive_latency, calls_during_outage


@functools.cache
def the_breaker_converts_timeouts_into_fast_failures() -> bool:
    """Through the outage, the naive caller burns 30,400 timeout units; the breaker 670.

    The naive path pays the full timeout on every one of a thousand failing calls. The
    breaker pays it five times, opens, and answers the rest instantly: the fast failure
    count absorbs what would have been timeout waits, which is the caller's threads handed
    back to the caller.
    """
    breaker, _, _ = _outage(with_breaker=True)
    _, naive_latency, _ = _outage(with_breaker=False)
    return (
        naive_latency > 25000
        and breaker.latency_spent < naive_latency / 20
        and breaker.fast_failures > 800
    )


@functools.cache
def the_open_breaker_spares_the_dependency() -> bool:
    """During the outage the dependency sees ten calls instead of a thousand.

    Every open-circuit refusal is a call the dependency never received, the retry
    module's peak argument at the level of one caller: recovery needs quiet, and the
    breaker is how one caller volunteers its share. The count is taken at the recovery
    tick, because afterwards the dependency is healthy and calls are the point.
    """
    _, _, during = _outage(with_breaker=True)
    _, _, naive_during = _outage(with_breaker=False)
    return during < 15 and naive_during == 1000


@functools.cache
def exactly_one_probe_tests_each_cooldown() -> bool:
    """During a long outage, probes arrive one per cooldown window, not one per caller.

    Ten callers per tick and a twenty tick cooldown produce probes only when the half-open
    door opens, and the first caller through it closes the door behind them by reopening
    the breaker on failure. Probe count across a hundred down ticks stays a handful, which
    is the anti-herd property the half-open state exists for.
    """
    breaker, _, _ = _outage(with_breaker=True)
    return 2 <= breaker.probes <= 8


@functools.cache
def recovery_is_noticed_within_one_cooldown() -> bool:
    """After the dependency heals, the next probe closes the breaker and calls flow.

    The cost of the fail-fast posture is discovery latency, at most one cooldown of
    refusing calls that would now succeed, and the breaker ends the outage-plus-cooldown
    window in the closed state with recent failures at zero.
    """
    breaker, _, _ = _outage(with_breaker=True)
    return breaker.state == "closed" and breaker.recent_failures == 0


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "timeouts_become_fast_failures": the_breaker_converts_timeouts_into_fast_failures(),
        "the_dependency_gets_quiet": the_open_breaker_spares_the_dependency(),
        "one_probe_per_cooldown": exactly_one_probe_tests_each_cooldown(),
        "recovery_is_noticed": recovery_is_noticed_within_one_cooldown(),
    }
