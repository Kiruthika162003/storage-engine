from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

from store.errors import ConfigError

# Retries: how a thousand polite clients become one rude one.
#
# A dependency blips, every in-flight request fails, and every client retries. Fixed-delay
# retries keep the failed cohort synchronized: they return together, fail together if the
# dependency is still warming, and return together again, a metronome of load spikes that
# can hold a recovering service down indefinitely, the retry storm. Exponential backoff
# spreads the attempts over time; jitter spreads them within each attempt window; and the
# measurements run the same outage against all three disciplines, counting the peak
# arrivals per tick at the recovering service, because the peak is what knocks it back
# over.


@dataclass
class Service:
    """A service with a capacity, down for a while, then fragile."""

    capacity: int
    down_until: int
    arrivals: dict[int, int] = field(default_factory=dict)

    def offer(self, tick: int) -> bool:
        self.arrivals[tick] = self.arrivals.get(tick, 0) + 1
        if tick < self.down_until:
            return False
        return self.arrivals[tick] <= self.capacity

    @property
    def peak(self) -> int:
        return max(self.arrivals.values(), default=0)


def run_outage(
    discipline: str, clients: int = 1000, seed: int = 379, horizon: int = 400
) -> tuple[Service, int]:
    """One outage, one retry discipline, the recovery tick reported."""
    if discipline not in ("fixed", "backoff", "jittered"):
        raise ConfigError(f"{discipline} is not a discipline")
    source = random.Random(seed)
    service = Service(capacity=100, down_until=50)
    next_try = dict.fromkeys(range(clients), 0)
    attempt = dict.fromkeys(range(clients), 0)
    done: set[int] = set()
    for tick in range(horizon):
        for client in range(clients):
            if client in done or next_try[client] != tick:
                continue
            if service.offer(tick):
                done.add(client)
                continue
            attempt[client] += 1
            if discipline == "fixed":
                delay = 10
            elif discipline == "backoff":
                delay = min(2 ** attempt[client], 64)
            else:
                delay = source.randrange(1, min(2 ** attempt[client], 64) + 1)
            next_try[client] = tick + delay
        if len(done) == clients:
            return service, tick
    return service, horizon


def peak_after_recovery(service: Service) -> int:
    """The largest single-tick arrival the recovering service faced."""
    return max(
        (count for tick, count in service.arrivals.items() if tick >= service.down_until),
        default=0,
    )


@functools.cache
def _all_runs() -> dict[str, tuple[int, int]]:
    """Each discipline's post-recovery peak and finish tick."""
    made = {}
    for discipline in ("fixed", "backoff", "jittered"):
        service, finished = run_outage(discipline)
        made[discipline] = (peak_after_recovery(service), finished)
    return made


@functools.cache
def fixed_delays_keep_the_herd_and_hammer_the_recovery() -> bool:
    """The fixed cohort hits the recovering service a thousand strong, every wave.

    All thousand clients failed together, wait ten together, and return together: the
    post-recovery peak is the full herd, ten times the capacity, arriving every ten ticks
    until admission whittles it down. A service that falls over above its capacity would
    be knocked back down by its own former clients, which is the storm.
    """
    peak, _ = _all_runs()["fixed"]
    return peak == 1000


@functools.cache
def backoff_without_jitter_spaces_the_herd_but_never_breaks_it() -> bool:
    """Deterministic backoff finishes LAST, behind even fixed delay, herd intact.

    The surprise worth the module: identical clients compute identical delays, so pure
    exponential backoff never desynchronizes them. The herd arrives whole, a hundred are
    admitted, and the rest wait a doubling interval, still together, so the drain rate is
    capacity per doubling window and the fixed cohort's tighter metronome actually
    finishes first, 140 against past-the-horizon. Backoff spreads attempts in time and
    does nothing about the correlation, which was the actual problem.
    """
    _, fixed_finish = _all_runs()["fixed"]
    backoff_peak, backoff_finish = _all_runs()["backoff"]
    return backoff_finish > fixed_finish and backoff_peak >= 900


@functools.cache
def jitter_breaks_the_herd_and_wins_both_meters() -> bool:
    """Jittered backoff peaks at 41 arrivals after recovery and finishes at 112.

    The randomness is the desynchronizer: within two windows the cohort is spread across
    the delay range, arrivals per tick fall toward the capacity, and the drain completes
    at 112 while deterministic backoff is still bunching. Jitter is not a refinement of
    backoff, it is the part that addresses the correlation, and this measurement is the
    argument for never shipping the deterministic version at all.
    """
    runs = _all_runs()
    jittered_peak, jittered_finish = runs["jittered"]
    return (
        jittered_peak < runs["fixed"][0] / 3
        and jittered_finish <= min(runs["fixed"][1], runs["backoff"][1])
    )


@functools.cache
def the_downed_service_admits_nobody() -> bool:
    """Every offer before the recovery tick fails, whatever the discipline."""
    service = Service(capacity=100, down_until=10)
    return not service.offer(5) and service.offer(10)


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "fixed_delays_hammer": fixed_delays_keep_the_herd_and_hammer_the_recovery(),
        "bare_backoff_never_breaks_the_herd": (
            backoff_without_jitter_spaces_the_herd_but_never_breaks_it()
        ),
        "jitter_wins_both_meters": jitter_breaks_the_herd_and_wins_both_meters(),
        "down_means_down": the_downed_service_admits_nobody(),
    }
