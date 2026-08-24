from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

from store.errors import ConfigError

# Priority scheduling and the starvation it breeds, cured by aging.
#
# The load shedder ranked classes at the door; a scheduler ranks them at service time, and
# strict priority has a famous pathology: under sustained high-priority load, low
# priority work waits forever, not slowly, forever. Aging is the classic cure, a job's
# effective priority rising with its wait, so patience is a currency and every job
# eventually outbids the fresh arrivals. The measurements build the starvation with a
# literal infinite wait, then show aging bounding the worst wait at a knob-controlled
# cost to the high class, because the cure is a trade and the knob deserves a number.


@dataclass
class Job:
    """One unit of work."""

    name: int
    priority: int
    arrived: int


@dataclass
class Scheduler:
    """Serves one job per tick, strict or aging."""

    aging_rate: float
    queue: list[Job] = field(default_factory=list)
    now: int = field(default=0)
    waits: dict[int, list[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.aging_rate < 0:
            raise ConfigError("the aging rate is not negative")

    def submit(self, job: Job) -> None:
        self.queue.append(job)

    def effective(self, job: Job) -> float:
        return job.priority + self.aging_rate * (self.now - job.arrived)

    def tick(self) -> Job | None:
        """Serve the best effective priority, ties to the oldest."""
        self.now += 1
        if not self.queue:
            return None
        best = max(self.queue, key=lambda job: (self.effective(job), -job.arrived))
        self.queue.remove(best)
        self.waits.setdefault(best.priority, []).append(self.now - best.arrived)
        return best

    def worst_wait(self, priority: int) -> int:
        served = self.waits.get(priority, [])
        pending = [self.now - job.arrived for job in self.queue if job.priority == priority]
        candidates = served + pending
        return max(candidates, default=0)


def _pressure(scheduler: Scheduler, ticks: int = 500, seed: int = 401) -> None:
    """One low job at the start, then high arrivals that never let the queue drain.

    The first draft arrived at 0.95 per tick against a service rate of one, and the queue
    emptied in the five percent gaps, serving the low job at tick seven under strict
    priority: no starvation, because starvation needs the pressure to be sustained, not
    merely heavy. One arrival every tick plus an occasional second keeps the queue fed,
    which is the regime the pathology lives in.
    """
    source = random.Random(seed)
    scheduler.submit(Job(name=0, priority=0, arrived=0))
    name = 1
    for _ in range(ticks):
        scheduler.submit(Job(name=name, priority=10, arrived=scheduler.now))
        name += 1
        if source.random() < 0.1:
            scheduler.submit(Job(name=name, priority=10, arrived=scheduler.now))
            name += 1
        scheduler.tick()


@functools.cache
def strict_priority_starves_the_low_job_literally() -> bool:
    """Five hundred ticks of high traffic and the low job has waited all five hundred.

    Not slowly served, never served: any high arrival outranks it forever, and the
    starvation is unbounded by construction, which the pending-wait meter states as a
    number that equals the whole run.
    """
    scheduler = Scheduler(aging_rate=0.0)
    _pressure(scheduler)
    return scheduler.worst_wait(0) >= 500 and not scheduler.waits.get(0)


@functools.cache
def aging_bounds_the_wait_by_the_priority_gap_over_the_rate() -> bool:
    """At rate 0.05 the low job is served at tick 220, near the gap over the rate.

    The low job needs its age times the rate to cover the priority gap of ten, which is
    two hundred ticks, and the measured service lands within a few ticks of it. The bound
    is arithmetic, not tuning folklore: worst wait equals gap over rate plus queueing
    noise, so the knob converts directly into a promise.
    """
    scheduler = Scheduler(aging_rate=0.05)
    _pressure(scheduler)
    waits = scheduler.waits.get(0, [])
    return len(waits) == 1 and 180 <= waits[0] <= 240


@functools.cache
def a_faster_rate_shortens_the_worst_wait_and_costs_the_high_class() -> bool:
    """Rate 0.2 serves the low job at 54 and pushes the high-class mean wait up.

    The trade on both meters: the low job's wait falls fourfold with the fourfold rate,
    and the high class pays with its own queue time, because every tick spent on aged
    work is a tick a fresh high job waits. Aging does not create capacity either; it
    redistributes waiting, and the rate decides the exchange.
    """
    slow = Scheduler(aging_rate=0.05)
    fast = Scheduler(aging_rate=0.2)
    _pressure(slow)
    _pressure(fast)
    slow_low = slow.waits.get(0, [999])[0]
    fast_low = fast.waits.get(0, [999])[0]
    slow_high = sum(slow.waits[10]) / len(slow.waits[10])
    fast_high = sum(fast.waits[10]) / len(fast.waits[10])
    return fast_low < slow_low / 2 and fast_high >= slow_high


@functools.cache
def equal_priorities_serve_in_arrival_order() -> bool:
    """With one class and no aging pressure, the scheduler is FIFO.

    The tie-break is the oldest job, so a single-class workload degenerates to the fair
    queue, which is the sanity floor: a scheduler that reorders equals is injecting
    latency variance for nothing.
    """
    scheduler = Scheduler(aging_rate=0.1)
    for at in range(5):
        scheduler.submit(Job(name=at, priority=5, arrived=at))
    served = [scheduler.tick().name for _ in range(5)]
    return served == [0, 1, 2, 3, 4]


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "strict_priority_starves": strict_priority_starves_the_low_job_literally(),
        "the_bound_is_gap_over_rate": aging_bounds_the_wait_by_the_priority_gap_over_the_rate(),
        "the_rate_is_the_trade": a_faster_rate_shortens_the_worst_wait_and_costs_the_high_class(),
        "equals_are_fifo": equal_priorities_serve_in_arrival_order(),
    }
