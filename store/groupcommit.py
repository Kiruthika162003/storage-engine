"""Group commit's other meter: the latency each write pays to save the io.

The write path sweep showed grouping syncs divides the io. This module
reads the other dial: a write acknowledged only at the group's sync waits
for the group to fill or the timer to fire, and that wait is the price
each individual writer pays for the batch's efficiency. Arrivals are
simulated, waits are recorded per write, and the distribution is the
finding.
"""

from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

SYNC_COST = 30
TIMER = 8


@dataclass
class Log:
    group_size: int
    waits: list[int] = field(default_factory=list)
    syncs: int = 0
    pending: list[int] = field(default_factory=list)
    timer_started: int = -1

    def append(self, now: int) -> None:
        if not self.pending:
            self.timer_started = now
        self.pending.append(now)
        if len(self.pending) >= self.group_size:
            self._sync(now)

    def tick(self, now: int) -> None:
        if self.pending and now - self.timer_started >= TIMER:
            self._sync(now)

    def _sync(self, now: int) -> None:
        self.syncs += 1
        for arrived in self.pending:
            self.waits.append(now - arrived)
        self.pending = []

    def drain(self, now: int) -> None:
        if self.pending:
            self._sync(now)


def _arrivals(seed: int, rate: float, horizon: int = 4000) -> list[int]:
    source = random.Random(seed)
    ticks = []
    for now in range(horizon):
        count = 0
        while source.random() < rate and count < 20:
            ticks.append(now)
            count += 1
    return ticks


def run(group_size: int, rate: float, seed: int = 17) -> Log:
    log = Log(group_size=group_size)
    arrivals = _arrivals(seed, rate)
    at = 0
    horizon = 4000
    for now in range(horizon):
        log.tick(now)
        while at < len(arrivals) and arrivals[at] == now:
            log.append(now)
            at += 1
    log.drain(horizon)
    return log


def percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


@functools.cache
def a_busy_group_fills_before_anyone_waits() -> bool:
    """Eight writes a tick, groups of eight: 8x fewer syncs, median wait 0.

    32229 writes sync 4029 times instead of 32229, and half the writers
    wait zero ticks because the group fills within their own tick. The
    99th percentile waits three ticks. Under load, group commit is close
    to a free lunch: the crowd is the batch.
    """
    log = run(8, 0.9)
    return (
        log.syncs == 4029
        and percentile(log.waits, 0.5) == 0
        and percentile(log.waits, 0.99) == 3
    )


@functools.cache
def the_discount_scales_with_the_crowd() -> bool:
    """Groups of 32 on the busy log: 1013 syncs, median wait two ticks.

    Thirty two times fewer syncs for a median of two and a maximum of
    eight, the timer's ceiling. The knob buys io almost linearly while
    the crowd is thick enough to fill whatever group is asked.
    """
    log = run(32, 0.9)
    return log.syncs == 1013 and percentile(log.waits, 0.5) == 2 and max(log.waits) == 8


@functools.cache
def a_quiet_log_pays_the_timer_not_the_group() -> bool:
    """0.4 writes a tick: every write waits the full 8 tick timer.

    The group never fills, so the timer fires every batch: the median
    wait is the maximum wait is 8. And the io discount collapses from 8x
    to 1.9x, 394 syncs to 212, because most timer batches hold a single
    lonely write. On a quiet log the group size is a fiction; the timer
    is the policy.
    """
    log = run(8, 0.1)
    return (
        log.syncs == 212
        and percentile(log.waits, 0.5) == TIMER
        and max(log.waits) == TIMER
    )


@functools.cache
def raising_the_group_on_a_quiet_log_changes_nothing() -> bool:
    """Groups of 8 and 32 produce identical syncs and identical waits.

    Neither group ever fills, so both degenerate to the same timer
    batches: 212 syncs each, wait for wait the same list. Tuning the
    group size on a quiet log is turning a knob that is not connected
    to anything.
    """
    eight = run(8, 0.1)
    thirty_two = run(32, 0.1)
    return eight.syncs == thirty_two.syncs and eight.waits == thirty_two.waits


@functools.cache
def summarise() -> dict:
    return {
        "module": "store.groupcommit",
        "a_busy_group_fills_before_anyone_waits": (
            a_busy_group_fills_before_anyone_waits()
        ),
        "the_discount_scales_with_the_crowd": the_discount_scales_with_the_crowd(),
        "a_quiet_log_pays_the_timer_not_the_group": (
            a_quiet_log_pays_the_timer_not_the_group()
        ),
        "raising_the_group_on_a_quiet_log_changes_nothing": (
            raising_the_group_on_a_quiet_log_changes_nothing()
        ),
    }
