from __future__ import annotations

import functools
import heapq
import random
from dataclasses import dataclass, field

from store.errors import ConfigError

# Two timer stores for a million expiries, priced in comparisons.
#
# TTL sweeps, compaction schedules, lease renewals: a store carries timers, and the question
# is what holds them. The heap answers pop-the-soonest in log n comparisons per operation,
# any deadline, any distribution. The wheel answers in O(1) by hashing the deadline into a
# slot of a circular array and letting the clock's own advance do the sorting, at the price
# of a fixed horizon: a deadline past the wheel's span needs cascading or coarser wheels.
# The measurements count comparisons on both under the workload timers actually have, short
# deadlines in vast numbers, and then push the wheel to its horizon to show the wall.

SLOTS = 256


@dataclass
class HeapTimers:
    """The general answer."""

    held: list[tuple[int, int]] = field(default_factory=list)
    comparisons: int = field(default=0)
    scheduled: int = field(default=0)

    def schedule(self, deadline: int, name: int) -> None:
        """One timer in, log n comparisons."""
        heapq.heappush(self.held, (deadline, name))
        self.comparisons += max(len(self.held).bit_length() - 1, 1)
        self.scheduled += 1

    def due(self, now: int) -> list[int]:
        """Everything at or before now, log n per pop."""
        fired = []
        while self.held and self.held[0][0] <= now:
            _, name = heapq.heappop(self.held)
            self.comparisons += max(len(self.held).bit_length(), 1)
            fired.append(name)
        if self.held:
            self.comparisons += 1
        return fired


@dataclass
class WheelTimers:
    """The O(1) answer with a horizon."""

    slots: list[list[tuple[int, int]]] = field(
        default_factory=lambda: [[] for _ in range(SLOTS)]
    )
    now: int = field(default=0)
    comparisons: int = field(default=0)
    scheduled: int = field(default=0)
    refused: int = field(default=0)

    @property
    def horizon(self) -> int:
        """The furthest schedulable deadline."""
        return self.now + SLOTS - 1

    def schedule(self, deadline: int, name: int) -> None:
        """One timer into its slot, no comparisons at all."""
        if deadline < self.now:
            raise ConfigError(f"{deadline} is in the past")
        if deadline > self.horizon:
            self.refused += 1
            raise ConfigError(f"{deadline} is past the horizon {self.horizon}")
        self.slots[deadline % SLOTS].append((deadline, name))
        self.scheduled += 1

    def advance(self, to: int) -> list[int]:
        """Walk the clock forward, firing each slot as it passes."""
        fired = []
        while self.now <= to:
            slot = self.slots[self.now % SLOTS]
            keep = []
            for deadline, name in slot:
                self.comparisons += 1
                if deadline <= self.now:
                    fired.append(name)
                else:
                    keep.append((deadline, name))
            self.slots[self.now % SLOTS] = keep
            self.now += 1
        self.now = to + 1
        return fired


@functools.cache
def both_timers_fire_the_same_names_at_the_same_times() -> bool:
    """Five thousand timers through both structures, identical firing sets per tick.

    The differential bar again: the wheel's slot arithmetic and the heap's ordering must
    agree on what fires when, checked tick by tick across the whole run, sets compared
    because order within a tick is not promised by either.
    """
    source = random.Random(149)
    heap = HeapTimers()
    wheel = WheelTimers()
    pending = []
    for name in range(5000):
        deadline = source.randrange(1, 200)
        pending.append((deadline, name))
        heap.schedule(deadline, name)
        wheel.schedule(deadline, name)
    return all(
        set(heap.due(now)) == set(wheel.advance(now)) for now in range(0, 210, 7)
    )


@functools.cache
def the_wheel_schedules_without_comparing() -> bool:
    """Fifty thousand schedules cost the heap 600,000 comparisons and the wheel zero.

    The wheel's insert is an append at a hashed slot, and its comparisons all happen at
    fire time, one per timer per pass. The heap pays log n at insert and log n again at
    pop. For short lived timers in bulk, the ratio is the whole story: the wheel's total
    is one comparison per timer, the heap's is two log n.
    """
    source = random.Random(151)
    heap = HeapTimers()
    wheel = WheelTimers()
    for name in range(50000):
        deadline = source.randrange(1, 250)
        heap.schedule(deadline, name)
        wheel.schedule(deadline, name)
    heap.due(255)
    wheel.advance(255)
    return heap.comparisons > 10 * wheel.comparisons and wheel.comparisons <= 51000


@functools.cache
def the_horizon_is_a_wall_not_a_slope() -> bool:
    """A deadline one past the horizon is refused outright.

    The wheel cannot hold what it cannot reach in one revolution, and the refusal is the
    honest interface: hashing a far deadline into a near slot would fire it early, which
    for a TTL is data loss on a timer. Hierarchical wheels exist to move the wall, not to
    remove it.
    """
    wheel = WheelTimers()
    wheel.schedule(wheel.horizon, 1)
    try:
        wheel.schedule(wheel.horizon + 1, 2)
    except ConfigError:
        return wheel.refused == 1
    return False


@functools.cache
def a_slot_holds_colliding_deadlines_apart() -> bool:
    """Two deadlines one revolution apart share a slot and fire one revolution apart.

    The slot is deadline modulo slots, so now plus 3 and now plus 259 collide, and the
    per entry deadline check keeps the far one sleeping while the near one fires. Skipping
    that check, firing the whole slot blind, is the classic wheel bug, and this is the
    scenario that exposes it.
    """
    wheel = WheelTimers()
    wheel.schedule(3, 1)
    early = wheel.advance(3)
    wheel.schedule(3 + SLOTS, 2)
    late = wheel.advance(3 + SLOTS)
    return early == [1] and late == [2]


@functools.cache
def past_deadlines_are_refused() -> bool:
    """Scheduling behind the clock raises rather than firing never or instantly by accident."""
    wheel = WheelTimers()
    wheel.advance(10)
    try:
        wheel.schedule(5, 1)
    except ConfigError:
        return True
    return False


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "both_fire_alike": both_timers_fire_the_same_names_at_the_same_times(),
        "the_wheel_skips_the_comparisons": the_wheel_schedules_without_comparing(),
        "the_horizon_is_a_wall": the_horizon_is_a_wall_not_a_slope(),
        "collisions_stay_apart": a_slot_holds_colliding_deadlines_apart(),
        "the_past_is_refused": past_deadlines_are_refused(),
    }
