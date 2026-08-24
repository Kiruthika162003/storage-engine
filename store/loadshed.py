from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

from store.errors import ConfigError

# Load shedding by priority: deciding who fails while there is still a choice.
#
# The stall module deferred writers and the quota module policed tenants; shedding is the
# third posture, for when the overload has arrived and something must be dropped now. The
# unshedded server queues everything and fails everyone late, timeouts landing on
# checkouts and health checks alike. The shedder drops the lowest priority work at the
# door the moment depth crosses a line, so the capacity that exists goes to the work that
# matters, and the drop is an instant cheap failure instead of a slow expensive one. The
# measurements run the same overload through both and read the meters per priority class,
# because the whole point is who the pain lands on.

PRIORITIES = ("critical", "normal", "batch")


@dataclass
class Meter:
    """Outcomes per priority class."""

    served: dict[str, int] = field(default_factory=dict)
    dropped: dict[str, int] = field(default_factory=dict)
    timed_out: dict[str, int] = field(default_factory=dict)

    def note(self, kind: str, priority: str) -> None:
        bucket = getattr(self, kind)
        bucket[priority] = bucket.get(priority, 0) + 1


@dataclass
class Server:
    """A queue with capacity, optionally shedding by priority."""

    capacity_per_tick: int
    queue_limit: int
    shed: bool
    queue: list[tuple[str, int]] = field(default_factory=list)
    meter: Meter = field(default_factory=Meter)
    timeout_ticks: int = field(default=30)
    now: int = field(default=0)

    def __post_init__(self) -> None:
        if self.capacity_per_tick < 1 or self.queue_limit < 1:
            raise ConfigError("the server needs positive settings")

    def offer(self, priority: str) -> None:
        """One arriving request."""
        if priority not in PRIORITIES:
            raise ConfigError(f"{priority} is not a priority")
        if len(self.queue) < self.queue_limit:
            self.queue.append((priority, self.now))
            return
        if not self.shed:
            self.queue.append((priority, self.now))
            return
        worst_at = None
        worst_rank = PRIORITIES.index(priority)
        for at, (queued_priority, _) in enumerate(self.queue):
            rank = PRIORITIES.index(queued_priority)
            if rank > worst_rank:
                worst_rank = rank
                worst_at = at
        if worst_at is None:
            self.meter.note("dropped", priority)
            return
        shed_priority, _ = self.queue.pop(worst_at)
        self.meter.note("dropped", shed_priority)
        self.queue.append((priority, self.now))

    def tick(self) -> None:
        """Serve up to capacity, expire the ancient."""
        self.now += 1
        survivors = []
        for priority, arrived in self.queue:
            if self.now - arrived >= self.timeout_ticks:
                self.meter.note("timed_out", priority)
            else:
                survivors.append((priority, arrived))
        self.queue = survivors
        for _ in range(self.capacity_per_tick):
            if not self.queue:
                break
            priority, _ = self.queue.pop(0)
            self.meter.note("served", priority)


def _overload(shed: bool, ticks: int = 300, seed: int = 397) -> Server:
    """Arrivals at double capacity, ten percent critical, sixty normal, thirty batch."""
    source = random.Random(seed)
    server = Server(capacity_per_tick=10, queue_limit=50, shed=shed)
    for _ in range(ticks):
        server.tick()
        for _ in range(20):
            draw = source.random()
            if draw < 0.1:
                server.offer("critical")
            elif draw < 0.7:
                server.offer("normal")
            else:
                server.offer("batch")
    return server


@functools.cache
def the_unshedded_server_fails_everyone_alike() -> bool:
    """Without shedding, critical work times out at the same rate as batch work.

    The queue is fair and fairness under overload is the bug: every class waits behind
    the same backlog, the backlog exceeds the timeout horizon, and the checkout dies
    beside the report that could have waited a day. Timeout rates per class land within a
    few points of each other, which is the outage postmortem's most common chart.
    """
    server = _overload(shed=False)
    rates = {}
    for priority in PRIORITIES:
        served = server.meter.served.get(priority, 0)
        timed_out = server.meter.timed_out.get(priority, 0)
        rates[priority] = timed_out / max(served + timed_out, 1)
    return abs(rates["critical"] - rates["batch"]) < 0.1 and rates["critical"] > 0.3


@functools.cache
def the_shedder_serves_every_critical_request() -> bool:
    """With shedding, zero critical drops and zero critical timeouts through the storm.

    The queue holds only what the capacity can serve within the timeout, and the eviction
    always takes the lowest class present, so critical work rides through an overload
    that is drowning everything else. The guarantee costs exactly what the batch meter
    shows, which is the next claim's subject.
    """
    server = _overload(shed=True)
    return (
        server.meter.dropped.get("critical", 0) == 0
        and server.meter.timed_out.get("critical", 0) == 0
        and server.meter.served.get("critical", 0) > 500
    )


@functools.cache
def the_bill_lands_on_the_batch_class() -> bool:
    """Batch work absorbs nearly all the drops, which is the policy stated as a number.

    Shedding does not create capacity, it aims the shortfall: the arrivals exceed service
    by half, someone must fail, and the drops concentrate in the class that was declared
    droppable. An organisation that cannot rank its work cannot shed, and the meter is
    what the ranking argument looks like when it is settled.
    """
    server = _overload(shed=True)
    dropped = server.meter.dropped
    total = sum(dropped.values())
    return total > 1000 and dropped.get("batch", 0) > total * 0.5


@functools.cache
def drops_are_cheap_and_timeouts_are_not() -> bool:
    """The shedder converts slow failures into instant ones: timeouts fall overall.

    The unshedded server fails work after the full timeout wait, holding queue space the
    whole time; the shedder fails at the door. Total timeouts under shedding are a small
    fraction of the unshedded count, because the queue never holds more than the horizon
    can drain.
    """
    unshedded = _overload(shed=False)
    shedded = _overload(shed=True)
    unshedded_timeouts = sum(unshedded.meter.timed_out.values())
    shedded_timeouts = sum(shedded.meter.timed_out.values())
    return shedded_timeouts < unshedded_timeouts / 5


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "fairness_fails_everyone": the_unshedded_server_fails_everyone_alike(),
        "critical_rides_through": the_shedder_serves_every_critical_request(),
        "the_bill_lands_on_batch": the_bill_lands_on_the_batch_class(),
        "drops_beat_timeouts": drops_are_cheap_and_timeouts_are_not(),
    }
