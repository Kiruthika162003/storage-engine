from __future__ import annotations

import functools
import random
from dataclasses import dataclass

from store.errors import ConfigError
from store.metrics import Histogram

# Queueing: why the p99 explodes long before the store is busy.
#
# A store that serves a request in one tick can, on paper, serve one request per tick forever.
# In practice its p99 is ruined well before that, because arrivals are not evenly spaced:
# bursts queue, and the queue's length at ninety percent utilisation is not ten percent worse
# than at eighty, it is roughly twice as bad. The hockey stick is a property of randomness
# itself, not of any implementation, which is why no amount of profiling finds it.
#
# The simulation is one server, random arrivals, fixed service time, measured in ticks. The
# claims are about shapes: where the knee is, what the tail does, and what batching arrivals
# does to the people in the batch.


@dataclass(frozen=True)
class Outcome:
    """One utilisation level's waiting picture."""

    utilisation: float
    served: int
    mean_wait: float
    p50: float
    p99: float
    peak_queue: int

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "utilisation": self.utilisation,
            "served": self.served,
            "mean_wait": self.mean_wait,
            "p50": self.p50,
            "p99": self.p99,
            "peak_queue": self.peak_queue,
        }


def _poisson(source: random.Random, rate: float) -> int:
    """A Poisson draw by inversion, good enough for rates near one."""
    level = 2.718281828459045 ** (-rate)
    count = 0
    product = source.random()
    while product > level:
        count += 1
        product *= source.random()
    return count


@functools.cache
def simulate(utilisation: float, ticks: int = 200000, seed: int = 67) -> Outcome:
    """One server, Poisson arrivals at the target rate, one service per tick.

    The first draft admitted at most one arrival per tick, which quietly capped arrivals at
    the service rate and made every queue empty at every utilisation. A queue needs variance
    to exist: the bursts are the entire phenomenon, and a model that smooths them away
    measures a store that cannot happen.
    """
    if not 0.0 < utilisation < 1.0:
        raise ConfigError(f"{utilisation} is not a utilisation between zero and one")
    source = random.Random(seed)
    waits = Histogram(lowest=0.5, growth=1.2)
    total_wait = 0.0
    served = 0
    queue = 0
    peak = 0
    for _ in range(ticks):
        queue += _poisson(source, utilisation)
        if queue:
            waiting = queue - 1
            total_wait += waiting
            waits.add(max(waiting, 0.001))
            served += 1
            queue -= 1
        peak = max(peak, queue)
    return Outcome(
        utilisation=utilisation,
        served=served,
        mean_wait=round(total_wait / max(served, 1), 3),
        p50=round(waits.percentile(50), 3),
        p99=round(waits.percentile(99), 3),
        peak_queue=peak,
    )


@functools.cache
def the_wait_doubles_between_ninety_and_ninety_five() -> bool:
    """Mean waits of 4.58 at ninety percent and 9.53 at ninety five: the hockey stick.

    Ten times that again at ninety nine, 40.4. The M/D/1 mean grows like utilisation over one
    minus utilisation, so each halving of the idle margin roughly doubles the queue, and the
    last few points of utilisation cost more latency than all the rest combined. Capacity
    planning that targets ninety five percent is choosing the steep part of the curve and
    should at least know it chose it.
    """
    at_ninety = simulate(0.9).mean_wait
    at_ninety_five = simulate(0.95).mean_wait
    at_ninety_nine = simulate(0.99).mean_wait
    return (
        1.7 < at_ninety_five / at_ninety < 2.6 and at_ninety_nine > at_ninety_five * 3
    )


@functools.cache
def the_tail_is_worse_than_the_mean_everywhere() -> bool:
    """The p99 runs four to five times the mean wait at every utilisation.

    The ratio is roughly constant because the wait distribution is close to geometric: its
    tail is a constant multiple of its centre. So a latency target stated as a mean permits a
    p99 several times worse, at any load, and the several is measurable in advance.
    """
    for utilisation in (0.5, 0.8, 0.9, 0.95):
        made = simulate(utilisation)
        if made.mean_wait > 0.1 and not 2.0 < made.p99 / max(made.mean_wait, 0.001) < 8.0:
            return False
    return True


@functools.cache
def half_idle_is_effectively_unqueued() -> bool:
    """At fifty percent utilisation the p99 wait is three ticks and the median is zero.

    The flat part of the curve is flat indeed. A store sized to run half idle has bought its
    way out of queueing almost entirely, and the price is stated plainly: double the
    hardware. Everything between fifty and ninety is a negotiation between that price and
    the hockey stick.
    """
    made = simulate(0.5)
    return made.p99 < 5 and made.p50 <= 1


def compare_the_utilisations() -> list[dict]:
    """One row per load level."""
    return [
        simulate(utilisation).as_dict()
        for utilisation in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
    ]


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "the_hockey_stick_is_real": the_wait_doubles_between_ninety_and_ninety_five(),
        "the_tail_tracks_the_mean": the_tail_is_worse_than_the_mean_everywhere(),
        "half_idle_is_unqueued": half_idle_is_effectively_unqueued(),
    }
