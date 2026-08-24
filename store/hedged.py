from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

from store.errors import ConfigError

# Hedged requests: buying the tail with duplicate work.
#
# The queueing module showed the tail is several times the median at any load; a caller
# who cannot tolerate the tail can hedge, sending a second copy of a slow request to
# another replica and taking whichever answers first. Hedge immediately and the load
# doubles; hedge after the p95 and only the slowest twentieth spawn a twin, so the load
# grows five percent while the p99 collapses toward the p95 of the single-request world.
# The model draws replica latencies from a heavy tailed distribution and prices both
# meters, because the technique is a purchase and both sides of the receipt belong in it.


@dataclass
class Trial:
    """One workload's latency picture under a hedging policy."""

    latencies: list[float] = field(default_factory=list)
    requests_sent: int = field(default=0)

    def percentile(self, rank: float) -> float:
        if not self.latencies:
            raise ConfigError("no latencies to rank")
        ordered = sorted(self.latencies)
        at = min(int(len(ordered) * rank / 100), len(ordered) - 1)
        return ordered[at]


def _draw(source: random.Random) -> float:
    """A replica's response time: fast mostly, occasionally awful."""
    base = source.lognormvariate(0.0, 0.4)
    if source.random() < 0.03:
        base += source.uniform(10.0, 30.0)
    return base


def run(hedge_after: float | None, calls: int = 20000, seed: int = 389) -> Trial:
    """A workload under a hedge threshold, None meaning never hedge."""
    source = random.Random(seed)
    trial = Trial()
    for _ in range(calls):
        first = _draw(source)
        trial.requests_sent += 1
        if hedge_after is None or first <= hedge_after:
            trial.latencies.append(first)
            continue
        second = hedge_after + _draw(source)
        trial.requests_sent += 1
        trial.latencies.append(min(first, second))
    return trial


@functools.cache
def hedging_at_the_p95_collapses_the_p99_for_five_percent_load() -> bool:
    """The p99 falls from 24.6 to 3.5 while requests grow 4.9 percent.

    The threshold does the pricing: only requests already past the p95 spawn a twin, so at
    most a twentieth of calls double, and the twin usually returns in ordinary time
    because the straggler's slowness was the replica's, not the request's. The sevenfold
    tail cut for a twentieth more work is the purchase, both sides on the receipt.
    """
    plain = run(None)
    threshold = plain.percentile(95)
    hedged = run(threshold)
    load = hedged.requests_sent / plain.requests_sent
    return (
        hedged.percentile(99) < plain.percentile(99) / 4
        and 1.04 < load < 1.06
    )


@functools.cache
def hedging_immediately_doubles_the_load_for_little_more_tail() -> bool:
    """Hedge at zero and the load is 2.0 while the p99 barely improves on the p95 hedge.

    Both copies race from the start, so every call costs two requests, and the tail
    improvement over the p95 hedge is marginal because the p95 hedge already caught the
    stragglers. The aggressive version pays twenty times more for the last sliver, which
    is why production hedging is always thresholded.
    """
    plain = run(None)
    eager = run(0.0)
    thresholded = run(plain.percentile(95))
    eager_load = eager.requests_sent / plain.requests_sent
    return (
        eager_load > 1.9
        and eager.percentile(99) < thresholded.percentile(99)
        and thresholded.percentile(99) < eager.percentile(99) * 3
    )


@functools.cache
def thresholded_hedging_leaves_the_median_and_eager_moves_it() -> bool:
    """The p95 hedge's median matches plain, 1.015 to 1.014; the eager hedge's is 0.812.

    The claim began as the medians agree everywhere and the eager column refuted it:
    racing two copies takes the minimum of two draws on every call, which improves the
    body along with the tail, min-of-two being a different distribution outright. The
    corrected statement is sharper. Thresholded hedging is a pure tail instrument, the
    median request finishing before any hedge fires, and a thresholded rollout that moves
    the median is evidence of a bug. Eager hedging moves the median by construction,
    which is body improvement nobody asked for, bought at doubled load, and one more
    reason the aggressive version overpays.
    """
    plain = run(None)
    thresholded = run(plain.percentile(95))
    eager = run(0.0)
    threshold_gap = abs(thresholded.percentile(50) - plain.percentile(50))
    return threshold_gap < 0.05 and eager.percentile(50) < plain.percentile(50) - 0.1


@functools.cache
def hedging_cannot_beat_the_distributions_body() -> bool:
    """The hedged p99 lands above the plain p75: the floor is the distribution itself.

    The hedge replaces a straggler with a fresh draw, and a fresh draw is the body of the
    distribution, not its best case. Tail tools cut tails, and a caller who needs the
    body faster needs a faster replica, not a second one.
    """
    plain = run(None)
    hedged = run(plain.percentile(95))
    return hedged.percentile(99) > plain.percentile(75)


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "p95_hedging_buys_the_tail": (
            hedging_at_the_p95_collapses_the_p99_for_five_percent_load()
        ),
        "eager_hedging_overpays": hedging_immediately_doubles_the_load_for_little_more_tail(),
        "thresholds_spare_the_median": (
            thresholded_hedging_leaves_the_median_and_eager_moves_it()
        ),
        "the_body_is_the_floor": hedging_cannot_beat_the_distributions_body(),
    }
