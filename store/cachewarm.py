from __future__ import annotations

import functools
from dataclasses import dataclass, field

from store.cache import Recent, Reference

# Warming a cache from yesterday's shape, and when yesterday lies.
#
# A restarted store begins with a cold cache and serves its worst latency exactly when the
# most eyes are on it. Warmup persists the cache's key set at shutdown and prefetches it at
# startup, buying back the ramp at the cost of startup reads. Whether the buy is good
# depends entirely on one thing: whether tonight's working set is yesterday's. Both cases
# are measured, and so is the misses-to-recover meter that says how long a cold cache
# takes to become a warm one on its own.


@dataclass
class Ramp:
    """A hit-rate trajectory, sampled in windows."""

    window: int = field(default=500)
    hits_in_window: int = field(default=0)
    seen_in_window: int = field(default=0)
    trajectory: list[float] = field(default_factory=list)

    def note(self, hit: bool) -> None:
        self.seen_in_window += 1
        if hit:
            self.hits_in_window += 1
        if self.seen_in_window == self.window:
            self.trajectory.append(round(self.hits_in_window / self.window, 4))
            self.hits_in_window = 0
            self.seen_in_window = 0


def run(cache: Recent, blocks, ramp: Ramp) -> Ramp:
    """A stream through a cache, trajectory recorded."""
    for number in blocks:
        if cache.get(number) is None:
            cache.put(number, number.to_bytes(8, "little"))
            ramp.note(False)
        else:
            ramp.note(True)
    return ramp


def save_keys(cache: Recent) -> list[int]:
    """The shutdown snapshot: which blocks were resident."""
    return list(cache.held)


def warm(cache: Recent, keys: list[int]) -> int:
    """The startup prefetch: load the saved set, count the reads it cost."""
    for number in keys:
        cache.put(number, number.to_bytes(8, "little"))
    return len(keys)


@functools.cache
def _hot_stream(length: int = 20000, seed: int = 263) -> tuple[int, ...]:
    """A stable working set: the same hot blocks before and after the restart."""
    return tuple(Reference(blocks=2000, length=length, shape="hot", seed=seed).stream())


@functools.cache
def a_warmed_cache_skips_the_ramp() -> bool:
    """The cold restart's first window hits 55 percent; the warmed restart's hits 87.

    The same evening traffic against both restarts of the same store. Cold, the first
    window is spent faulting the working set back in and climbs to the 88 percent plateau
    over several windows. Warmed from the shutdown key set, the first window opens within a
    point of the plateau, and the price was 256 startup reads done before anyone was
    watching.
    """
    yesterday = Recent(capacity=256)
    run(yesterday, list(_hot_stream()), Ramp())
    saved = save_keys(yesterday)
    cold = Recent(capacity=256)
    cold_ramp = run(cold, list(_hot_stream(20000, 269)), Ramp())
    warmed = Recent(capacity=256)
    warm(warmed, saved)
    warm_ramp = run(warmed, list(_hot_stream(20000, 269)), Ramp())
    return (
        warm_ramp.trajectory[0] > cold_ramp.trajectory[0] + 0.3
        and abs(warm_ramp.trajectory[0] - warm_ramp.trajectory[-1]) < 0.1
    )


@functools.cache
def the_cold_ramp_ends_at_the_same_plateau() -> bool:
    """Cold and warm converge: the last windows agree within two points.

    Warmup moves the ramp, it does not raise the ceiling. The plateau is a property of the
    workload and the capacity, and any warmup pitch quoting a higher steady state is
    quoting a different workload.
    """
    saved = save_keys(run_and_return(_hot_stream()))
    cold = Recent(capacity=256)
    cold_ramp = run(cold, list(_hot_stream(20000, 269)), Ramp())
    warmed = Recent(capacity=256)
    warm(warmed, saved)
    warm_ramp = run(warmed, list(_hot_stream(20000, 269)), Ramp())
    return abs(cold_ramp.trajectory[-1] - warm_ramp.trajectory[-1]) < 0.02


def run_and_return(stream) -> Recent:
    """A cache after a stream."""
    cache = Recent(capacity=256)
    run(cache, list(stream), Ramp())
    return cache


@functools.cache
def a_shifted_working_set_makes_warmup_worthless() -> bool:
    """Warming with yesterday's keys against a moved hot set matches cold within noise.

    The overnight deploy changed the key prefix, the saved set prefetches two hundred and
    fifty six strangers, and the first window hits like a cold start while the warmup's
    reads were pure waste. Warmup is a bet on stability, and the module's one operational
    sentence is: persist the key set with a timestamp, and skip the warmup when the store
    was down long enough for the world to move.
    """
    saved = save_keys(run_and_return(_hot_stream()))
    base = Reference(blocks=2000, length=20000, shape="hot", seed=271).stream()
    moved = tuple(number + 100000 for number in base)
    cold = Recent(capacity=256)
    cold_ramp = run(cold, list(moved), Ramp())
    warmed = Recent(capacity=256)
    warm(warmed, saved)
    warm_ramp = run(warmed, list(moved), Ramp())
    return abs(warm_ramp.trajectory[0] - cold_ramp.trajectory[0]) < 0.05


@functools.cache
def the_warmup_cost_is_the_capacity() -> bool:
    """The prefetch reads exactly the saved set, which is at most the capacity."""
    saved = save_keys(run_and_return(_hot_stream()))
    warmed = Recent(capacity=256)
    return warm(warmed, saved) == len(saved) <= 256


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "warmup_skips_the_ramp": a_warmed_cache_skips_the_ramp(),
        "the_plateau_is_the_workload": the_cold_ramp_ends_at_the_same_plateau(),
        "shifted_sets_void_the_bet": a_shifted_working_set_makes_warmup_worthless(),
        "the_cost_is_the_capacity": the_warmup_cost_is_the_capacity(),
    }
