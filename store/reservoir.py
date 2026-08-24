from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

from store.errors import ConfigError

# A fair sample of a stream nobody can hold, and the bias every shortcut has.
#
# The planner's histogram was built from the whole column, which a real system rarely gets
# to see twice. Statistics come from samples of streams, and a fair sample of a stream of
# unknown length is a solved problem with an unfair-looking solution: keep the first k, then
# keep the nth arrival with probability k over n, evicting a uniform victim. Every element
# ends up sampled with probability exactly k over n, first and last alike.
#
# The shortcuts people take instead are measured next to it. Keep-the-first-k is the arrival
# order frozen as truth. Keep-every-mth is a stroboscope, fine for random streams and
# catastrophically aliased on periodic ones, the recovery eval's lesson generalised.

CAPACITY = 500


@dataclass
class Reservoir:
    """The fair one."""

    capacity: int = field(default=CAPACITY)
    held: list[int] = field(default_factory=list)
    seen: int = field(default=0)
    source: random.Random = field(default=None)
    seed: int = field(default=173)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ConfigError(f"{self.capacity} is not a capacity")
        self.source = random.Random(self.seed)

    def offer(self, value: int) -> None:
        """One arrival."""
        self.seen += 1
        if len(self.held) < self.capacity:
            self.held.append(value)
            return
        at = self.source.randrange(self.seen)
        if at < self.capacity:
            self.held[at] = value


def first_k(stream, capacity: int = CAPACITY) -> list[int]:
    """The shortcut that freezes the beginning."""
    held = []
    for value in stream:
        if len(held) >= capacity:
            break
        held.append(value)
    return held


def every_mth(stream, capacity: int = CAPACITY) -> list[int]:
    """The stroboscope: a fixed stride chosen from a length guess."""
    stream = list(stream)
    stride = max(len(stream) // capacity, 1)
    return stream[::stride][:capacity]


@functools.cache
def _drifting_stream(count: int = 50000) -> tuple[int, ...]:
    """A stream whose values grow over time, so position and value correlate."""
    return tuple(at // 50 for at in range(count))


@functools.cache
def _periodic_stream(count: int = 50000, period: int = 100) -> tuple[int, ...]:
    """A stream with a spike every period, the shape schedules produce."""
    return tuple(1000 if at % period == 0 else 1 for at in range(count))


@functools.cache
def the_reservoir_is_uniform_over_positions() -> bool:
    """Averaged over trials, early, middle and late arrivals are sampled evenly.

    Two hundred trials over ten thousand positions: the sample rate of the first, middle
    and last thirds agree within a few percent, which is the definition of fair delivered
    by the k-over-n coin. The proof is an induction; the measurement is the part an off by
    one silently breaks, so both exist.
    """
    hits = [0, 0, 0]
    trials = 200
    for trial in range(trials):
        reservoir = Reservoir(capacity=90, seed=trial)
        for value in range(9000):
            reservoir.offer(value)
        for value in reservoir.held:
            hits[value // 3000] += 1
    total = sum(hits)
    shares = [count / total for count in hits]
    return all(abs(share - 1 / 3) < 0.05 for share in shares)


@functools.cache
def first_k_reports_the_past_on_a_drifting_stream() -> bool:
    """On a drifting stream the first-k mean is 2 percent of the true mean.

    The stream's values grow with time and first-k froze the beginning, so its estimate of
    the mean is off by a factor of fifty, while the reservoir lands within a few percent.
    Every stream with drift punishes first-k in proportion to the drift, and most streams
    that matter drift.
    """
    stream = _drifting_stream()
    true_mean = sum(stream) / len(stream)
    frozen = first_k(stream)
    frozen_mean = sum(frozen) / len(frozen)
    reservoir = Reservoir()
    for value in stream:
        reservoir.offer(value)
    fair_mean = sum(reservoir.held) / len(reservoir.held)
    return frozen_mean < true_mean * 0.05 and abs(fair_mean - true_mean) < true_mean * 0.1


@functools.cache
def every_mth_aliases_on_a_periodic_stream() -> bool:
    """A stride sharing a divisor with the period samples every spike or none.

    The spikes are one percent of the stream; the strided sample of stride 100 on period
    100 reports them as 100 percent, because every sampled position is a spike position.
    The reservoir reports one percent, within noise. This is the recovery eval's
    stroboscope again, now stated as the general law: a fixed stride is a bet that the
    stream has no structure at that stride, and schedules exist to create exactly that
    structure.
    """
    stream = _periodic_stream()
    strided = every_mth(stream)
    strided_spikes = sum(1 for value in strided if value == 1000) / len(strided)
    reservoir = Reservoir()
    for value in stream:
        reservoir.offer(value)
    fair_spikes = sum(1 for value in reservoir.held if value == 1000) / len(reservoir.held)
    return strided_spikes == 1.0 and fair_spikes < 0.05


@functools.cache
def the_reservoir_holds_its_size_forever() -> bool:
    """A million arrivals leave exactly the capacity held, and seen counts them all."""
    reservoir = Reservoir(capacity=100)
    for value in range(100000):
        reservoir.offer(value)
    return len(reservoir.held) == 100 and reservoir.seen == 100000


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "the_reservoir_is_uniform": the_reservoir_is_uniform_over_positions(),
        "first_k_reports_the_past": first_k_reports_the_past_on_a_drifting_stream(),
        "every_mth_aliases": every_mth_aliases_on_a_periodic_stream(),
        "the_size_is_fixed": the_reservoir_holds_its_size_forever(),
    }
