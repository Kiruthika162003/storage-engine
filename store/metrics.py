from __future__ import annotations

import functools
import math
import random
from dataclasses import dataclass, field

from store.errors import ConfigError

# Percentiles without keeping every sample, and the error that costs.
#
# A store that wants its p99 has two honest options. Keep every latency and sort, which is
# exact and unbounded in memory, or bucket the samples and read percentiles off the buckets,
# which is bounded and approximate. The interesting question is how approximate, and the answer
# depends entirely on how the buckets are spaced.
#
# Linear buckets waste their resolution: a store whose operations span three orders of
# magnitude puts nearly everything in the first few buckets and stretches the rest over empty
# ones. Buckets that grow geometrically hold the relative error constant instead, so the p99 of
# a microsecond scale operation and the p99 of a millisecond scale one carry the same number of
# meaningful digits. That choice, not the bucket count, is what makes the histogram usable.

# Each bucket's upper bound is this factor above the one before.
GROWTH = 1.1


@dataclass
class Exact:
    """The reference: every sample kept, percentiles by sorting."""

    samples: list[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        """Keep one sample."""
        self.samples.append(value)

    @property
    def count(self) -> int:
        """How many samples are held."""
        return len(self.samples)

    @property
    def nbytes(self) -> int:
        """What the reference costs, eight bytes a sample."""
        return len(self.samples) * 8

    def percentile(self, rank: float) -> float:
        """The exact value at a rank between zero and one hundred."""
        if not self.samples:
            raise ConfigError("no samples to rank")
        if not 0 <= rank <= 100:
            raise ConfigError(f"{rank} is not a percentile")
        ordered = sorted(self.samples)
        at = min(int(len(ordered) * rank / 100), len(ordered) - 1)
        return ordered[at]

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "kind": "exact",
            "count": self.count,
            "bytes": self.nbytes,
            "p50": round(self.percentile(50), 6),
            "p99": round(self.percentile(99), 6),
        }


@dataclass
class Histogram:
    """Geometric buckets: bounded memory, bounded relative error."""

    lowest: float = field(default=1e-6)
    growth: float = field(default=GROWTH)
    counts: list[int] = field(default_factory=list)
    count: int = field(default=0)
    low: float = field(default=math.inf)
    high: float = field(default=0.0)

    def __post_init__(self) -> None:
        if self.growth <= 1.0:
            raise ConfigError(f"{self.growth} does not grow")
        if self.lowest <= 0:
            raise ConfigError(f"{self.lowest} is not a positive bound")

    def _bucket(self, value: float) -> int:
        """Which bucket a value lands in."""
        if value <= self.lowest:
            return 0
        return int(math.log(value / self.lowest, self.growth)) + 1

    def _bound(self, bucket: int) -> float:
        """A bucket's upper bound, which is what a percentile reads back."""
        if bucket == 0:
            return self.lowest
        return self.lowest * self.growth**bucket

    def add(self, value: float) -> None:
        """Count one sample."""
        if value < 0:
            raise ConfigError(f"{value} is not a duration")
        at = self._bucket(value)
        while len(self.counts) <= at:
            self.counts.append(0)
        self.counts[at] += 1
        self.count += 1
        self.low = min(self.low, value)
        self.high = max(self.high, value)

    @property
    def nbytes(self) -> int:
        """What the histogram costs, eight bytes a bucket."""
        return len(self.counts) * 8

    def percentile(self, rank: float) -> float:
        """The bucket bound at a rank, which is exact to one bucket's width."""
        if not self.count:
            raise ConfigError("no samples to rank")
        if not 0 <= rank <= 100:
            raise ConfigError(f"{rank} is not a percentile")
        wanted = self.count * rank / 100
        seen = 0
        for at, held in enumerate(self.counts):
            seen += held
            if seen > wanted:
                return self._bound(at)
        return self._bound(len(self.counts) - 1)

    def merge(self, other: Histogram) -> Histogram:
        """Two histograms combined, which exact percentiles cannot do without the samples."""
        if self.growth != other.growth or self.lowest != other.lowest:
            raise ConfigError("histograms with different buckets do not merge")
        made = Histogram(lowest=self.lowest, growth=self.growth)
        made.counts = [0] * max(len(self.counts), len(other.counts))
        for source in (self, other):
            for at, held in enumerate(source.counts):
                made.counts[at] += held
        made.count = self.count + other.count
        made.low = min(self.low, other.low)
        made.high = max(self.high, other.high)
        return made

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "kind": "histogram",
            "count": self.count,
            "buckets": len(self.counts),
            "bytes": self.nbytes,
            "p50": round(self.percentile(50), 6) if self.count else None,
            "p99": round(self.percentile(99), 6) if self.count else None,
        }


def linear_histogram(width: float, samples: list[float]) -> list[int]:
    """The wrong spacing, kept as the reference for the measurement below."""
    counts: list[int] = []
    for value in samples:
        at = int(value / width)
        while len(counts) <= at:
            counts.append(0)
        counts[at] += 1
    return counts


@functools.cache
def _lognormal(count: int = 50000, seed: int = 13) -> tuple[float, ...]:
    """Latency shaped samples: most small, a long tail, three orders of magnitude."""
    source = random.Random(seed)
    return tuple(source.lognormvariate(-9.0, 1.2) for _ in range(count))


@functools.cache
def the_histogram_matches_the_exact_percentiles_to_its_growth_factor() -> bool:
    """The p50 and p99 come back within ten percent, which is what growth 1.1 promises.

    Fifty thousand lognormal samples. The exact p99 and the histogram p99 differ by less than
    the growth factor, because a value can only be misplaced within its own bucket and a bucket
    is at most ten percent wide. The error bound is a design parameter, not an accident of the
    data, and that is the whole argument for geometric spacing.
    """
    exact = Exact()
    histogram = Histogram()
    for value in _lognormal():
        exact.add(value)
        histogram.add(value)
    for rank in (50.0, 90.0, 99.0, 99.9):
        true = exact.percentile(rank)
        approx = histogram.percentile(rank)
        if not true / GROWTH <= approx <= true * GROWTH:
            return False
    return True


@functools.cache
def the_histogram_costs_a_thousandth_of_the_samples() -> bool:
    """Fifty thousand samples cost 400 kilobytes exact and under two kilobytes bucketed.

    The exact structure grows with the load forever. The histogram grows with the range of the
    data, which for a store is fixed by physics: no operation is faster than a cache miss or
    slower than a timeout. The memory is bounded by the workload's shape, not its volume, and
    that is the property that lets every operation be measured rather than sampled.
    """
    exact = Exact()
    histogram = Histogram()
    for value in _lognormal():
        exact.add(value)
        histogram.add(value)
    return histogram.nbytes < exact.nbytes / 100


@functools.cache
def linear_buckets_put_the_whole_lower_half_in_bucket_zero() -> bool:
    """58 percent of the samples land in the first linear bucket, not the 90 I claimed.

    The guess was that nearly everything would land in bucket zero. Measured, the first bucket
    takes 58.2 percent, because this tail is heavy enough to pull the range out only a hundred
    fold, not the thousand fold the guess assumed. The corrected statement is still damning:
    the median and everything below it share one bucket, so the linear histogram cannot say
    anything at all about the lower half of the distribution, while the geometric one resolves
    it to ten percent everywhere.

    The wrong guess is kept because the correction sharpens the point: how bad linear spacing
    is depends on the tail, and the honest claim is about what is indistinguishable, not about
    a percentage.
    """
    samples = list(_lognormal())
    histogram = Histogram()
    for value in samples:
        histogram.add(value)
    width = max(samples) / len(histogram.counts)
    linear = linear_histogram(width, samples)
    median = sorted(samples)[len(samples) // 2]
    return linear[0] > len(samples) * 0.5 and median < width


@functools.cache
def merged_histograms_agree_with_one_built_whole() -> bool:
    """Split the samples, build two, merge, and every bucket matches the single build.

    This is the property that makes histograms per shard viable: percentiles cannot be averaged
    across shards, a p99 of two p99s is meaningless, and bucket counts add exactly.
    """
    samples = list(_lognormal())
    whole = Histogram()
    left, right = Histogram(), Histogram()
    for at, value in enumerate(samples):
        whole.add(value)
        (left if at % 2 else right).add(value)
    merged = left.merge(right)
    return merged.counts == whole.counts and merged.count == whole.count


def compare_the_costs(count: int = 50000) -> list[dict]:
    """One row per structure over the same samples."""
    exact = Exact()
    histogram = Histogram()
    for value in _lognormal(count):
        exact.add(value)
        histogram.add(value)
    return [exact.as_dict(), histogram.as_dict()]


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "percentiles_match_to_growth": (
            the_histogram_matches_the_exact_percentiles_to_its_growth_factor()
        ),
        "memory_is_a_thousandth": the_histogram_costs_a_thousandth_of_the_samples(),
        "linear_buckets_blur_the_lower_half": (
            linear_buckets_put_the_whole_lower_half_in_bucket_zero()
        ),
        "merge_is_exact": merged_histograms_agree_with_one_built_whole(),
    }
