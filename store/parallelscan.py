from __future__ import annotations

import functools
import random
from dataclasses import dataclass

from store.errors import ConfigError

# Splitting one scan across workers, and why equal ranges are unequal work.
#
# A full scan parallelises by cutting the keyspace into ranges and giving each worker one.
# The obvious cut, equal spans of the key space, balances only when the data is uniform,
# and data never is: a tenant prefix, a time cluster, a hot shard, and one worker carries
# the day while the rest idle. The remedy the sstable already paid for is the sample: cut
# at quantiles of the actual keys, taken from the index or a reservoir, and every worker
# gets the same row count regardless of the distribution's shape. The imbalance meter is
# the makespan ratio, slowest worker over mean, which is the number that decides when the
# job finishes.


@dataclass(frozen=True)
class Split:
    """One worker's range and its workload."""

    worker: int
    rows: int


def cut_by_keyspace(keys: list[int], workers: int, span: int) -> list[Split]:
    """Equal spans of the key space."""
    if workers < 1:
        raise ConfigError(f"{workers} is not a worker count")
    width = span // workers
    counts = [0] * workers
    for key in keys:
        at = min(key // max(width, 1), workers - 1)
        counts[at] += 1
    return [Split(worker=at, rows=count) for at, count in enumerate(counts)]


def cut_by_quantiles(keys: list[int], workers: int) -> list[Split]:
    """Cuts at the observed quantiles."""
    if workers < 1:
        raise ConfigError(f"{workers} is not a worker count")
    ordered = sorted(keys)
    bounds = [
        ordered[min(len(ordered) - 1, (at * len(ordered)) // workers)]
        for at in range(1, workers)
    ]
    counts = [0] * workers
    for key in keys:
        at = 0
        while at < len(bounds) and key >= bounds[at]:
            at += 1
        counts[at] += 1
    return [Split(worker=at, rows=count) for at, count in enumerate(counts)]


def makespan_ratio(splits: list[Split]) -> float:
    """Slowest worker over the mean: one is perfect, workers is worst."""
    rows = [split.rows for split in splits]
    mean = sum(rows) / len(rows)
    if mean == 0:
        return 1.0
    return round(max(rows) / mean, 3)


SPAN = 1 << 30


@functools.cache
def _clustered(count: int = 40000, seed: int = 307) -> tuple[int, ...]:
    """Keys clustered the way tenants and time cluster them: 90 percent in 5 percent."""
    source = random.Random(seed)
    made = []
    for _ in range(count):
        if source.random() < 0.9:
            made.append(source.randrange(SPAN // 20))
        else:
            made.append(source.randrange(SPAN))
    return tuple(made)


@functools.cache
def _uniform(count: int = 40000, seed: int = 311) -> tuple[int, ...]:
    """The distribution the naive cut silently assumes."""
    source = random.Random(seed)
    return tuple(source.randrange(SPAN) for _ in range(count))


@functools.cache
def keyspace_cuts_balance_only_uniform_data() -> bool:
    """Uniform keys: makespan 1.05. Clustered keys: 7.3 of a possible eight.

    The same eight workers, the same cut rule, and the clustered workload lands 90 percent
    of its rows on worker zero, so the parallel scan runs at nearly single-worker speed
    while seven workers idle. The rule did not get worse; the assumption it encodes did,
    and the assumption was never stated anywhere it could be reviewed.
    """
    even = makespan_ratio(cut_by_keyspace(list(_uniform()), 8, SPAN))
    skewed = makespan_ratio(cut_by_keyspace(list(_clustered()), 8, SPAN))
    return even < 1.1 and skewed > 6.0


@functools.cache
def quantile_cuts_balance_anything() -> bool:
    """The same clustered keys under quantile cuts: makespan 1.0 to the third decimal.

    Cut where the data is rather than where the space is, and the shape stops mattering.
    The quantiles came from sorting the keys here; a real engine reads them from the
    sstable index for free, which is one more thing sorted storage quietly pays for.
    """
    skewed = makespan_ratio(cut_by_quantiles(list(_clustered()), 8))
    even = makespan_ratio(cut_by_quantiles(list(_uniform()), 8))
    return skewed <= 1.01 and even <= 1.01


@functools.cache
def both_cuts_assign_every_row_exactly_once() -> bool:
    """Row counts across workers sum to the input for both rules, both shapes.

    The conservation check that makes the ratios meaningful: an unbalanced cut and a lossy
    cut can produce the same pretty makespan.
    """
    for keys in (list(_uniform()), list(_clustered())):
        for splits in (cut_by_keyspace(keys, 8, SPAN), cut_by_quantiles(keys, 8)):
            if sum(split.rows for split in splits) != len(keys):
                return False
    return True


@functools.cache
def eight_times_the_workers_buys_three_under_the_bad_cut() -> bool:
    """Keyspace cut: 8x workers move the slowest from 36,508 rows to 11,449, 3.2x.

    I expected the finish line to barely move; it moved sublinearly instead, and the
    arithmetic explains both the movement and the shortfall. The cluster covers a twentieth
    of the space, so at 64 workers it spans about three worker widths and gets subdivided
    that far and no further: the workers that land inside the cluster gain, the dozens
    outside it split idle space. Quantile cuts convert the same 8x into exactly 8x, 5,000
    to 625. Parallelism is a multiplier on the cut rule, and the bad cut spends most of the
    factor on nothing.
    """
    keys = list(_clustered())
    eight = max(split.rows for split in cut_by_keyspace(keys, 8, SPAN))
    sixty_four = max(split.rows for split in cut_by_keyspace(keys, 64, SPAN))
    q_eight = max(split.rows for split in cut_by_quantiles(keys, 8))
    q_sixty_four = max(split.rows for split in cut_by_quantiles(keys, 64))
    keyspace_gain = eight / sixty_four
    quantile_gain = q_eight / q_sixty_four
    return 2.0 < keyspace_gain < 5.0 and quantile_gain == 8.0


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "keyspace_cuts_need_uniformity": keyspace_cuts_balance_only_uniform_data(),
        "quantile_cuts_balance_anything": quantile_cuts_balance_anything(),
        "rows_are_conserved": both_cuts_assign_every_row_exactly_once(),
        "workers_multiply_the_cut_rule": eight_times_the_workers_buys_three_under_the_bad_cut(),
    }
