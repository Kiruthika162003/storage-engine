from __future__ import annotations

import functools
import random
from dataclasses import dataclass

from store.engine import Store

# How the costs move as the store grows, which is the question sizing actually asks.
#
# A benchmark at one size answers nothing about another size unless the scaling shape is
# known. These runs grow the store across a factor of sixteen and watch three quantities: how
# many files a read considers, how many times a record is rewritten, and how much of the store
# is stale. The claims are about the shapes, linear or logarithmic or flat, because the shape
# is what survives a change of machine.


@dataclass(frozen=True)
class Size:
    """One store size's costs."""

    writes: int
    tables: int
    flushes: int
    folds: int
    filter_skips_per_miss: float
    stale: float

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "writes": self.writes,
            "tables": self.tables,
            "flushes": self.flushes,
            "folds": self.folds,
            "filter_skips_per_miss": self.filter_skips_per_miss,
            "stale": self.stale,
        }


@functools.cache
def measure(writes: int, seed: int = 51) -> Size:
    """One store grown to a size, with a probe pass at the end."""
    source = random.Random(seed)
    store = Store(flush_at=500, fold_at=4)
    keys = max(writes // 2, 1)
    for _ in range(writes):
        store.put(f"k{source.randrange(keys):08d}".encode(), source.randbytes(16))
    live = len(store.items())
    held = len(store.memtable.records()) + sum(len(t.records) for t in store.tables)
    probes = 500
    before = store.filter_skips
    for at in range(probes):
        store.get(f"absent:{at:06d}".encode())
    return Size(
        writes=writes,
        tables=len(store.tables),
        flushes=store.flushes,
        folds=store.folds,
        filter_skips_per_miss=round((store.filter_skips - before) / probes, 3),
        stale=round((held - live) / max(held, 1), 4),
    )


SIZES = (2000, 4000, 8000, 16000, 32000)


@functools.cache
def flushes_grow_faster_than_the_volume_because_dedup_fades() -> bool:
    """Sixteen times the writes gave 31.5 times the flushes, and the excess is the memtable.

    The expectation was proportional: five hundred records a flush, forever. Measured, 2,000
    writes flush twice, a thousand writes a flush, while 32,000 writes flush 63 times, 508 a
    flush. The memtable holds one entry per key, so at the small size, where the keyspace is
    1,000 keys, half the writes are overwrites that grow the memtable not at all, and at the
    large size, with 16,000 keys, almost every write is a fresh key.

    So the flush count is not the write volume. It is the write volume minus the overwrites
    the memtable absorbed, and any capacity plan that multiplies writes by a flush constant is
    assuming a key distribution without saying so.

    The table count is bounded at every size, 1 to 3 files, which is the part that held: a
    read considers a fixed number of files no matter how much has been written.
    """
    small = measure(2000)
    large = measure(32000)
    flush_ratio = large.flushes / max(small.flushes, 1)
    return flush_ratio > 20 and large.tables <= small.tables + 3


@functools.cache
def folds_converge_to_a_third_of_flushes_not_a_quarter() -> bool:
    """The threshold is four tables and the asymptotic rate is one fold per three flushes.

    I expected volume over four. The measured ratios climb 0.167, 0.267, 0.323 and land at
    0.318, approaching a third from below, because a fold consumes four tables and produces
    one, and that one counts toward the next threshold: only three fresh flushes are needed
    to trigger again. The steady state of a fold-at-N policy is one fold per N minus one
    flushes, and the threshold as written overstates the interval by a third.

    The approach is not monotonic to the last digit, 0.3226 then 0.3175, because a fold of
    nothing but dead versions can install no output file and the next fold then needs the
    full four. The smallest size never folds at all, two tables never reaching four, which is
    its own reminder that asymptotic claims say nothing about small stores.
    """
    ratios = []
    for writes in SIZES[1:]:
        made = measure(writes)
        ratios.append(made.folds / made.flushes)
    return all(0.28 <= ratio <= 0.36 for ratio in ratios[-2:]) and measure(2000).folds == 0


@functools.cache
def a_miss_stays_cheap_at_every_size() -> bool:
    """Filter skips per absent key stay near the table count, which stays bounded.

    A miss costs one filter query per candidate table. Because the table count is bounded by
    folding, the miss cost is flat across a sixteenfold growth, which is the property that
    makes the store's read latency a function of its shape rather than its history.
    """
    costs = [measure(writes).filter_skips_per_miss for writes in SIZES]
    return max(costs) <= 4.0


def table() -> list[dict]:
    """One row per size."""
    return [measure(writes).as_dict() for writes in SIZES]


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "flushes_outrun_the_volume": flushes_grow_faster_than_the_volume_because_dedup_fades(),
        "folds_converge_to_a_third": folds_converge_to_a_third_of_flushes_not_a_quarter(),
        "misses_stay_cheap": a_miss_stays_cheap_at_every_size(),
    }
