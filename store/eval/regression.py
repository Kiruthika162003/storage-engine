from __future__ import annotations

import functools

from store.amplification import btree_point
from store.compaction import compare_the_policies
from store.eval.latency import simulate
from store.eval.scaling import measure
from store.hll import _sketched
from store.metrics import Histogram, _lognormal
from store.sstable import _table, probe

# Baselines with tolerances, so a drift is a diff and not a feeling.
#
# The ledger answers whether every claim still holds, which is binary. This module holds the
# numbers themselves: the measured quantities the package's story is built on, each with the
# tolerance inside which a change is noise and outside which it is news. A refactor that
# shifts levelled write amplification from 7.165 to 7.9 flips no claim, and it is exactly the
# kind of change someone should have to look at. The baselines are code, reviewed like code,
# and the check names each drifting quantity with its expected and observed values.

BASELINES = {
    "compaction.levelled_amplification": (7.165, 0.15),
    "compaction.tiered_amplification": (2.434, 0.15),
    "compaction.levelled_read_cost": (2.981, 0.15),
    "compaction.tiered_read_cost": (3.981, 0.15),
    "amplification.btree_write": (43.969, 0.15),
    "amplification.btree_space": (1.01, 0.05),
    "sstable.filter_reads_per_miss": (0.0069, 0.5),
    "metrics.histogram_bytes": (824.0, 0.1),
    "latency.mean_wait_at_ninety": (4.583, 0.2),
    "latency.mean_wait_at_ninety_nine": (40.4, 0.25),
    "scaling.fold_to_flush_ratio": (0.317, 0.1),
    "hll.error_at_half_million": (0.0082, 1.0),
}


@functools.cache
def observe() -> dict[str, float]:
    """Measure every baselined quantity now."""
    levelled, tiered = compare_the_policies()
    tree = btree_point()
    absent = [f"gone:{one:08d}".encode() for one in range(10000)]
    filtered = probe(_table(20000), absent)
    histogram = Histogram()
    for value in _lognormal():
        histogram.add(value)
    grown = measure(32000)
    sketch = _sketched(500000)
    return {
        "compaction.levelled_amplification": levelled["amplification"],
        "compaction.tiered_amplification": tiered["amplification"],
        "compaction.levelled_read_cost": levelled["read_cost"],
        "compaction.tiered_read_cost": tiered["read_cost"],
        "amplification.btree_write": tree.write,
        "amplification.btree_space": tree.space,
        "sstable.filter_reads_per_miss": filtered.reads_per_miss,
        "metrics.histogram_bytes": float(histogram.nbytes),
        "latency.mean_wait_at_ninety": simulate(0.9).mean_wait,
        "latency.mean_wait_at_ninety_nine": simulate(0.99).mean_wait,
        "scaling.fold_to_flush_ratio": grown.folds / grown.flushes,
        "hll.error_at_half_million": abs(sketch.estimate() - 500000) / 500000,
    }


def drifts() -> list[dict]:
    """Every quantity outside its tolerance, with the numbers side by side."""
    found = []
    observed = observe()
    for name, (expected, tolerance) in BASELINES.items():
        got = observed[name]
        relative = abs(got - expected) / max(abs(expected), 1e-12)
        if relative > tolerance:
            found.append(
                {
                    "quantity": name,
                    "expected": expected,
                    "observed": round(got, 6),
                    "relative_drift": round(relative, 4),
                    "tolerance": tolerance,
                }
            )
    return found


def report() -> dict:
    """The regression check in one mapping."""
    found = drifts()
    return {
        "baselines": len(BASELINES),
        "drifting": len(found),
        "clean": not found,
        "details": found,
    }
