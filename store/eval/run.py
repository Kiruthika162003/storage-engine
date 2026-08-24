from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass, field

from store.engine import Store
from store.eval.workload import MIXES, stream

# One harness that drives a store through a workload and reads every meter at the end.
#
# The measurements are counts, not clock times, for the same reason as everywhere else in the
# package: a count is reproducible on any machine and a nanosecond is a fact about the laptop.
# Where a real benchmark would report microseconds per get, this reports memtable probes, table
# probes, filter skips and records merged, which are the quantities the microseconds are made
# of.


@dataclass
class Meter:
    """What one workload run cost, in the engine's own units."""

    mix: str
    operations: int = field(default=0)
    gets: int = field(default=0)
    puts: int = field(default=0)
    deletes: int = field(default=0)
    scans: int = field(default=0)
    scan_records: int = field(default=0)
    flushes: int = field(default=0)
    folds: int = field(default=0)
    filter_skips: int = field(default=0)
    tables_after: int = field(default=0)
    hits: int = field(default=0)
    misses: int = field(default=0)

    @property
    def hit_rate(self) -> float:
        """What fraction of gets found a value."""
        return round(self.hits / max(self.gets, 1), 4)

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "mix": self.mix,
            "operations": self.operations,
            "gets": self.gets,
            "puts": self.puts,
            "deletes": self.deletes,
            "scans": self.scans,
            "hit_rate": self.hit_rate,
            "flushes": self.flushes,
            "folds": self.folds,
            "filter_skips": self.filter_skips,
            "tables_after": self.tables_after,
        }


def drive(store: Store, operations) -> Meter:
    """Run a stream against a store and account for it."""
    meter = Meter(mix="")
    for operation in operations:
        meter.operations += 1
        if operation.kind == "get":
            meter.gets += 1
            if store.get(operation.key) is not None:
                meter.hits += 1
            else:
                meter.misses += 1
        elif operation.kind == "put":
            meter.puts += 1
            store.put(operation.key, operation.value)
        elif operation.kind == "delete":
            meter.deletes += 1
            store.delete(operation.key)
        else:
            meter.scans += 1
            found = list(itertools.islice(store.scan(operation.key), operation.length))
            meter.scan_records += len(found)
    meter.flushes = store.flushes
    meter.folds = store.folds
    meter.filter_skips = store.filter_skips
    meter.tables_after = len(store.tables)
    return meter


@functools.cache
def run(mix: str, flush_at: int = 500, fold_at: int = 4) -> Meter:
    """One named workload against a fresh store."""
    store = Store(flush_at=flush_at, fold_at=fold_at)
    meter = drive(store, stream(mix))
    meter.mix = mix
    return meter


def table() -> list[dict]:
    """Every mix, one row each."""
    return [run(mix.name).as_dict() for mix in MIXES]
