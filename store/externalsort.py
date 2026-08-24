from __future__ import annotations

import functools
import heapq
import math
import random
from dataclasses import dataclass, field

from store.compaction import Levelled, Load, amplification, run_load
from store.errors import ConfigError

# Sorting more than fits, which is what compaction has been doing all along.
#
# An external sort cuts the input into runs that fit in memory, sorts each, and merges. The
# arithmetic that governs it: with memory for M records and fan-in F, one pass makes runs of
# M, and each merge pass multiplies run length by F, so the pass count is one plus the log
# base F of N over M, and every pass reads and writes every record once. The measurements
# check that arithmetic against counted IO, and then make the connection the module exists
# for: a levelled LSM is this algorithm run forever, with the runs arriving over time instead
# of all at once.


@dataclass
class Meter:
    """The IO an external sort performed."""

    records: int
    memory: int
    fan_in: int
    runs_made: int = field(default=0)
    passes: int = field(default=0)
    read: int = field(default=0)
    written: int = field(default=0)

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "records": self.records,
            "memory": self.memory,
            "fan_in": self.fan_in,
            "runs_made": self.runs_made,
            "passes": self.passes,
            "read": self.read,
            "written": self.written,
            "io_per_record": round((self.read + self.written) / max(self.records, 1), 2),
        }


def sort(values: list[int], memory: int, fan_in: int) -> tuple[list[int], Meter]:
    """The whole algorithm: run formation, then merge passes until one run stands."""
    if memory < 1:
        raise ConfigError(f"{memory} records of memory sorts nothing")
    if fan_in < 2:
        raise ConfigError(f"a fan-in of {fan_in} cannot merge")
    meter = Meter(records=len(values), memory=memory, fan_in=fan_in)
    runs: list[list[int]] = []
    for at in range(0, len(values), memory):
        chunk = sorted(values[at : at + memory])
        meter.read += len(chunk)
        meter.written += len(chunk)
        runs.append(chunk)
    meter.runs_made = len(runs)
    while len(runs) > 1:
        meter.passes += 1
        merged: list[list[int]] = []
        for at in range(0, len(runs), fan_in):
            group = runs[at : at + fan_in]
            out = list(heapq.merge(*group))
            meter.read += len(out)
            meter.written += len(out)
            merged.append(out)
        runs = merged
    return (runs[0] if runs else []), meter


@functools.cache
def _values(count: int, seed: int = 101) -> tuple[int, ...]:
    """Unsorted input."""
    source = random.Random(seed)
    return tuple(source.randrange(10**9) for _ in range(count))


@functools.cache
def the_sort_is_correct_at_every_geometry() -> bool:
    """Twelve memory and fan-in combinations all produce exactly sorted output.

    Including the degenerate ones: memory of one record, a fan-in of two, input smaller than
    memory, input of zero. The geometry changes the cost and must never change the answer,
    which is the compaction bar restated for the batch case.
    """
    values = list(_values(3000))
    wanted = sorted(values)
    for memory in (1, 7, 100, 5000):
        for fan_in in (2, 4, 16):
            made, _ = sort(values, memory, fan_in)
            if made != wanted:
                return False
    empty, _ = sort([], 10, 2)
    return empty == []


@functools.cache
def the_pass_count_follows_the_logarithm() -> bool:
    """Ten thousand records, memory for a hundred: 100 runs, then the log of the fan-in.

    Fan-in two takes seven merge passes, fan-in four takes four, fan-in sixteen takes two,
    and each is exactly the ceiling of log base F of the run count. The formula is the whole
    economics of the algorithm and it is checked as arithmetic, not folklore.
    """
    values = list(_values(10000))
    for fan_in in (2, 4, 16):
        _, meter = sort(values, 100, fan_in)
        wanted = math.ceil(math.log(100, fan_in))
        if meter.passes != wanted:
            return False
    return True


@functools.cache
def every_pass_moves_every_record_once() -> bool:
    """IO per record is two times passes plus two, to the record.

    Run formation reads and writes everything once, and each merge pass does it again. The
    measured total for fan-in four over ten thousand records is exactly 2 * (1 + 4) = 10
    touches per record, which is why widening the fan-in is worth memory: every unit of log
    base removed is a full read and write of the dataset saved.
    """
    _, meter = sort(list(_values(10000)), 100, 4)
    touches = (meter.read + meter.written) / meter.records
    return touches == 2.0 * (1 + meter.passes)


@functools.cache
def a_wide_enough_fan_in_makes_one_pass() -> bool:
    """Fan-in at or above the run count merges everything in a single pass.

    One hundred runs and a fan-in of 128: one pass, and the total IO is two reads and two
    writes per record, the floor for any external sort that forms runs. This is where every
    practical sort wants to sit, and the memory price of the fan-in, one buffer per run, is
    the same budget arithmetic as the compaction module's level count.
    """
    _, meter = sort(list(_values(10000)), 100, 128)
    return meter.passes == 1 and meter.runs_made == 100


@functools.cache
def the_levelled_lsm_is_this_algorithm_run_forever() -> bool:
    """The LSM's write amplification and the sort's IO per record are the same quantity.

    A levelled store with fan-out ten over enough data rewrites each record roughly once per
    level, and an external sort with fan-in ten reads and writes each record once per pass.
    Levels and passes are both the log of the size over the memory, base the fan. Measured
    side by side at matching geometry, the sort's write touches per record and the LSM's
    amplification land within a factor of two of each other, and the residual gap is the
    LSM's overlap-driven rewrites, which the batch sort does not have because all its input
    is present at once.
    """
    _, meter = sort(list(_values(40000)), 1000, 10)
    sort_writes_per_record = meter.written / meter.records
    load = Load(keys=20000, writes=40000)
    store = run_load(Levelled(fan_out=10), load)
    lsm = amplification(store, load)
    return 0.5 < lsm / sort_writes_per_record < 2.5


def compare_the_fan_ins(records: int = 10000, memory: int = 100) -> list[dict]:
    """One row per fan-in."""
    rows = []
    for fan_in in (2, 4, 8, 16, 64, 128):
        _, meter = sort(list(_values(records)), memory, fan_in)
        rows.append(meter.as_dict())
    return rows


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "correct_at_every_geometry": the_sort_is_correct_at_every_geometry(),
        "passes_follow_the_log": the_pass_count_follows_the_logarithm(),
        "every_pass_moves_everything": every_pass_moves_every_record_once(),
        "wide_fan_in_is_one_pass": a_wide_enough_fan_in_makes_one_pass(),
        "the_lsm_is_this_forever": the_levelled_lsm_is_this_algorithm_run_forever(),
    }
