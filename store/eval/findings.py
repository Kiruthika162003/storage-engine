from __future__ import annotations

import functools

from store.eval.run import run
from store.eval.workload import MIXES

# What the workload runs actually say, each claim tied to the numbers that back it.


@functools.cache
def a_read_heavy_mix_barely_exercises_the_write_path() -> bool:
    """Nineteen thousand gets produce one flush and no folds.

    The read heavy mix writes five percent of the time, which over twenty thousand operations
    is not enough to fill the memtable twice. Everything the compaction modules measure is
    idle here, and the whole cost of the mix is the read path: 8,403 of the misses never
    reached a file because the filter turned them away.

    A store sized for this mix is a cache with a durability story, and tuning its compaction
    is tuning the part that never runs.
    """
    meter = run("read_heavy")
    return meter.flushes <= 2 and meter.folds == 0 and meter.filter_skips > 5000


@functools.cache
def an_insert_mix_is_all_maintenance() -> bool:
    """Nineteen thousand puts produce 37 flushes and 12 folds.

    The insert heavy mix is the opposite corner: the read path is idle and the store spends
    its life flushing and folding. The two mixes bound the design space, and every real
    workload is somewhere on the line between them, which is why the same engine defaults
    cannot be right for both ends.
    """
    reads = run("read_heavy")
    inserts = run("insert_heavy")
    return inserts.flushes > reads.flushes * 10 and inserts.folds > 10


@functools.cache
def hot_reads_hit_two_thirds_and_uniform_reads_one_tenth() -> bool:
    """The same blend, the same key count, and the hit rate goes 9.7 to 66 percent.

    The only difference between read_heavy and hot_reads is that ninety percent of hot reads
    land in five percent of the keyspace. The writes concentrate the same way, so the keys
    being read are the keys that exist. The hit rate is a property of the correlation between
    the read and write distributions, not of the store, and no engine tuning moves it.
    """
    uniform = run("read_heavy")
    hot = run("hot_reads")
    return uniform.hit_rate < 0.15 and hot.hit_rate > 0.5


@functools.cache
def scans_keep_more_tables_alive() -> bool:
    """The scan mix ends with more files than the balanced mix, from fewer operations.

    Scans do not trigger folds and their interleaved puts trickle, so the file count drifts
    up. A real engine folds on a timer as well as on a threshold for exactly this reason, and
    the absence of that timer here is visible in the meter.
    """
    scans = run("scan_heavy")
    balanced = run("balanced")
    return scans.tables_after >= balanced.tables_after


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "read_heavy_barely_writes": a_read_heavy_mix_barely_exercises_the_write_path(),
        "insert_heavy_is_maintenance": an_insert_mix_is_all_maintenance(),
        "hit_rate_is_correlation": hot_reads_hit_two_thirds_and_uniform_reads_one_tenth(),
        "scans_keep_tables_alive": scans_keep_more_tables_alive(),
    }


def everything() -> dict:
    """The full evaluation: every mix's meter and every claim."""
    return {
        "mixes": [run(mix.name).as_dict() for mix in MIXES],
        "claims": summarise(),
    }
