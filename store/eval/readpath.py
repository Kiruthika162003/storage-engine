"""One read path, eight configurations, three workloads, costs in charges.

The read path has three optional organs: a bloom filter per table, a block
cache shared across tables, and a sparse index whose granule sets how many
blocks a lookup must read when nothing else helps. Every configuration is
charged the same way, one charge per filter probe and ten per block read,
and the sweep shows which organ pays under which workload rather than
arguing about it.
"""

from __future__ import annotations

import functools
import random
from collections import OrderedDict
from dataclasses import dataclass, field

PROBE = 1
READ = 10
TABLES = 8
KEYS_PER_TABLE = 1024
SPACE = TABLES * KEYS_PER_TABLE * 2
BLOCK = 16


@dataclass
class Path:
    bloom: bool
    cache_size: int
    granule: int
    tables: list[set[int]] = field(default_factory=list)
    cache: OrderedDict = field(default_factory=OrderedDict)
    charges: int = 0

    @classmethod
    def build(cls, seed: int, bloom: bool, cache_size: int, granule: int) -> Path:
        source = random.Random(seed)
        space = list(range(SPACE))
        source.shuffle(space)
        tables = [
            set(space[at * KEYS_PER_TABLE : (at + 1) * KEYS_PER_TABLE])
            for at in range(TABLES)
        ]
        return cls(bloom=bloom, cache_size=cache_size, granule=granule, tables=tables)

    def _blocks_per_lookup(self) -> int:
        return max(1, self.granule // BLOCK)

    def _fetch(self, table_at: int, key: int) -> None:
        granule_at = key // self.granule
        for offset in range(self._blocks_per_lookup()):
            tag = (table_at, granule_at * self._blocks_per_lookup() + offset)
            if self.cache_size and tag in self.cache:
                self.cache.move_to_end(tag)
                continue
            self.charges += READ
            if self.cache_size:
                self.cache[tag] = True
                if len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)

    def get(self, key: int) -> bool:
        for table_at, table in enumerate(self.tables):
            holds = key in table
            if self.bloom:
                self.charges += PROBE
                if not holds:
                    continue
            self._fetch(table_at, key)
            if holds:
                return True
        return False


def _workload(name: str, seed: int, count: int) -> list[int]:
    source = random.Random(seed)
    present = [key for key in range(SPACE) if _held(key)]
    if name == "uniform":
        return [source.choice(present) for _ in range(count)]
    if name == "hot":
        hot = present[:64]
        return [
            source.choice(hot) if source.random() < 0.9 else source.choice(present)
            for _ in range(count)
        ]
    return [
        source.choice(present) if source.random() < 0.2 else SPACE + source.randrange(SPACE)
        for _ in range(count)
    ]


@functools.cache
def _held(key: int) -> bool:
    return any(key in table for table in Path.build(19, False, 0, BLOCK).tables)


@functools.cache
def sweep() -> tuple:
    rows = []
    for name in ("uniform", "hot", "absent"):
        wanted = _workload(name, 5, 4000)
        for bloom in (False, True):
            for cache_size in (0, 4096):
                path = Path.build(19, bloom, cache_size, BLOCK)
                for key in wanted:
                    path.get(key)
                rows.append(
                    {
                        "workload": name,
                        "bloom": bloom,
                        "cache": cache_size,
                        "charges": path.charges,
                    }
                )
    return tuple(rows)


def _charges(workload: str, bloom: bool, cache: int) -> int:
    for row in sweep():
        if row["workload"] == workload and row["bloom"] == bloom and row["cache"] == cache:
            return row["charges"]
    raise KeyError(workload)


@functools.cache
def the_filter_pays_on_present_keys_too() -> bool:
    """Uniform present reads: 182220 charges bare, 58222 with the filter.

    The folk rule says bloom filters pay only for absent keys. Wrong for a
    levelled store: a present key lives in one of eight tables, so seven
    probes say no and save seven block reads. The filter cuts a purely
    present workload by 3.1x before any cache is involved.
    """
    return _charges("uniform", False, 0) / _charges("uniform", True, 0) > 3.0


@functools.cache
def a_hot_cache_makes_the_filter_overhead() -> bool:
    """Hot reads with a big cache: 16630 charges alone, 22182 with bloom.

    Once the hot set is cached, block reads are free and the filter's
    probes are the only cost left: 5552 charges of pure overhead per 4000
    reads. The organ that wins one workload loses another.
    """
    return _charges("hot", False, 4096) < _charges("hot", True, 4096) * 0.8


@functools.cache
def the_filter_owns_the_absent_workload() -> bool:
    """80 percent absent reads: 291470 bare, 37117 with the filter, 7.9x.

    Adding the cache on top of the filter saves only 850 more charges,
    because the filter already removed the reads the cache would have
    served. The cache alone leaves 187760, worse than the filter alone.
    """
    ratio = _charges("absent", False, 0) / _charges("absent", True, 0)
    stacked = _charges("absent", True, 0) - _charges("absent", True, 4096)
    return ratio > 7.5 and stacked < 1000


@functools.cache
def the_organs_overlap_rather_than_add() -> bool:
    """Uniform: bloom saves 123998, cache saves 98580, together 137548.

    The sum of the solo savings is 222578; the joint saving is 62 percent
    of that. Both organs remove the same wasted reads from tables that do
    not hold the key, so their benefits overlap instead of adding.
    """
    base = _charges("uniform", False, 0)
    solo = (base - _charges("uniform", True, 0)) + (base - _charges("uniform", False, 4096))
    joint = base - _charges("uniform", True, 4096)
    return joint < solo * 0.65


@functools.cache
def a_coarse_granule_multiplies_reads() -> bool:
    """Granule 128 reads 8 blocks where granule 16 reads 1: charges say 8.0x.

    The sparse index's granule is the number of keys between indexed
    points; a lookup reads every block the granule spans. Coarsening the
    index eightfold multiplies uncached read cost by exactly eight.
    """
    fine = Path.build(19, False, 0, 16)
    coarse = Path.build(19, False, 0, 128)
    wanted = _workload("uniform", 9, 500)
    for key in wanted:
        fine.get(key)
        coarse.get(key)
    return coarse.charges == fine.charges * 8


def render() -> str:
    lines = ["workload  bloom  cache  charges"]
    for row in sweep():
        lines.append(
            f"{row['workload']:<9} {row['bloom']!s:<6} "
            f"{row['cache']:<6} {row['charges']}"
        )
    return "\n".join(lines)


@functools.cache
def summarise() -> dict:
    return {
        "module": "store.eval.readpath",
        "the_filter_pays_on_present_keys_too": the_filter_pays_on_present_keys_too(),
        "a_hot_cache_makes_the_filter_overhead": a_hot_cache_makes_the_filter_overhead(),
        "the_filter_owns_the_absent_workload": the_filter_owns_the_absent_workload(),
        "the_organs_overlap_rather_than_add": the_organs_overlap_rather_than_add(),
        "a_coarse_granule_multiplies_reads": a_coarse_granule_multiplies_reads(),
    }
