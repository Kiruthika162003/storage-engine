"""Batched point reads against a levelled table set, in fetches and round trips.

A client that needs sixty four keys can ask sixty four times or once. The
single ask saves network round trips by construction; whether it also saves
block fetches depends on whether sorting the batch lets adjacent keys share
blocks. Both effects are counted here, not assumed.
"""

from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

BLOCK = 16
TABLES = 6
KEYS_PER_TABLE = 512


@dataclass
class Table:
    """A sorted run: keys ascending, grouped into blocks of BLOCK keys."""

    keys: list[int]
    fetches: int = 0
    last_block: int = -1

    def block_of(self, key: int) -> int:
        low, high = 0, len(self.keys)
        while low < high:
            mid = (low + high) // 2
            if self.keys[mid] < key:
                low = mid + 1
            else:
                high = mid
        if low >= len(self.keys) or self.keys[low] != key:
            return -1
        return low // BLOCK

    def read(self, key: int, remember: bool) -> bool:
        which = self.block_of(key)
        if which < 0:
            return False
        if not (remember and which == self.last_block):
            self.fetches += 1
        self.last_block = which if remember else -1
        return True

    def forget(self) -> None:
        self.last_block = -1


@dataclass
class Tier:
    tables: list[Table] = field(default_factory=list)

    @classmethod
    def build(cls, seed: int) -> Tier:
        source = random.Random(seed)
        space = list(range(TABLES * KEYS_PER_TABLE * 2))
        source.shuffle(space)
        tables = []
        for at in range(TABLES):
            picked = space[at * KEYS_PER_TABLE : (at + 1) * KEYS_PER_TABLE]
            tables.append(Table(keys=sorted(picked)))
        return cls(tables=tables)

    def singles(self, wanted: list[int]) -> dict:
        for table in self.tables:
            table.fetches = 0
            table.forget()
        found = set()
        for key in wanted:
            for table in self.tables:
                table.forget()
                if table.read(key, remember=False):
                    found.add(key)
                    break
        return {
            "fetches": sum(table.fetches for table in self.tables),
            "trips": len(wanted),
            "found": len(found),
        }

    def batched(self, wanted: list[int]) -> dict:
        for table in self.tables:
            table.fetches = 0
            table.forget()
        remaining = sorted(set(wanted))
        found = 0
        for table in self.tables:
            table.forget()
            missed = []
            for key in remaining:
                if table.read(key, remember=True):
                    found += 1
                else:
                    missed.append(key)
            remaining = missed
        return {
            "fetches": sum(table.fetches for table in self.tables),
            "trips": 1,
            "found": found,
        }


def _wanted(seed: int, count: int, span: int) -> list[int]:
    source = random.Random(seed)
    return [source.randrange(span) for _ in range(count)]


@functools.cache
def the_batch_is_one_trip() -> bool:
    """64 singles cost 64 round trips; the batch costs 1. Both find 34 keys."""
    tier = Tier.build(11)
    wanted = _wanted(3, 64, TABLES * KEYS_PER_TABLE * 2)
    alone = tier.singles(wanted)
    together = tier.batched(wanted)
    return (
        alone["trips"] == 64
        and together["trips"] == 1
        and alone["found"] == together["found"] > 0
    )


@functools.cache
def a_scattered_batch_shares_almost_nothing() -> bool:
    """64 keys over a 6144 key space: singles fetch 34 blocks, batched 33.

    The guess was a clean tie. Measured, the batch saves exactly one fetch:
    once in 34 present keys, two land in the same block of the same table
    and sorting makes them adjacent. The saving is trips, not fetches, and
    the stray shared block is the exception that proves it.
    """
    tier = Tier.build(11)
    wanted = _wanted(3, 64, TABLES * KEYS_PER_TABLE * 2)
    alone = tier.singles(wanted)
    together = tier.batched(wanted)
    return alone["fetches"] - together["fetches"] == 1 and alone["fetches"] > 30


@functools.cache
def a_narrow_batch_shares_blocks() -> bool:
    """64 keys drawn from a 300 key window: 25 fetches fall to 11.

    Inside a window that spans a couple of blocks per table, sorting makes
    neighbours adjacent and the remembered block absorbs the second read.
    Three of the fourteen saved fetches are duplicate requests the batch
    deduplicated; adjacency pays for the other eleven. Batching pays for
    fetches only when the batch has locality.
    """
    tier = Tier.build(11)
    wanted = _wanted(3, 64, 300)
    alone = tier.singles(wanted)
    together = tier.batched(wanted)
    return together["fetches"] < alone["fetches"] * 0.8


@functools.cache
def absent_keys_cost_nothing_here_and_everything_in_life() -> bool:
    """Keys present nowhere fetch zero blocks: the binary search says no free.

    The in-memory search plays the role of a bloom filter with no false
    positives. A real engine pays a filter probe per table per absent key;
    this model shows the floor that filter is buying down to.
    """
    tier = Tier.build(11)
    ceiling = TABLES * KEYS_PER_TABLE * 2
    absent = [ceiling + at for at in range(64)]
    alone = tier.singles(absent)
    return alone["fetches"] == 0 and alone["found"] == 0


@functools.cache
def duplicates_pay_again_alone() -> bool:
    """The narrow batch repeats 7 keys; singles re-fetch 3, the batch none.

    64 draws from 300 keys collide: 57 unique. Three of the repeats name
    present keys, and the one-at-a-time client pays a block fetch for each
    repeat (25 fetches for 22 unique hits). The batch sorts and dedupes
    before it walks, so a repeated key costs zero extra by construction.
    """
    tier = Tier.build(11)
    wanted = _wanted(3, 64, 300)
    alone = tier.singles(wanted)
    return len(set(wanted)) == 57 and alone["fetches"] - alone["found"] == 3


@functools.cache
def summarise() -> dict:
    return {
        "module": "store.multiget",
        "the_batch_is_one_trip": the_batch_is_one_trip(),
        "a_scattered_batch_shares_almost_nothing": a_scattered_batch_shares_almost_nothing(),
        "a_narrow_batch_shares_blocks": a_narrow_batch_shares_blocks(),
        "duplicates_pay_again_alone": duplicates_pay_again_alone(),
        "absent_keys_cost_nothing_here_and_everything_in_life": (
            absent_keys_cost_nothing_here_and_everything_in_life()
        ),
    }
