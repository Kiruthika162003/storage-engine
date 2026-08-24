"""Two caches stacked, one budget split, and what duplication costs.

An engine cache sits above the operating system's page cache and both hold
blocks. With a fixed memory budget the operator chooses a split, and the
folk warning is that a naive split holds every hot block twice. The sims
here measure hit rates for one big cache, an even split with inclusive
behaviour, and an even split where the lower layer is told to drop what
the upper layer admits.
"""

from __future__ import annotations

import functools
import random
from collections import OrderedDict
from dataclasses import dataclass, field

BUDGET = 512
BLOCKS = 4096
READS = 40000


@dataclass
class Lru:
    size: int
    held: OrderedDict = field(default_factory=OrderedDict)
    hits: int = 0
    misses: int = 0

    def get(self, block: int) -> bool:
        if block in self.held:
            self.held.move_to_end(block)
            self.hits += 1
            return True
        self.misses += 1
        return False

    def admit(self, block: int) -> int | None:
        if self.size == 0:
            return None
        self.held[block] = True
        self.held.move_to_end(block)
        if len(self.held) > self.size:
            evicted, _ = self.held.popitem(last=False)
            return evicted
        return None

    def drop(self, block: int) -> None:
        self.held.pop(block, None)


def _reads(seed: int) -> list[int]:
    source = random.Random(seed)
    hot = list(range(400))
    return [
        source.choice(hot) if source.random() < 0.8 else source.randrange(BLOCKS)
        for _ in range(READS)
    ]


@dataclass
class Stack:
    """Upper engine cache over lower page cache; disk reads counted."""

    upper: Lru
    lower: Lru
    exclusive: bool
    disk_reads: int = 0

    def read(self, block: int) -> None:
        if self.upper.get(block):
            return
        if self.lower.get(block):
            if self.exclusive:
                self.lower.drop(block)
            self._admit_up(block)
            return
        self.disk_reads += 1
        self._admit_up(block)
        if not self.exclusive:
            self.lower.admit(block)

    def _admit_up(self, block: int) -> None:
        demoted = self.upper.admit(block)
        if self.exclusive and demoted is not None:
            self.lower.admit(demoted)

    def duplicated(self) -> int:
        return len(set(self.upper.held) & set(self.lower.held))


def single(seed: int) -> Stack:
    stack = Stack(upper=Lru(size=BUDGET), lower=Lru(size=0), exclusive=True)
    for block in _reads(seed):
        stack.read(block)
    return stack


def split(seed: int, exclusive: bool) -> Stack:
    stack = Stack(
        upper=Lru(size=BUDGET // 2),
        lower=Lru(size=BUDGET // 2),
        exclusive=exclusive,
    )
    for block in _reads(seed):
        stack.read(block)
    return stack


@functools.cache
def the_inclusive_split_holds_the_hot_set_twice() -> bool:
    """220 of the lower layer's 256 blocks also sit in the upper layer.

    Inclusive stacking admits every disk read to both layers, so 86
    percent of the lower half's memory repeats what the upper half holds.
    The budget bought 512 slots and stores 292 distinct blocks.
    """
    stack = split(3, exclusive=False)
    return stack.duplicated() == 220 and len(stack.lower.held) == BUDGET // 2


@functools.cache
def duplication_costs_eighty_five_percent_more_disk() -> bool:
    """One cache of 512: 11811 disk reads. The inclusive split: 21904.

    The duplicated half is memory that stops no disk read, and the miss
    count rises 85 percent for the same budget and the same trace.
    """
    return split(3, exclusive=False).disk_reads / single(3).disk_reads > 1.8


@functools.cache
def exclusion_makes_two_caches_one() -> bool:
    """The exclusive split's 11811 disk reads equal the single cache's exactly.

    Demote on eviction, drop on promotion: the pair holds 512 distinct
    blocks in strict recency order, one LRU living in two rooms. The match
    is exact, not approximate, read for read across 40000 reads.
    """
    return split(3, exclusive=True).disk_reads == single(3).disk_reads


@functools.cache
def the_lower_layer_earns_its_keep_only_under_exclusion() -> bool:
    """Lower layer hits: 1863 inclusive, 11956 exclusive, 6.4 times more.

    Inclusively, the lower layer mostly re-holds upper blocks that never
    miss, so it answers little. Exclusively it holds the next-warmest 256
    blocks and serves them; both stacks saw identical upper hits, 16233,
    because the upper halves behave identically either way.
    """
    inclusive = split(3, exclusive=False)
    exclusive = split(3, exclusive=True)
    return (
        exclusive.lower.hits > inclusive.lower.hits * 6
        and exclusive.upper.hits == inclusive.upper.hits
    )


@functools.cache
def summarise() -> dict:
    return {
        "module": "store.doublecache",
        "the_inclusive_split_holds_the_hot_set_twice": (
            the_inclusive_split_holds_the_hot_set_twice()
        ),
        "duplication_costs_eighty_five_percent_more_disk": (
            duplication_costs_eighty_five_percent_more_disk()
        ),
        "exclusion_makes_two_caches_one": exclusion_makes_two_caches_one(),
        "the_lower_layer_earns_its_keep_only_under_exclusion": (
            the_lower_layer_earns_its_keep_only_under_exclusion()
        ),
    }
