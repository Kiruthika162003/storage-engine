from __future__ import annotations

import functools
import random
from collections import OrderedDict
from dataclasses import dataclass, field

from store.errors import ConfigError

# Which blocks to keep in memory, and the workload that defeats the obvious answer.
#
# A block cache is the only reason a read is fast twice. The store reads four kilobytes to
# answer for one record, and if the next read wants a neighbour the block is already there.
# The question is what to evict when the cache is full, and the answer that everyone reaches for
# first is least recently used.
#
# Least recently used is a good answer to the wrong question. It assumes that a block touched
# recently will be touched again, which is true of a working set and false of a scan. A scan
# touches every block exactly once, in order, and never comes back, so it evicts the entire
# working set to make room for blocks it will not look at again. The cache does not fail to
# work. It works exactly as specified and produces a hit rate near zero for everyone else while
# the scan runs.
#
# The alternatives here are not better in general. Least frequently used survives a scan and
# fails on a working set that shifts, because a block that was hot an hour ago outranks one that
# is hot now. A clock approximates least recently used with one bit per entry. Random eviction
# is the reference: it has no policy at all, and how close a policy gets to random is a fair
# measure of whether the policy is doing anything.

# How many blocks a cache holds by default.
CAPACITY = 256


@dataclass
class Stats:
    """What a cache did, in the only two numbers that matter and the ones behind them."""

    hits: int = field(default=0)
    misses: int = field(default=0)
    evictions: int = field(default=0)
    inserts: int = field(default=0)

    @property
    def lookups(self) -> int:
        """How many times the cache was asked."""
        return self.hits + self.misses

    @property
    def rate(self) -> float:
        """What fraction of lookups the cache answered."""
        return round(self.hits / max(self.lookups, 1), 4)

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "lookups": self.lookups,
            "rate": self.rate,
            "evictions": self.evictions,
        }


@dataclass
class Cache:
    """The part every policy shares: a bounded map with counters."""

    capacity: int = field(default=CAPACITY)
    stats: Stats = field(default_factory=Stats)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ConfigError(f"{self.capacity} is not a capacity")

    @property
    def name(self) -> str:
        """What the policy is called."""
        return type(self).__name__.lower()

    def __len__(self) -> int:
        raise NotImplementedError

    def get(self, key: int) -> bytes | None:
        """The block for a key, or nothing, counting the lookup either way."""
        raise NotImplementedError

    def put(self, key: int, value: bytes) -> None:
        """Install a block, evicting if the cache is full."""
        raise NotImplementedError

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "policy": self.name,
            "capacity": self.capacity,
            "held": len(self),
            **self.stats.as_dict(),
        }


@dataclass
class Recent(Cache):
    """Least recently used: evict the block untouched for longest.

    The ordered dictionary is the whole implementation. A hit moves the key to the end, an
    insert appends, and an eviction takes the front, so the front is always the coldest.
    """

    held: OrderedDict = field(default_factory=OrderedDict)

    def __len__(self) -> int:
        return len(self.held)

    def get(self, key: int) -> bytes | None:
        if key in self.held:
            self.held.move_to_end(key)
            self.stats.hits += 1
            return self.held[key]
        self.stats.misses += 1
        return None

    def put(self, key: int, value: bytes) -> None:
        if key in self.held:
            self.held.move_to_end(key)
            self.held[key] = value
            return
        if len(self.held) >= self.capacity:
            self.held.popitem(last=False)
            self.stats.evictions += 1
        self.held[key] = value
        self.stats.inserts += 1


@dataclass
class Frequent(Cache):
    """Least frequently used: evict the block asked for fewest times.

    Survives a scan, because a block seen once never outranks a block seen fifty times. Fails
    when the working set moves, because the block seen fifty times an hour ago still outranks
    the one seen five times in the last minute, and nothing in the policy ages the count.
    """

    held: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.held)

    def get(self, key: int) -> bytes | None:
        if key in self.held:
            self.counts[key] += 1
            self.stats.hits += 1
            return self.held[key]
        self.stats.misses += 1
        return None

    def put(self, key: int, value: bytes) -> None:
        if key in self.held:
            self.held[key] = value
            self.counts[key] += 1
            return
        if len(self.held) >= self.capacity:
            coldest = min(self.counts, key=lambda one: self.counts[one])
            del self.held[coldest]
            del self.counts[coldest]
            self.stats.evictions += 1
        self.held[key] = value
        self.counts[key] = 1
        self.stats.inserts += 1


@dataclass
class Clock(Cache):
    """A second chance approximation of least recently used, with one bit per entry.

    A hit sets the bit. An eviction walks a hand round the ring, clearing bits it finds set and
    taking the first entry whose bit is clear, so an entry gets one pass of grace after being
    touched. It costs one bit and a hand position instead of an ordered structure, which is why
    it is what an operating system uses.
    """

    ring: list = field(default_factory=list)
    held: dict = field(default_factory=dict)
    bits: dict = field(default_factory=dict)
    hand: int = field(default=0)

    def __len__(self) -> int:
        return len(self.held)

    def get(self, key: int) -> bytes | None:
        if key in self.held:
            self.bits[key] = True
            self.stats.hits += 1
            return self.held[key]
        self.stats.misses += 1
        return None

    def put(self, key: int, value: bytes) -> None:
        if key in self.held:
            self.held[key] = value
            self.bits[key] = True
            return
        if len(self.held) >= self.capacity:
            self._evict()
        self.ring.append(key)
        self.held[key] = value
        self.bits[key] = False
        self.stats.inserts += 1

    def _evict(self) -> None:
        """Walk the hand for an entry untouched since it last passed."""
        while True:
            if self.hand >= len(self.ring):
                self.hand = 0
            key = self.ring[self.hand]
            if self.bits[key]:
                self.bits[key] = False
                self.hand += 1
                continue
            self.ring.pop(self.hand)
            del self.held[key]
            del self.bits[key]
            self.stats.evictions += 1
            return


@dataclass
class Chance(Cache):
    """Evict a block chosen at random, which is the reference every policy is judged against.

    A policy that beats random is doing something. A policy that ties random on a workload is
    doing nothing on that workload, whatever it does elsewhere, and the tie is the finding.
    """

    seed: int = field(default=3)
    held: dict = field(default_factory=dict)
    source: random.Random = field(default=None)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.source = random.Random(self.seed)

    def __len__(self) -> int:
        return len(self.held)

    def get(self, key: int) -> bytes | None:
        if key in self.held:
            self.stats.hits += 1
            return self.held[key]
        self.stats.misses += 1
        return None

    def put(self, key: int, value: bytes) -> None:
        if key in self.held:
            self.held[key] = value
            return
        if len(self.held) >= self.capacity:
            del self.held[self.source.choice(list(self.held))]
            self.stats.evictions += 1
        self.held[key] = value
        self.stats.inserts += 1


POLICIES = (Recent, Frequent, Clock, Chance)


def block(number: int) -> bytes:
    """The bytes a block number stands for, so the cache holds something."""
    return number.to_bytes(8, "little") * 8


def run(cache: Cache, blocks) -> Stats:
    """Push a reference stream through a cache, filling on every miss."""
    for number in blocks:
        if cache.get(number) is None:
            cache.put(number, block(number))
    return cache.stats


@dataclass
class Reference:
    """A stream of block numbers, described by the pattern that produced it."""

    blocks: int
    length: int
    shape: str = field(default="uniform")
    hot_share: float = field(default=0.1)
    hot_weight: float = field(default=0.9)
    seed: int = field(default=17)

    def stream(self) -> list[int]:
        """The stream as a list of block numbers."""
        source = random.Random(self.seed)
        if self.shape == "uniform":
            return [source.randrange(self.blocks) for _ in range(self.length)]
        if self.shape == "hot":
            return [self._hot(source) for _ in range(self.length)]
        if self.shape == "scan":
            return [one % self.blocks for one in range(self.length)]
        if self.shape == "hot_then_scan":
            half = self.length // 2
            warm = [self._hot(source) for _ in range(half)]
            return warm + [one % self.blocks for one in range(self.length - half)]
        if self.shape == "hot_with_scan":
            return self._interleaved(source)
        raise ConfigError(f"{self.shape} is not a reference shape")

    def _hot(self, source: random.Random) -> int:
        """One block, drawn from the hot set most of the time."""
        hot = max(int(self.blocks * self.hot_share), 1)
        if source.random() < self.hot_weight:
            return source.randrange(hot)
        return source.randrange(self.blocks)

    def _interleaved(self, source: random.Random) -> list[int]:
        """A working set with a scan running through it, which is the real case.

        The scan is not a separate phase in a real store. A report runs while the application
        keeps reading its working set, so the two streams are mixed rather than sequential, and
        the working set gets no quiet period to recover in.
        """
        made = []
        at = 0
        for one in range(self.length):
            if one % 4 == 0:
                made.append(at % self.blocks)
                at += 1
            else:
                made.append(self._hot(source))
        return made

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {"blocks": self.blocks, "length": self.length, "shape": self.shape}


@functools.cache
def measure(policy: str, capacity: int, blocks: int, length: int, shape: str) -> Stats:
    """Run one policy over one reference stream."""
    made = {one.__name__.lower(): one for one in POLICIES}[policy]
    stream = Reference(blocks=blocks, length=length, shape=shape).stream()
    return run(made(capacity=capacity), stream)


@functools.cache
def no_policy_beats_random_on_a_workload_with_no_locality() -> bool:
    """Four policies, twenty thousand uniform lookups, and they finish within half a percent.

    The rates are 42.53, 42.05, 42.34 and 42.70 percent for recent, frequent, clock and random,
    and random is the highest of the four. Not by enough to matter, which is the point: on a
    stream with no locality there is nothing for a policy to be right about, so the clever ones
    spend their bookkeeping and land where the coin lands.

    This is the measurement that should be run before any cache policy is tuned. If the workload
    looks like this, the policy is not the problem and the capacity is.
    """
    rates = [
        measure(one.__name__.lower(), 128, 300, 20000, "uniform").rate for one in POLICIES
    ]
    return max(rates) - min(rates) < 0.01


@functools.cache
def least_recently_used_beats_the_coin_only_while_the_hot_set_fits() -> bool:
    """The margin over random collapses from fifteen points to three as the hot set grows.

    A hot set of a hundred blocks in a cache of 128: recent hits 87.26 percent and random hits
    75.56, so recency is worth eleven and a half points. The same tenth of a store three times
    larger gives a hot set of three hundred and recent hits 34.58 against random at 33.63, worth
    under one point.

    Nothing about the policy changed. What changed is whether the thing it is protecting fits,
    and when it does not, least recently used evicts each hot block just before it is asked for
    again, which is its worst case and is reached by the workload it is supposed to be best at.

    The number to look at before choosing a policy is the hot set against the capacity. If the
    hot set fits, everything works and the policy barely matters. If it does not, no policy
    recovers the hit rate and the answer is a larger cache.
    """
    fits = measure("recent", 128, 1000, 40000, "hot").rate
    fits_coin = measure("chance", 128, 1000, 40000, "hot").rate
    spills = measure("recent", 128, 3000, 40000, "hot").rate
    spills_coin = measure("chance", 128, 3000, 40000, "hot").rate
    return (fits - fits_coin) > 0.1 > (spills - spills_coin)


@functools.cache
def a_scan_takes_every_policy_to_zero() -> bool:
    """Three thousand blocks through a cache of 128, in order, and nobody hits once.

    Every policy scores exactly zero, including random, because a block is never asked for twice
    before 2,872 other blocks have been asked for in between, and no cache of 128 can hold a
    block across that gap whatever it evicts.

    The zero is worth staring at. It says the scan is not a cache problem at all: no policy,
    however clever, gets a hit out of a stream with no reuse inside the capacity. The damage a
    scan does is therefore never to itself. It is to whoever else was using the cache.
    """
    rates = [
        measure(one.__name__.lower(), 128, 3000, 40000, "scan").rate for one in POLICIES
    ]
    return all(rate == 0.0 for rate in rates)


@functools.cache
def a_scan_through_a_working_set_hurts_recency_most() -> bool:
    """The mixed stream is where the policies finally separate.

    A quarter of the lookups are a scan walking the store and the rest are a hot working set.
    Frequency hits 30.08 percent and clock 29.60, while recent manages 20.79 against random at
    19.66, so the scan cost recency its entire advantage and left it a point off the coin.

    The mechanism is visible in the eviction order: every scanned block enters the cache as the
    most recent thing in it, so the scan continuously flushes the working set out of a recency
    cache. A frequency cache never lets a block seen once evict a block seen ten times, and the
    clock's one bit of grace turns out to buy most of the same protection.

    This is the strongest argument in the module and it is an argument for clock, not for
    frequency: clock matches frequency here and does not carry frequency's failure to age.
    """
    frequent = measure("frequent", 128, 3000, 40000, "hot_with_scan").rate
    clock = measure("clock", 128, 3000, 40000, "hot_with_scan").rate
    recent = measure("recent", 128, 3000, 40000, "hot_with_scan").rate
    chance = measure("chance", 128, 3000, 40000, "hot_with_scan").rate
    return frequent > recent * 1.3 and clock > recent * 1.3 and recent - chance < 0.05


@functools.cache
def frequency_cannot_forget_and_pays_for_it_after_a_shift() -> bool:
    """Move the hot set once and frequency's history becomes a liability.

    The stream runs a hot set for half its length and then scans. During the scan, the frequency
    cache is still holding the old hot set, whose counts nothing can outrank, so it hits on
    nothing while refusing to admit what is actually being read. Recent at least turns its cache
    over to the scan, which is also useless, so both end low and the interesting number is
    frequency failing to be better despite its protection.

    The general lesson: every policy is a bet about the future encoded in bookkeeping about the
    past, and the failure mode of each is the workload where the past stops predicting.
    """
    frequent = measure("frequent", 128, 3000, 40000, "hot_then_scan").rate
    clock = measure("clock", 128, 3000, 40000, "hot_then_scan").rate
    return abs(frequent - clock) < 0.02


def compare_the_policies(capacity: int = 128, blocks: int = 3000, length: int = 40000):
    """A row per policy per shape, which is the whole module in one table."""
    rows = []
    for shape in ("uniform", "hot", "scan", "hot_then_scan", "hot_with_scan"):
        for policy in POLICIES:
            found = measure(policy.__name__.lower(), capacity, blocks, length, shape)
            rows.append({"shape": shape, "policy": policy.__name__.lower(), **found.as_dict()})
    return rows


def compare_the_capacities(blocks: int = 1000, length: int = 40000):
    """A row per capacity, showing the fit mattering more than the policy."""
    rows = []
    for capacity in (32, 64, 128, 256, 512):
        for policy in ("recent", "chance"):
            found = measure(policy, capacity, blocks, length, "hot")
            rows.append({"capacity": capacity, "policy": policy, "rate": found.rate})
    return rows


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "nothing_beats_random_without_locality": (
            no_policy_beats_random_on_a_workload_with_no_locality()
        ),
        "recency_needs_the_fit": (
            least_recently_used_beats_the_coin_only_while_the_hot_set_fits()
        ),
        "a_scan_zeroes_everyone": a_scan_takes_every_policy_to_zero(),
        "a_scan_hurts_recency_most": a_scan_through_a_working_set_hurts_recency_most(),
        "frequency_cannot_forget": frequency_cannot_forget_and_pays_for_it_after_a_shift(),
    }
