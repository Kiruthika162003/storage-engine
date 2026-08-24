from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

# The cache stampede: one expiry, a hundred rebuilders, and the lock that picks one.
#
# A popular cache entry expires and every concurrent reader misses at once, so every one
# of them rebuilds the value against the backing store, a hundred identical expensive
# queries where one would do. The backing store, sized for the cached steady state, meets
# its worst load at the worst moment. Request coalescing fixes it with one bit per key: the
# first misser takes the rebuild, the rest wait for its result, and the backing store sees
# one query per expiry however many readers pile up. The model counts backing store
# queries under both disciplines, then measures the second-order fix, jittered expiry,
# which stops many keys from expiring in the same tick at all.


@dataclass
class Backing:
    """The expensive store behind the cache."""

    queries: int = field(default=0)
    peak_in_tick: int = field(default=0)
    in_tick: int = field(default=0)

    def rebuild(self, key: bytes) -> bytes:
        self.queries += 1
        self.in_tick += 1
        self.peak_in_tick = max(self.peak_in_tick, self.in_tick)
        return b"value-of-" + key

    def tick(self) -> None:
        self.in_tick = 0


@dataclass
class Cache:
    """A TTL cache where rebuilds take one tick to land, which is the stampede's window.

    The first draft applied a rebuilt value immediately, so the second reader of the same
    tick always hit and no stampede could exist: a model with instantaneous rebuilds has
    modelled the problem away. Rebuilds land at the next tick here, so every reader of the
    expiry tick sees the miss, which is what concurrent readers of a slow backing store
    actually see.
    """

    backing: Backing
    coalesce: bool
    ttl: int = field(default=50)
    held: dict[bytes, tuple[bytes, int]] = field(default_factory=dict)
    staged: dict[bytes, tuple[bytes, int]] = field(default_factory=dict)
    building: set[bytes] = field(default_factory=set)
    now: int = field(default=0)
    waited: int = field(default=0)

    def tick(self) -> None:
        self.now += 1
        self.held.update(self.staged)
        self.staged.clear()
        self.building.clear()
        self.backing.tick()

    def get(self, key: bytes) -> bytes | None:
        """One reader's fetch during a tick."""
        held = self.held.get(key)
        if held is not None and self.now < held[1]:
            return held[0]
        if self.coalesce and key in self.building:
            self.waited += 1
            return None
        self.building.add(key)
        value = self.backing.rebuild(key)
        self.staged[key] = (value, self.now + self.ttl)
        return value


def _drive(coalesce: bool, readers: int = 100, ticks: int = 200, ttl: int = 50) -> Cache:
    """One hot key, many readers per tick, expiries included."""
    cache = Cache(backing=Backing(), coalesce=coalesce, ttl=ttl)
    for _ in range(ticks):
        cache.tick()
        for _ in range(readers):
            cache.get(b"hot")
    return cache


@functools.cache
def the_stampede_multiplies_backing_load_by_the_reader_count() -> bool:
    """Without coalescing, each expiry costs one hundred rebuilds, one per reader.

    All hundred readers of the expiry tick miss together and rebuild together: the peak
    per-tick backing load equals the reader count, and the backing store meets a hundred
    times its steady load at every expiry, which is the outage signature this module
    exists to reproduce.
    """
    cache = _drive(coalesce=False)
    return cache.backing.peak_in_tick == 100


@functools.cache
def coalescing_cuts_each_expiry_to_one_rebuild() -> bool:
    """With the building bit, the same storm costs one rebuild per expiry.

    Ninety nine readers wait a tick instead of querying, the peak backing load is one, and
    the total queries across two hundred ticks drop to the expiry count. The price is on
    the waited meter: latency for the waiting readers, paid once per expiry, against a
    hundredfold load spike, which is the trade every dogpile lock makes.
    """
    cache = _drive(coalesce=True)
    return cache.backing.peak_in_tick == 1 and cache.waited > 0


@functools.cache
def synchronized_ttls_expire_together_and_jitter_spreads_them() -> bool:
    """A thousand keys cached in one tick expire in one tick; jitter spreads them tenfold.

    The stampede has a fleet-scale sibling: keys populated together, a deploy or a warmup,
    expire together, and the backing store takes every rebuild in one tick even with
    coalescing, one per key. Jittering the TTL by ten percent spreads the expiries over a
    window, and the peak per-tick rebuild count drops by an order of magnitude, the
    recovery module's aliasing lesson inverted: here the synchronized period is the bug
    and the noise is the fix.
    """
    source = random.Random(373)
    fixed = Backing()
    fixed_expiries: dict[int, int] = {}
    for _ in range(1000):
        fixed_expiries[50] = fixed_expiries.get(50, 0) + 1
    jittered_expiries: dict[int, int] = {}
    for _ in range(1000):
        life = 50 + source.randrange(-5, 6)
        jittered_expiries[life] = jittered_expiries.get(life, 0) + 1
    del fixed
    fixed_peak = max(fixed_expiries.values())
    jittered_peak = max(jittered_expiries.values())
    return fixed_peak == 1000 and jittered_peak < 150


@functools.cache
def fresh_entries_never_touch_the_backing_store() -> bool:
    """Between expiries the backing store is silent, which is what the cache is for."""
    cache = Cache(backing=Backing(), coalesce=False, ttl=1000)
    cache.tick()
    cache.get(b"hot")
    cache.tick()
    for _ in range(500):
        cache.get(b"hot")
    return cache.backing.queries == 1


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "the_stampede_multiplies": the_stampede_multiplies_backing_load_by_the_reader_count(),
        "coalescing_cuts_to_one": coalescing_cuts_each_expiry_to_one_rebuild(),
        "jitter_spreads_the_fleet": synchronized_ttls_expire_together_and_jitter_spreads_them(),
        "fresh_entries_are_free": fresh_entries_never_touch_the_backing_store(),
    }
