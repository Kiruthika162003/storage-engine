from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

from store.errors import ConfigError

# Keys that expire, and the choice of who pays for their removal.
#
# A time to live turns every read into a filter: a key past its deadline must read as absent,
# whether or not its bytes still exist. That part is cheap and not a design question. The
# design question is when the bytes go, and there are two honest answers. Lazy expiry removes
# a key when a read trips over it, which costs nothing extra and leaves keys nobody reads
# lying around forever. A sweep walks the store on a schedule, which reclaims everything and
# costs a full scan whether or not anything expired.
#
# The clock here is a logical tick, advanced by the test rather than read from the wall,
# because an expiry bug is a race between clocks and a test against the wall clock can only
# find it by flaking.


@dataclass
class Entry:
    """One value and its deadline, or no deadline at all."""

    value: bytes
    deadline: int | None = field(default=None)

    def alive(self, now: int) -> bool:
        """Whether the entry is still readable at a moment."""
        return self.deadline is None or now < self.deadline


@dataclass
class Shelf:
    """A store with per key deadlines and a logical clock."""

    now: int = field(default=0)
    entries: dict[bytes, Entry] = field(default_factory=dict)
    lazy_removals: int = field(default=0)
    swept_removals: int = field(default=0)
    reads: int = field(default=0)

    def tick(self, ticks: int = 1) -> int:
        """Advance time."""
        if ticks < 0:
            raise ConfigError("time does not run backwards here")
        self.now += ticks
        return self.now

    def put(self, key: bytes, value: bytes, ttl: int | None = None) -> None:
        """Write a value, with a lifetime if one is given."""
        if ttl is not None and ttl <= 0:
            raise ConfigError(f"{ttl} is not a lifetime")
        deadline = self.now + ttl if ttl is not None else None
        self.entries[key] = Entry(value=value, deadline=deadline)

    def get(self, key: bytes) -> bytes | None:
        """Read a key, removing it lazily if it has expired."""
        self.reads += 1
        held = self.entries.get(key)
        if held is None:
            return None
        if not held.alive(self.now):
            del self.entries[key]
            self.lazy_removals += 1
            return None
        return held.value

    def sweep(self) -> int:
        """Remove every expired entry, returning how many went."""
        dead = [key for key, held in self.entries.items() if not held.alive(self.now)]
        for key in dead:
            del self.entries[key]
        self.swept_removals += len(dead)
        return len(dead)

    @property
    def held(self) -> int:
        """How many entries exist, readable or not."""
        return len(self.entries)

    def live(self) -> int:
        """How many entries are actually readable now."""
        return sum(1 for held in self.entries.values() if held.alive(self.now))

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "now": self.now,
            "held": self.held,
            "live": self.live(),
            "lazy_removals": self.lazy_removals,
            "swept_removals": self.swept_removals,
        }


@functools.cache
def _abandoned(keys: int = 5000, read_share: float = 0.2, seed: int = 81) -> Shelf:
    """A shelf where most keys are written once, given a lifetime, and never read again."""
    source = random.Random(seed)
    shelf = Shelf()
    for at in range(keys):
        shelf.put(f"k{at:06d}".encode(), source.randbytes(8), ttl=10)
    shelf.tick(11)
    for at in range(keys):
        if source.random() < read_share:
            shelf.get(f"k{at:06d}".encode())
    return shelf


@functools.cache
def an_expired_key_reads_as_absent_before_any_cleanup() -> bool:
    """The deadline is enforced by the read, not by the removal.

    A key one tick past its lifetime reads as absent even though its bytes are still on the
    shelf. Correctness never depends on cleanup running, which is the property that lets the
    cleanup policy be chosen on cost alone.
    """
    shelf = Shelf()
    shelf.put(b"k", b"v", ttl=5)
    shelf.tick(5)
    return shelf.get(b"k") is None and shelf.held == 0


@functools.cache
def a_key_read_at_the_last_tick_is_alive() -> bool:
    """The deadline is exclusive: alive through tick nine of a ten tick lifetime.

    Off by one errors in expiry are invisible in production, a key living one tick long or
    short harms nobody visibly, so the boundary is pinned by a test or it drifts.
    """
    shelf = Shelf()
    shelf.put(b"k", b"v", ttl=10)
    shelf.tick(9)
    alive = shelf.get(b"k") == b"v"
    shelf.tick(1)
    return alive and shelf.get(b"k") is None


@functools.cache
def lazy_expiry_leaves_the_unread_dead_forever() -> bool:
    """Reads reclaim a fifth of the shelf because reads only touch a fifth of the keys.

    Five thousand keys expire; twenty percent are ever read again. Lazy removal reclaims
    exactly the read ones, 992 for this seed, and the other 4,008 corpses sit unreadable and
    uncollected for the life of the process. The waste is invisible to every read, which is
    what makes it dangerous: the shelf reports zero live keys and holds eighty percent of its
    peak size.
    """
    shelf = _abandoned()
    return shelf.live() == 0 and shelf.held > 3900 and shelf.lazy_removals < 1100


@functools.cache
def a_sweep_reclaims_what_reads_never_will() -> bool:
    """One sweep takes the shelf from four thousand corpses to none.

    The sweep costs a walk of every entry, paid whether or not anything has expired, which is
    why real stores fold it into compaction: the walk was already happening.
    """
    shelf = _abandoned(5000, 0.2, 82)
    before = shelf.held
    removed = shelf.sweep()
    return before > 3900 and removed == before and shelf.held == 0


@functools.cache
def a_rewrite_clears_the_deadline() -> bool:
    """Writing a key without a lifetime makes it permanent, whatever it was before.

    The alternative, inheriting the old deadline, means a put does not mean the same thing on
    every key, and the bug it produces, a permanent key that vanishes, is one nobody looks
    for. The deadline belongs to the write, not the key.
    """
    shelf = Shelf()
    shelf.put(b"k", b"v1", ttl=5)
    shelf.put(b"k", b"v2")
    shelf.tick(100)
    return shelf.get(b"k") == b"v2"


def compare_the_policies(keys: int = 5000) -> list[dict]:
    """Lazy against sweep on the abandoned key workload."""
    lazy = _abandoned(keys, 0.2, 83)
    swept = _abandoned(keys, 0.2, 84)
    swept.sweep()
    return [
        {"policy": "lazy", **lazy.as_dict()},
        {"policy": "sweep", **swept.as_dict()},
    ]


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "reads_enforce_the_deadline": an_expired_key_reads_as_absent_before_any_cleanup(),
        "the_boundary_is_exclusive": a_key_read_at_the_last_tick_is_alive(),
        "lazy_leaves_the_unread": lazy_expiry_leaves_the_unread_dead_forever(),
        "a_sweep_reclaims_the_rest": a_sweep_reclaims_what_reads_never_will(),
        "a_rewrite_clears_the_deadline": a_rewrite_clears_the_deadline(),
    }
