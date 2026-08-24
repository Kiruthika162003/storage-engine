from __future__ import annotations

import functools
import hashlib
import random
from dataclasses import dataclass, field

from store.errors import ConfigError

# Splitting one keyspace across many stores, and what each split rule costs to change.
#
# Two rules cover practice. Hash sharding sends a key to the shard its hash names, which
# spreads any workload evenly and destroys the ordering, so a scan has to visit every shard.
# Range sharding gives each shard a contiguous slice, which keeps scans local and hands the
# hottest prefix to whichever shard drew it.
#
# The measurement that separates them is not the steady state, it is the resize. Adding one
# shard to a modulo hashed cluster moves almost every key, because nearly every hash changes
# its remainder. A hash ring moves only the keys the new shard takes over, which is the point
# of the ring. A range split moves half of one shard and nothing else.

HASH_SLOTS = 128


def _digest(key: bytes) -> int:
    """A stable hash, deliberately not Python's own, which is salted per process."""
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")


@dataclass
class Modulo:
    """Hash modulo the shard count, which is what everyone builds first."""

    shards: int

    def __post_init__(self) -> None:
        if self.shards < 1:
            raise ConfigError(f"{self.shards} is not a shard count")

    def place(self, key: bytes) -> int:
        """Which shard a key lives on."""
        return _digest(key) % self.shards

    def grown(self) -> Modulo:
        """The same rule with one more shard."""
        return Modulo(shards=self.shards + 1)


@dataclass
class Ring:
    """Consistent hashing: shards own arcs of a circle, keys walk to the next owner."""

    shards: int
    slots_per_shard: int = field(default=HASH_SLOTS)
    points: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.shards < 1:
            raise ConfigError(f"{self.shards} is not a shard count")
        if not self.points:
            for shard in range(self.shards):
                for slot in range(self.slots_per_shard):
                    where = _digest(f"shard-{shard}-slot-{slot}".encode())
                    self.points.append((where, shard))
            self.points.sort()

    def place(self, key: bytes) -> int:
        """The first shard point at or past the key's position, wrapping."""
        where = _digest(key)
        low, high = 0, len(self.points)
        while low < high:
            middle = (low + high) // 2
            if self.points[middle][0] < where:
                low = middle + 1
            else:
                high = middle
        if low == len(self.points):
            low = 0
        return self.points[low][1]

    def grown(self) -> Ring:
        """The same ring with one more shard's points added."""
        made = Ring(shards=self.shards + 1, slots_per_shard=self.slots_per_shard)
        return made


@dataclass
class Ranges:
    """Contiguous slices, split boundaries kept sorted."""

    boundaries: list[bytes] = field(default_factory=list)

    def place(self, key: bytes) -> bytes:
        """The lower boundary of the slice holding the key, which is the shard's name.

        The name is the boundary, not the slice's position, and the first draft got this
        wrong: returning the index made an early split look like a total reshuffle, because
        inserting one boundary renumbers every slice after it. 95 percent of keys changed
        shard number while zero percent of data would have moved. A shard's identity has to
        be stable under other shards' splits, and the lower boundary is.
        """
        low, high = 0, len(self.boundaries)
        while low < high:
            middle = (low + high) // 2
            if self.boundaries[middle] <= key:
                low = middle + 1
            else:
                high = middle
        return self.boundaries[low - 1] if low else b""

    @property
    def shards(self) -> int:
        """One more shard than boundaries."""
        return len(self.boundaries) + 1

    def split(self, boundary: bytes) -> Ranges:
        """A new layout with one slice cut in two."""
        if boundary in self.boundaries:
            raise ConfigError(f"{boundary!r} is already a boundary")
        made = sorted([*self.boundaries, boundary])
        return Ranges(boundaries=made)


@functools.cache
def _keys(count: int = 20000, seed: int = 91) -> tuple[bytes, ...]:
    """A keyspace with the usual shape: prefixed, dense, some hot prefixes."""
    source = random.Random(seed)
    made = []
    for _ in range(count):
        tenant = int(20 * source.random() ** 2)
        made.append(f"tenant:{tenant:03d}:item:{source.randrange(100000):08d}".encode())
    return tuple(made)


def moved(before, after, keys) -> float:
    """What fraction of keys change shards between two layouts."""
    changed = sum(1 for key in keys if before.place(key) != after.place(key))
    return round(changed / max(len(keys), 1), 4)


def spread(layout, keys) -> float:
    """The ratio of the fullest shard to the emptiest, one meaning perfectly even."""
    counts: dict[int, int] = {}
    for key in keys:
        counts[layout.place(key)] = counts.get(layout.place(key), 0) + 1
    if not counts:
        return 1.0
    return round(max(counts.values()) / max(min(counts.values()), 1), 3)


@functools.cache
def adding_a_shard_to_a_modulo_cluster_moves_nearly_everything() -> bool:
    """Nine shards to ten: 90.2 percent of keys change homes.

    The remainder changes for every key whose hash is not a multiple of both counts, which is
    nearly all of them. The move is not rebalancing, it is a full reshuffle, and every moved
    key is a copy across the network followed by a delete: the cheapest possible growth event
    costs about one full copy of the cluster.
    """
    before = Modulo(shards=9)
    keys = _keys()
    return moved(before, before.grown(), keys) > 0.85


@functools.cache
def a_ring_moves_only_the_new_shards_share() -> bool:
    """Nine shards to ten on a ring: 9.7 percent of keys move, which is one tenth.

    The new shard's points claim arcs from every existing shard evenly, and only the keys on
    those arcs move. One tenth of the data for a tenth shard is the theoretical floor, and the
    ring sits on it to within noise, which is the entire argument for consistent hashing.
    """
    before = Ring(shards=9)
    keys = _keys()
    fraction = moved(before, before.grown(), keys)
    return 0.05 < fraction < 0.15


@functools.cache
def a_range_split_moves_half_of_one_shard() -> bool:
    """Splitting one slice moves 5.0 percent of the cluster and zero percent of the rest.

    The split boundary cuts one shard's slice in two, so the keys that move are the upper half
    of one shard, about half of a tenth here. Nothing else even notices: every other shard's
    keys have unchanged placements by construction, not by luck.
    """
    keys = sorted(_keys())
    boundaries = [keys[at] for at in range(2000, 20000, 2000)]
    before = Ranges(boundaries=boundaries)
    after = before.split(keys[1000])
    fraction = moved(before, after, keys)
    return 0.02 < fraction < 0.08


@functools.cache
def hashing_spreads_what_ranges_concentrate() -> bool:
    """The hash layout sits at a spread of 1.07 and the range layout at 6.3.

    The keyspace has twenty tenants whose sizes follow a square law, which is what tenant
    populations look like, and range sharding by prefix puts whole tenants on single shards,
    so the fullest shard holds six times the emptiest. Hashing splits every tenant across
    every shard and evens out anything. The first draft of this keyspace drew tenants
    uniformly, and the skew failed to appear, because a skew has to be in the data before a
    layout can amplify it.

    The skew is not an artefact to fix, it is the price of locality: the same contiguity that
    makes a tenant's scan touch one shard makes a tenant's bulk touch one shard.
    """
    keys = _keys()
    ordered = sorted(keys)
    boundaries = [ordered[at] for at in range(2000, 20000, 2000)]
    ranged = Ranges(boundaries=boundaries)
    hashed = Modulo(shards=10)
    even_hash = spread(hashed, keys) < 1.3
    ranged_layout = Ranges(boundaries=[f"tenant:{at:03d}".encode() for at in range(2, 20, 2)])
    skewed = spread(ranged_layout, keys) > 1.5
    return even_hash and skewed and ranged.shards == 10


@functools.cache
def a_scan_touches_one_range_shard_and_every_hash_shard() -> bool:
    """A tenant's scan is one shard under ranges and all ten under hashing.

    The count is the whole difference between the layouts from a query's point of view. A
    range layout with tenant boundaries answers a tenant scan from one shard; the hash layouts
    scatter the tenant everywhere, so the scan is a fan out and a merge, and its latency is
    the slowest shard's.
    """
    keys = [key for key in _keys() if key.startswith(b"tenant:007:")]
    ranged = Ranges(boundaries=[f"tenant:{at:03d}".encode() for at in range(2, 20, 2)])
    hashed = Modulo(shards=10)
    range_shards = {ranged.place(key) for key in keys}
    hash_shards = {hashed.place(key) for key in keys}
    return len(range_shards) == 1 and len(hash_shards) == 10


def compare_the_growth(shards: int = 9) -> list[dict]:
    """One row per layout, the fraction moved by adding a shard."""
    keys = _keys()
    ordered = sorted(keys)
    step = len(ordered) // (shards + 1)
    boundaries = [ordered[at] for at in range(step, len(ordered) - 1, step)][: shards - 1]
    ranged = Ranges(boundaries=boundaries)
    return [
        {
            "layout": "modulo",
            "moved": moved(Modulo(shards=shards), Modulo(shards=shards + 1), keys),
        },
        {"layout": "ring", "moved": moved(Ring(shards=shards), Ring(shards=shards + 1), keys)},
        {
            "layout": "range_split",
            "moved": moved(ranged, ranged.split(ordered[step // 2]), keys),
        },
    ]


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "modulo_moves_everything": adding_a_shard_to_a_modulo_cluster_moves_nearly_everything(),
        "the_ring_moves_a_tenth": a_ring_moves_only_the_new_shards_share(),
        "a_split_moves_half_a_shard": a_range_split_moves_half_of_one_shard(),
        "hashing_spreads": hashing_spreads_what_ranges_concentrate(),
        "scans_pick_their_layout": a_scan_touches_one_range_shard_and_every_hash_shard(),
    }
