from __future__ import annotations

import functools
import hashlib
import random
from dataclasses import dataclass, field

from store.bloom import build as build_bloom
from store.errors import ConfigError, TooLarge

# The cuckoo filter: what a bloom filter cannot do, bought with what it cannot lose.
#
# A bloom filter cannot delete. Clearing a bit clears it for every key that set it, so a store
# whose files come and go either rebuilds its filters or lets them rot. The cuckoo filter
# stores a short fingerprint of each key in one of two buckets, and a fingerprint can be
# removed, so deletion works. The price has a shape worth measuring: inserts can fail. When
# both buckets are full the filter evicts a fingerprint to its alternate bucket, that eviction
# can cascade, and past about 95 percent occupancy the cascades stop terminating and the
# filter refuses the insert. A bloom filter never refuses, it just lies more.
#
# The alternate bucket trick is the part worth understanding: the second bucket's index is the
# first XORed with the fingerprint's hash, an involution, so either bucket can compute the
# other without knowing which one it is. That is what lets an eviction move a fingerprint
# without consulting the key, which is long gone.

BUCKET_SLOTS = 4
FINGERPRINT_BITS = 12
MAX_KICKS = 500


def _fingerprint(key: bytes) -> int:
    """A short nonzero tag for a key."""
    digest = hashlib.blake2b(key, digest_size=8).digest()
    tag = int.from_bytes(digest, "big") & ((1 << FINGERPRINT_BITS) - 1)
    return tag or 1


def _index(key: bytes, buckets: int) -> int:
    """The primary bucket."""
    digest = hashlib.blake2b(key, digest_size=8, salt=b"index").digest()
    return int.from_bytes(digest, "big") % buckets


def _alternate(index: int, tag: int, buckets: int) -> int:
    """The other bucket, computable from either side."""
    spread = int.from_bytes(
        hashlib.blake2b(tag.to_bytes(2, "big"), digest_size=8).digest(), "big"
    )
    return (index ^ spread) % buckets


@dataclass
class Cuckoo:
    """A cuckoo filter over fixed buckets of four slots."""

    buckets: int
    held: list[list[int]] = field(default_factory=list)
    keys: int = field(default=0)
    kicks: int = field(default=0)
    refused: int = field(default=0)
    source: random.Random = field(default=None)

    def __post_init__(self) -> None:
        if self.buckets < 1:
            raise ConfigError(f"{self.buckets} is not a bucket count")
        if self.buckets & (self.buckets - 1):
            raise ConfigError(f"{self.buckets} buckets is not a power of two")
        if not self.held:
            self.held = [[] for _ in range(self.buckets)]
        self.source = random.Random(self.buckets)

    @property
    def slots(self) -> int:
        """Total capacity in fingerprints."""
        return self.buckets * BUCKET_SLOTS

    @property
    def occupancy(self) -> float:
        """How full the filter is."""
        return round(self.keys / max(self.slots, 1), 4)

    @property
    def nbytes(self) -> int:
        """What the filter costs, fingerprint bits per slot."""
        return (self.slots * FINGERPRINT_BITS + 7) // 8

    def add(self, key: bytes) -> None:
        """Insert a key, evicting as needed, refusing when the cascade will not end."""
        tag = _fingerprint(key)
        first = _index(key, self.buckets)
        second = _alternate(first, tag, self.buckets)
        for at in (first, second):
            if len(self.held[at]) < BUCKET_SLOTS:
                self.held[at].append(tag)
                self.keys += 1
                return
        at = self.source.choice((first, second))
        for _ in range(MAX_KICKS):
            slot = self.source.randrange(BUCKET_SLOTS)
            tag, self.held[at][slot] = self.held[at][slot], tag
            self.kicks += 1
            at = _alternate(at, tag, self.buckets)
            if len(self.held[at]) < BUCKET_SLOTS:
                self.held[at].append(tag)
                self.keys += 1
                return
        self.refused += 1
        raise TooLarge(f"the filter is {self.occupancy:.0%} full and the cascade did not end")

    def might_contain(self, key: bytes) -> bool:
        """No means no; yes means probably."""
        tag = _fingerprint(key)
        first = _index(key, self.buckets)
        second = _alternate(first, tag, self.buckets)
        return tag in self.held[first] or tag in self.held[second]

    def remove(self, key: bytes) -> bool:
        """Take one fingerprint out, which a bloom filter cannot do."""
        tag = _fingerprint(key)
        first = _index(key, self.buckets)
        second = _alternate(first, tag, self.buckets)
        for at in (first, second):
            if tag in self.held[at]:
                self.held[at].remove(tag)
                self.keys -= 1
                return True
        return False

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "buckets": self.buckets,
            "keys": self.keys,
            "occupancy": self.occupancy,
            "bytes": self.nbytes,
            "kicks": self.kicks,
            "refused": self.refused,
        }


def _keys(count: int, prefix: str = "k") -> list[bytes]:
    """Distinct keys."""
    return [f"{prefix}{at:08d}".encode() for at in range(count)]


@functools.cache
def no_false_negatives_below_the_refusal_point() -> bool:
    """Every inserted key answers yes, through all the evictions.

    The eviction cascade moves fingerprints between their two homes but never drops one, so
    membership survives any amount of shuffling. This is the invariant deletion support must
    not break, and it is checked after thousands of kicks have happened.
    """
    made = Cuckoo(buckets=4096)
    keys = _keys(14000)
    for key in keys:
        made.add(key)
    return made.kicks > 0 and all(made.might_contain(key) for key in keys)


@functools.cache
def deletion_works_and_bloom_has_nothing_to_compare() -> bool:
    """Remove half the keys and the removed half answers no again.

    The removed keys answer no except for fingerprint collisions, which are the same false
    positive rate the filter always had. A bloom filter's row in this table simply does not
    exist, which is the entire reason to pay the cuckoo's insert complexity.
    """
    made = Cuckoo(buckets=4096)
    keys = _keys(10000)
    for key in keys:
        made.add(key)
    for key in keys[:5000]:
        made.remove(key)
    stayed = all(made.might_contain(key) for key in keys[5000:])
    ghosts = sum(1 for key in keys[:5000] if made.might_contain(key))
    return stayed and ghosts < 100


@functools.cache
def inserts_start_failing_near_ninety_five_percent() -> bool:
    """The filter accepts to about 95 percent occupancy and then refuses.

    The refusal point is a property of bucket geometry, four slots and two choices, and it
    arrives abruptly: the kick counts climb as occupancy passes ninety percent and then a
    cascade fails to terminate. A store sizing cuckoo filters plans for the refusal, where a
    bloom filter would have quietly degraded instead. Failing loudly at a known line beats
    lying on a curve, but only if the sizing knows the line.
    """
    made = Cuckoo(buckets=1024)
    landed = 0
    try:
        for key in _keys(made.slots + 100):
            made.add(key)
            landed += 1
    except TooLarge:
        pass
    return 0.93 < landed / made.slots < 1.0


@functools.cache
def the_false_positive_rate_matches_the_fingerprint_width() -> bool:
    """Twelve bit fingerprints give about eight collisions per hundred thousand probes.

    The expected rate is 2 times bucket slots over 2 to the fingerprint bits, eight slots
    checked against a twelve bit space, about 0.2 percent at high occupancy. Measured with a
    hundred thousand absent probes the rate sits in that band, and the knob is the width: two
    more bits, a quarter the lies, a sixth more space.
    """
    made = Cuckoo(buckets=4096)
    for key in _keys(12000):
        made.add(key)
    probes = _keys(100000, "absent")
    lies = sum(1 for key in probes if made.might_contain(key))
    rate = lies / len(probes)
    return 0.0002 < rate < 0.01


@functools.cache
def the_involution_holds_only_at_powers_of_two() -> bool:
    """XOR then mod is its own inverse at 4096 buckets and is not at 4000.

    This was found the expensive way. The filter ran at 4000 buckets, the involution test ran
    at 4096, and the test passed while the filter lost 419 of 14,000 keys: every fingerprint
    evicted to a bucket whose XOR partner did not survive the modulo became unfindable, a
    false negative from the one structure whose only hard promise is no false negatives.

    Measured directly: at 4096 buckets ten thousand random tag and index pairs all map back
    to themselves, and at 4000 buckets most do not. The constructor now refuses bucket counts
    that are not powers of two, because the involution is not a property of XOR, it is a
    property of XOR on a space the modulo does not cut.
    """
    source = random.Random(5)
    broken = 0
    for _ in range(10000):
        tag = source.randrange(1, 1 << FINGERPRINT_BITS)
        start = source.randrange(4000)
        away = _alternate(start, tag, 4096)
        back = _alternate(away, tag, 4096)
        if back != start:
            return False
        wrong_away = _alternate(start, tag, 4000)
        wrong_back = _alternate(wrong_away, tag, 4000)
        if wrong_back != start:
            broken += 1
    return broken > 5000


def compare_with_bloom(keys: int = 12000, probes: int = 50000) -> list[dict]:
    """The two filters on the same keys, sizes and rates side by side."""
    inserted = _keys(keys)
    absent = _keys(probes, "absent")
    cuckoo = Cuckoo(buckets=4096)
    for key in inserted:
        cuckoo.add(key)
    bloom = build_bloom(inserted)
    cuckoo_lies = sum(1 for key in absent if cuckoo.might_contain(key))
    bloom_lies = sum(1 for key in absent if bloom.might_contain(key))
    return [
        {
            "filter": "cuckoo",
            "bytes": cuckoo.nbytes,
            "false_rate": round(cuckoo_lies / probes, 5),
            "deletes": True,
        },
        {
            "filter": "bloom",
            "bytes": len(bloom.bits),
            "false_rate": round(bloom_lies / probes, 5),
            "deletes": False,
        },
    ]


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "no_false_negatives": no_false_negatives_below_the_refusal_point(),
        "deletion_works": deletion_works_and_bloom_has_nothing_to_compare(),
        "refusal_near_95": inserts_start_failing_near_ninety_five_percent(),
        "the_rate_is_the_width": the_false_positive_rate_matches_the_fingerprint_width(),
        "the_involution_needs_a_power_of_two": the_involution_holds_only_at_powers_of_two(),
    }
