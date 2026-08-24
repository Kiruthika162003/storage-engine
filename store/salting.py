"""Key salting for hot partitions: writes spread, reads fan out, both counted.

One counter key takes every increment and its shard melts. Salting splits
the key into N salted copies, each landing on a different shard, and the
write storm spreads. The bill arrives at read time: the true value is now
the sum of N keys, and every read must visit all of them. The measurements
put numbers on both sides and find where the trade stops paying.
"""

from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

SHARDS = 8


@dataclass
class Cluster:
    shards: list[dict[bytes, int]] = field(
        default_factory=lambda: [{} for _ in range(SHARDS)]
    )
    writes: list[int] = field(default_factory=lambda: [0] * SHARDS)
    reads: list[int] = field(default_factory=lambda: [0] * SHARDS)

    def _place(self, key: bytes) -> int:
        return sum(key) % SHARDS

    def bump(self, key: bytes) -> None:
        shard = self._place(key)
        self.writes[shard] += 1
        self.shards[shard][key] = self.shards[shard].get(key, 0) + 1

    def read(self, key: bytes) -> int:
        shard = self._place(key)
        self.reads[shard] += 1
        return self.shards[shard].get(key, 0)

    def hottest_write_share(self) -> float:
        return max(self.writes) / max(1, sum(self.writes))


def salted(key: bytes, salt: int) -> bytes:
    return key + b"#" + str(salt).encode()


def bump_salted(cluster: Cluster, key: bytes, salts: int, source: random.Random) -> None:
    cluster.bump(salted(key, source.randrange(salts)))


def read_salted(cluster: Cluster, key: bytes, salts: int) -> int:
    return sum(cluster.read(salted(key, salt)) for salt in range(salts))


def _storm(salts: int, seed: int = 5) -> Cluster:
    source = random.Random(seed)
    cluster = Cluster()
    for _ in range(8000):
        if salts:
            bump_salted(cluster, b"views:home", salts, source)
        else:
            cluster.bump(b"views:home")
        if source.random() < 0.25:
            cluster.bump(f"views:page{source.randrange(200):03d}".encode())
    return cluster


@functools.cache
def one_hot_key_melts_one_shard() -> bool:
    """The unsalted counter sends 81.9 percent of the storm to one shard.

    The other seven shards idle at three percent each while shard one
    absorbs 8244 of 10064 writes. Hashing spreads keys, not traffic: a
    single hot key defeats any placement that keys on the key.
    """
    cluster = _storm(0)
    return 0.81 < cluster.hottest_write_share() < 0.83


@functools.cache
def eight_salts_flatten_the_storm() -> bool:
    """Eight salts drop the hottest shard from 81.9 to 13 percent.

    Each salted copy hashes somewhere else, and the storm lands almost
    uniformly, within a point of the 12.5 percent floor an eight shard
    cluster can ever reach.
    """
    cluster = _storm(8)
    return cluster.hottest_write_share() < 0.14


@functools.cache
def the_read_pays_one_key_per_salt_and_stays_correct() -> bool:
    """Reading the salted counter costs 8 reads and still answers 8000.

    The write side's relief is the read side's fan-out: every read must
    visit every salt, pay a read per shard, and sum. Salting suits
    counters written constantly and read rarely, not the reverse.
    """
    cluster = _storm(8)
    value = read_salted(cluster, b"views:home", 8)
    return value == 8000 and sum(cluster.reads) == 8


@functools.cache
def salts_beyond_shards_reconcentrate() -> bool:
    """Sixteen salts measure 17.7 percent hottest, worse than eight's 13.

    Doubling the salts past the shard count cannot spread further, and
    the salt suffixes collide unevenly under the placement hash, piling
    two salts onto some shards. The ceiling on spreading is the shard
    count, and extra salts only add read fan-out: 16 reads now, for a
    hotter shard than 8 salts bought.
    """
    eight = _storm(8)
    sixteen = _storm(16)
    return (
        sixteen.hottest_write_share() > eight.hottest_write_share() + 0.03
        and read_salted(sixteen, b"views:home", 16) == 8000
    )


@functools.cache
def summarise() -> dict:
    return {
        "module": "store.salting",
        "one_hot_key_melts_one_shard": one_hot_key_melts_one_shard(),
        "eight_salts_flatten_the_storm": eight_salts_flatten_the_storm(),
        "the_read_pays_one_key_per_salt_and_stays_correct": (
            the_read_pays_one_key_per_salt_and_stays_correct()
        ),
        "salts_beyond_shards_reconcentrate": salts_beyond_shards_reconcentrate(),
    }
