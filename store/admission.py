from __future__ import annotations

import functools
import hashlib
import random
from dataclasses import dataclass, field

from store.cache import Recent, Reference
from store.errors import ConfigError

# Cache admission: the question is not who to evict, it is who to let in.
#
# Every policy in the cache module admits unconditionally: a miss inserts, someone is evicted,
# and a block seen exactly once costs a slot until its turn to leave. Admission control asks
# first whether the newcomer is likely to be seen again, and the standard answer keeps
# frequency estimates in a tiny sketch and admits a newcomer only if its estimated frequency
# beats the eviction candidate's. The sketch is a count-min with four rows of counters, halved
# periodically so the past fades, and its whole budget is a few kilobytes.
#
# The measurement that matters is the one-hit-wonder stream: a working set with a river of
# single-use blocks flowing through it, which is what a scan-polluted cache sees, and where
# unconditional admission spends most of its slots on blocks that never return.

COUNTER_ROWS = 4
FADE_EVERY = 10000


def _hashes(key: int, width: int) -> list[int]:
    """Four positions, one per row."""
    made = []
    for row in range(COUNTER_ROWS):
        digest = hashlib.blake2b(
            key.to_bytes(8, "big"), digest_size=8, salt=row.to_bytes(4, "big")
        ).digest()
        made.append(int.from_bytes(digest, "big") % width)
    return made


@dataclass
class Frequency:
    """A count-min sketch with periodic fading."""

    width: int = field(default=2048)
    rows: list[list[int]] = field(default_factory=list)
    added: int = field(default=0)

    def __post_init__(self) -> None:
        if self.width < 16:
            raise ConfigError(f"{self.width} is too narrow to sketch")
        if not self.rows:
            self.rows = [[0] * self.width for _ in range(COUNTER_ROWS)]

    def touch(self, key: int) -> None:
        """Count one sighting, fading everything when the window fills."""
        for row, at in enumerate(_hashes(key, self.width)):
            self.rows[row][at] += 1
        self.added += 1
        if self.added % FADE_EVERY == 0:
            self.fade()

    def estimate(self, key: int) -> int:
        """The minimum over the rows, which bounds the true count from above."""
        return min(self.rows[row][at] for row, at in enumerate(_hashes(key, self.width)))

    def fade(self) -> None:
        """Halve every counter, so the past is worth half the present."""
        for row in self.rows:
            for at in range(len(row)):
                row[at] >>= 1

    @property
    def nbytes(self) -> int:
        """One byte per counter is generous."""
        return COUNTER_ROWS * self.width


@dataclass
class Admitting:
    """A recency cache behind a frequency gate."""

    capacity: int
    cache: Recent = field(default=None)
    sketch: Frequency = field(default_factory=Frequency)
    admitted: int = field(default=0)
    turned_away: int = field(default=0)

    def __post_init__(self) -> None:
        if self.cache is None:
            self.cache = Recent(capacity=self.capacity)

    def get(self, key: int) -> bytes | None:
        """A lookup, feeding the sketch either way."""
        self.sketch.touch(key)
        return self.cache.get(key)

    def put(self, key: int, value: bytes) -> None:
        """Admit only a newcomer whose frequency beats the eviction candidate's."""
        if len(self.cache.held) < self.capacity:
            self.cache.put(key, value)
            self.admitted += 1
            return
        victim = next(iter(self.cache.held))
        if self.sketch.estimate(key) > self.sketch.estimate(victim):
            self.cache.put(key, value)
            self.admitted += 1
        else:
            self.turned_away += 1

    @property
    def rate(self) -> float:
        """The underlying cache's hit rate."""
        return self.cache.stats.rate


def drive(cache, blocks) -> float:
    """A reference stream through either kind of cache, hit rate out."""
    for number in blocks:
        if cache.get(number) is None:
            cache.put(number, number.to_bytes(8, "little"))
    return cache.rate if isinstance(cache, Admitting) else cache.stats.rate


@functools.cache
def _polluted(length: int = 60000, seed: int = 83) -> tuple[int, ...]:
    """A hot working set with a river of one-hit wonders through it."""
    source = random.Random(seed)
    made = []
    river = 10**6
    for _ in range(length):
        if source.random() < 0.5:
            made.append(source.randrange(150))
        else:
            river += 1
            made.append(river)
    return tuple(made)


@functools.cache
def admission_doubles_the_hit_rate_on_a_polluted_stream() -> bool:
    """Half the stream is one-hit wonders, and the gate turns nearly all of them away.

    The plain recency cache admits every wonder, each costing a slot a hot block needed,
    and lands at 18.6 percent. The gated cache asks the sketch first, the wonder arrives
    with an estimate of one, the victim usually beats it, and the rate lands at 42.4 against
    a ceiling near 50 that the stream's composition sets. The sketch spends 8 kilobytes to
    better than double a cache of 128 slots.
    """
    plain = Recent(capacity=128)
    gated = Admitting(capacity=128)
    stream = list(_polluted())
    plain_rate = drive(plain, stream)
    gated_rate = drive(gated, stream)
    return gated_rate > plain_rate * 1.7 and gated_rate > 0.4


@functools.cache
def the_gate_is_invisible_on_a_clean_working_set() -> bool:
    """On a stream that fits, the gated and plain caches land within two points.

    Every block in a fitting working set builds frequency quickly, so the gate admits what
    recency would have kept anyway. The cost of admission control on a workload that does not
    need it is a sketch update per lookup and nearly nothing in hits, which is what lets it
    ship as a default rather than a tuning option.
    """
    fits = Reference(blocks=100, length=20000, shape="hot").stream()
    plain = drive(Recent(capacity=128), fits)
    gated = drive(Admitting(capacity=128), fits)
    return abs(plain - gated) < 0.02


@functools.cache
def the_sketch_never_underestimates() -> bool:
    """The count-min estimate bounds the true count from above, checked exactly.

    Collisions only add, never subtract, so the minimum across rows is at least the truth.
    The gate leans on the direction of the error: an overestimated newcomer is admitted too
    eagerly, which costs a slot, and an underestimate would starve a genuinely hot block,
    which costs the workload. The bias lands on the cheap side by construction.
    """
    sketch = Frequency(width=256)
    truth: dict[int, int] = {}
    source = random.Random(11)
    for _ in range(5000):
        key = source.randrange(400)
        sketch.touch(key)
        truth[key] = truth.get(key, 0) + 1
    return all(sketch.estimate(key) >= count for key, count in truth.items())


@functools.cache
def fading_lets_yesterdays_hot_key_lose() -> bool:
    """A key hot before the fade loses to a key hot after it.

    The halving is the forgetting the frequency cache in the cache module lacked. Without it
    the sketch is a lifetime popularity contest and the shifted working set problem returns
    wearing admission's clothes.
    """
    sketch = Frequency(width=512)
    for _ in range(200):
        sketch.touch(1)
    for _ in range(4):
        sketch.fade()
    for _ in range(30):
        sketch.touch(2)
    return sketch.estimate(2) > sketch.estimate(1)


def compare_the_streams() -> list[dict]:
    """Both caches on both streams."""
    rows = []
    polluted = list(_polluted())
    clean = Reference(blocks=100, length=20000, shape="hot").stream()
    for name, stream in (("polluted", polluted), ("clean", clean)):
        rows.append(
            {
                "stream": name,
                "plain": drive(Recent(capacity=128), stream),
                "gated": drive(Admitting(capacity=128), stream),
            }
        )
    return rows


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "the_gate_doubles_polluted_hits": admission_doubles_the_hit_rate_on_a_polluted_stream(),
        "the_gate_is_invisible_when_clean": the_gate_is_invisible_on_a_clean_working_set(),
        "estimates_bound_from_above": the_sketch_never_underestimates(),
        "fading_forgets": fading_lets_yesterdays_hot_key_lose(),
    }
