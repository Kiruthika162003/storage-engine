from __future__ import annotations

import functools
import random
from collections import Counter
from dataclasses import dataclass, field

from store.errors import ConfigError

# Heavy hitters in fixed space: the counter that forgets fairly.
#
# The admission module's sketch estimates any key's frequency; this module answers a narrower
# question with a stronger guarantee: which keys are hot, exactly the hot key list an
# operator wants when one tenant is eating the store. Misra-Gries keeps k counters. A tracked
# key increments its counter; an untracked key takes a free slot, or, when there is none,
# every counter decrements instead, one shared toll charged to all. The guarantee that makes
# it useful: a counter undercounts its key by at most n over k plus one, so any key with
# frequency above that bound is guaranteed present, no matter how adversarial the stream.

CAPACITY = 8


@dataclass
class Summary:
    """The k counters and the accounting behind the guarantee."""

    capacity: int = field(default=CAPACITY)
    counts: dict[bytes, int] = field(default_factory=dict)
    seen: int = field(default=0)
    decrements: int = field(default=0)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ConfigError(f"{self.capacity} is not a capacity")

    def note(self, key: bytes) -> None:
        """One sighting."""
        self.seen += 1
        if key in self.counts:
            self.counts[key] += 1
            return
        if len(self.counts) < self.capacity:
            self.counts[key] = 1
            return
        self.decrements += 1
        for held in list(self.counts):
            self.counts[held] -= 1
            if self.counts[held] == 0:
                del self.counts[held]

    def candidates(self) -> dict[bytes, int]:
        """The tracked keys and their lower-bound counts."""
        return dict(self.counts)

    @property
    def bound(self) -> float:
        """The largest undercount any key can have suffered."""
        return self.seen / (self.capacity + 1)

    def certainly_above(self, threshold: int) -> set[bytes]:
        """Keys whose true count provably exceeds the threshold."""
        return {
            key for key, count in self.counts.items() if count > threshold
        }


@functools.cache
def _skewed_stream(count: int = 50000, seed: int = 163) -> tuple[bytes, ...]:
    """A stream where a few keys dominate, the shape hot key problems have."""
    source = random.Random(seed)
    made = []
    for _ in range(count):
        if source.random() < 0.6:
            made.append(f"hot:{source.randrange(4)}".encode())
        else:
            made.append(f"cold:{source.randrange(20000):06d}".encode())
    return tuple(made)


@functools.cache
def every_truly_heavy_key_is_present() -> bool:
    """The four keys above the n over k+1 bound are all tracked, guaranteed and observed.

    Fifty thousand sightings through eight counters: the bound is 5,556, the four hot keys
    each true well above it, and all four sit in the summary. The guarantee is what
    separates this from a cache of recent keys, which an adversarial stream can flush at
    will; no stream can push a key with frequency above the bound out of a Misra-Gries
    summary, and that is a theorem wearing eight counters.
    """
    summary = Summary()
    truth = Counter()
    for key in _skewed_stream():
        summary.note(key)
        truth[key] += 1
    heavy = {key for key, count in truth.items() if count > summary.bound}
    return len(heavy) == 4 and heavy <= set(summary.candidates())


@functools.cache
def tracked_counts_undercount_within_the_bound() -> bool:
    """Every tracked count sits at or below the truth, within n over k+1 of it.

    The decrement tolls make every counter a lower bound, and the bound caps the toll. Both
    sides checked for every tracked key: no overcount ever, no undercount past the bound.
    The direction matters the way count-min's did, mirrored: count-min never undercounts,
    Misra-Gries never overcounts, and which lie is affordable depends on what the number
    feeds.
    """
    summary = Summary()
    truth = Counter()
    for key in _skewed_stream():
        summary.note(key)
        truth[key] += 1
    for key, count in summary.candidates().items():
        if count > truth[key]:
            return False
        if truth[key] - count > summary.bound:
            return False
    return True


@functools.cache
def a_uniform_stream_yields_no_certainties() -> bool:
    """When nothing is hot, the summary certifies nothing, which is the right answer.

    Twenty thousand distinct keys evenly spread: whatever eight keys happen to be tracked,
    none clears the certainty threshold, and the certainly_above set at the bound is empty.
    A hot key detector that names hot keys in a uniform stream is a random number generator
    with a dashboard.
    """
    source = random.Random(167)
    summary = Summary()
    for _ in range(50000):
        summary.note(f"k{source.randrange(20000):06d}".encode())
    return summary.certainly_above(int(summary.bound)) == set()


@functools.cache
def the_toll_is_rare_on_skewed_streams() -> bool:
    """Decrements happen on under a third of sightings even with the table always full.

    Hot keys are almost always already tracked, so the toll only fires when a cold key
    arrives with the table full. The decrement count is the summary's whole running cost
    beyond a dictionary probe, and its rarity on the streams that matter is why the
    structure is cheap exactly when it is useful.
    """
    summary = Summary()
    for key in _skewed_stream():
        summary.note(key)
    return summary.decrements < summary.seen * 0.35


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "heavy_keys_are_present": every_truly_heavy_key_is_present(),
        "counts_undercount_boundedly": tracked_counts_undercount_within_the_bound(),
        "uniform_streams_certify_nothing": a_uniform_stream_yields_no_certainties(),
        "the_toll_is_rare": the_toll_is_rare_on_skewed_streams(),
    }
