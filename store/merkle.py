"""Merkle trees over key ranges: finding the differing keys in log probes.

Two stores that should agree hold ten thousand keys each. Comparing them
key by key costs ten thousand probes; comparing hash trees costs a probe
per node along the paths to the differences. The measurements count the
probes for honest replicas, one divergent key, and a hundred, and show
the cost is set by the differences, not the data.
"""

from __future__ import annotations

import functools
import hashlib
import random
from dataclasses import dataclass, field

FANOUT = 16
KEYS = 10000


def _digest(parts: list[bytes]) -> bytes:
    joined = hashlib.sha256()
    for part in parts:
        joined.update(part)
    return joined.digest()


@dataclass
class Tree:
    """Hashes over sorted keys, grouped FANOUT-wise into levels."""

    keys: list[bytes]
    values: dict[bytes, bytes]
    levels: list[list[bytes]] = field(default_factory=list)

    @classmethod
    def over(cls, values: dict[bytes, bytes]) -> Tree:
        keys = sorted(values)
        tree = cls(keys=keys, values=values)
        level = [
            _digest([key, values[key]]) for key in keys
        ]
        tree.levels.append(level)
        while len(level) > 1:
            level = [
                _digest(level[at : at + FANOUT])
                for at in range(0, len(level), FANOUT)
            ]
            tree.levels.append(level)
        return tree

    def root(self) -> bytes:
        return self.levels[-1][0]


def diff(one: Tree, two: Tree) -> tuple[list[bytes], int]:
    """Keys whose hashes differ, and the node probes both sides spent."""
    probes = 0
    depth = len(one.levels) - 1
    suspects = [0]
    while depth > 0:
        below = []
        for at in suspects:
            probes += 2
            if one.levels[depth][at] == two.levels[depth][at]:
                continue
            start = at * FANOUT
            width = len(one.levels[depth - 1])
            below.extend(range(start, min(start + FANOUT, width)))
        suspects = below
        depth = 0 if not suspects else depth - 1
    differing = []
    for at in suspects:
        probes += 2
        if one.levels[0][at] != two.levels[0][at]:
            differing.append(one.keys[at])
    return differing, probes


def _replica(seed: int) -> dict[bytes, bytes]:
    source = random.Random(seed)
    return {
        f"key:{number:05d}".encode(): source.randbytes(8) for number in range(KEYS)
    }


@functools.cache
def _drifted(count: int) -> tuple[Tree, Tree, list[bytes]]:
    base = _replica(7)
    source = random.Random(3)
    bad = source.sample(sorted(base), count) if count else []
    drifted = dict(base)
    for key in bad:
        drifted[key] = b"corrupt!"
    return Tree.over(base), Tree.over(drifted), bad


@functools.cache
def honest_replicas_agree_in_one_probe_pair() -> bool:
    """Ten thousand equal keys are proven equal by comparing two roots.

    The usual case is the cheap case: replicas that agree spend two
    probes total, and the whole comparison budget is reserved for the
    day something is actually wrong.
    """
    one, two, _ = _drifted(0)
    differing, probes = diff(one, two)
    return differing == [] and probes == 2 and one.root() == two.root()


@functools.cache
def one_bad_key_costs_104_probes_not_20000() -> bool:
    """A single corrupted value is located with 104 probes, 192x fewer.

    The mismatch propagates up to the root, and the search walks back
    down comparing every level's children along one path: 2 at the root,
    6 at the next level, then three sixteen-wide sweeps. Key by key
    comparison would spend two probes per key on the 9999 innocents.
    """
    one, three, bad = _drifted(1)
    differing, probes = diff(one, three)
    return differing == bad and probes == 104


@functools.cache
def a_hundred_bad_keys_share_their_paths() -> bool:
    """100 corruptions cost 4184 probes, 40 per key, not 104 per key.

    Divergent keys scattered over the tree share upper levels: the
    hundred paths cross the same root and mostly the same second level,
    so the cost per difference falls as differences grow. The tree bills
    by the difference, with a volume discount, never by the data.
    """
    one, four, bad = _drifted(100)
    differing, probes = diff(one, four)
    return sorted(differing) == sorted(bad) and probes == 4184 and probes < 104 * 100


@functools.cache
def the_diff_names_exactly_the_guilty_keys() -> bool:
    """Every injected corruption is reported and nothing else is.

    For 1 and for 100 injected differences the reported key sets equal
    the injected sets exactly, which is what lets a repair copy only what
    the diff names instead of resynchronising the world.
    """
    for count in (1, 100):
        one, other, bad = _drifted(count)
        differing, _ = diff(one, other)
        if sorted(differing) != sorted(bad):
            return False
    return True


@functools.cache
def summarise() -> dict:
    return {
        "module": "store.merkle",
        "honest_replicas_agree_in_one_probe_pair": (
            honest_replicas_agree_in_one_probe_pair()
        ),
        "one_bad_key_costs_104_probes_not_20000": (
            one_bad_key_costs_104_probes_not_20000()
        ),
        "a_hundred_bad_keys_share_their_paths": a_hundred_bad_keys_share_their_paths(),
        "the_diff_names_exactly_the_guilty_keys": (
            the_diff_names_exactly_the_guilty_keys()
        ),
    }
