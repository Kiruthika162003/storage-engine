from __future__ import annotations

import functools
import hashlib
import math
import random
from dataclasses import dataclass, field

from store.errors import ConfigError

# Counting distinct keys in a kilobyte, and what the estimate is actually for.
#
# A compaction planner wants to know how many distinct keys two runs share before it merges
# them, because the overlap decides whether the merge is worth scheduling. Exact answers need
# a set the size of the keys. The hyperloglog gets within a few percent in a fixed few
# kilobytes, by a trick that deserves plain statement: the maximum number of leading zero bits
# seen in the hashes of a set is a logarithm of the set's size, because a hash with r leading
# zeros is a one in 2^r event. One maximum is a terrible estimator, so the sketch keeps
# thousands of them in registers indexed by the hash's first bits and averages them
# harmonically, which tames the outliers that ruin the plain mean.

REGISTER_BITS = 11


def _hash(key: bytes) -> int:
    """Sixty four stable bits."""
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")


def _alpha(registers: int) -> float:
    """The bias constant for the harmonic mean, from the paper."""
    if registers >= 128:
        return 0.7213 / (1 + 1.079 / registers)
    if registers == 64:
        return 0.709
    if registers == 32:
        return 0.697
    return 0.673


@dataclass
class Sketch:
    """A hyperloglog with 2^REGISTER_BITS registers."""

    register_bits: int = field(default=REGISTER_BITS)
    registers: list[int] = field(default_factory=list)
    added: int = field(default=0)

    def __post_init__(self) -> None:
        if not 4 <= self.register_bits <= 16:
            raise ConfigError(f"{self.register_bits} register bits is outside 4 to 16")
        if not self.registers:
            self.registers = [0] * (1 << self.register_bits)

    @property
    def nbytes(self) -> int:
        """One byte per register is generous and simple."""
        return len(self.registers)

    def add(self, key: bytes) -> None:
        """Fold one key in."""
        hashed = _hash(key)
        register = hashed >> (64 - self.register_bits)
        rest = hashed & ((1 << (64 - self.register_bits)) - 1)
        rank = (64 - self.register_bits) - rest.bit_length() + 1
        self.registers[register] = max(self.registers[register], rank)
        self.added += 1

    def estimate(self) -> int:
        """The distinct count, harmonically averaged, small range corrected."""
        registers = len(self.registers)
        total = sum(2.0**-value for value in self.registers)
        raw = _alpha(registers) * registers * registers / total
        if raw <= 2.5 * registers:
            zeros = self.registers.count(0)
            if zeros:
                return round(registers * math.log(registers / zeros))
        return round(raw)

    def merge(self, other: Sketch) -> Sketch:
        """The sketch of the union, which is registerwise maximum."""
        if self.register_bits != other.register_bits:
            raise ConfigError("sketches of different widths do not merge")
        made = Sketch(register_bits=self.register_bits)
        pairs = zip(self.registers, other.registers, strict=True)
        made.registers = [max(a, b) for a, b in pairs]
        made.added = self.added + other.added
        return made

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "registers": len(self.registers),
            "bytes": self.nbytes,
            "added": self.added,
            "estimate": self.estimate(),
        }


def exact_bytes(count: int, key_bytes: int = 17) -> int:
    """What the exact answer costs: a set of the keys."""
    return count * key_bytes


@functools.cache
def _sketched(count: int, seed: int = 0) -> Sketch:
    """A sketch over that many distinct keys."""
    made = Sketch()
    for at in range(count):
        made.add(f"s{seed}:k{at:09d}".encode())
    return made


@functools.cache
def the_error_sits_inside_the_promised_band() -> bool:
    """The estimate lands within a few percent at every size from 1,000 to 500,000.

    The standard error of a hyperloglog is 1.04 over the square root of the register count,
    which at 2,048 registers is 2.3 percent. Measured at five sizes across three orders of
    magnitude, the worst relative error stays inside three standard errors, and the sketch
    never grew past its 2,048 bytes while the exact set for the largest size would have taken
    8.5 megabytes.
    """
    for count in (1000, 10000, 50000, 200000, 500000):
        estimate = _sketched(count).estimate()
        error = abs(estimate - count) / count
        if error > 3 * 1.04 / math.sqrt(2048):
            return False
    return True


@functools.cache
def duplicates_do_not_move_the_estimate() -> bool:
    """A hundred thousand additions of ten thousand keys estimate ten thousand.

    The register maximum only rises when a new hash beats it, and the same key hashes the
    same way forever, so repetition is invisible. This is what makes the sketch a distinct
    counter rather than a counter, and it needs saying because the two are one keystroke
    apart in a metrics pipeline.
    """
    made = Sketch()
    source = random.Random(7)
    for _ in range(100000):
        made.add(f"k{source.randrange(10000):06d}".encode())
    estimate = made.estimate()
    return abs(estimate - 10000) / 10000 < 0.1


@functools.cache
def the_union_is_free_and_the_intersection_is_arithmetic() -> bool:
    """Merging sketches gives the union's count without touching a key.

    Two sets of 30,000 keys sharing 10,000 estimate a union near 50,000 by registerwise
    maximum. The intersection falls out by inclusion exclusion, this plus that minus the
    union, near 10,000, and its error is the sum of three estimates' errors, which is why
    intersections of small overlaps come out noisy: the signal is the difference of large
    numbers.
    """
    left = Sketch()
    right = Sketch()
    for at in range(30000):
        left.add(f"L:{at:07d}".encode())
        right.add(f"L:{at:07d}".encode() if at < 10000 else f"R:{at:07d}".encode())
    union = left.merge(right).estimate()
    meet = left.estimate() + right.estimate() - union
    return abs(union - 50000) / 50000 < 0.1 and abs(meet - 10000) / 10000 < 0.35


@functools.cache
def the_sketch_is_thousands_of_times_smaller_at_the_top_size() -> bool:
    """Half a million keys: 8.5 megabytes exactly, 2,048 bytes sketched.

    The ratio grows with the set because the sketch does not, which is the whole purchase:
    the cost of an approximate answer is fixed at sizing time and the cost of an exact one is
    decided by the data.
    """
    sketch = _sketched(500000)
    return exact_bytes(500000) / sketch.nbytes > 4000


def compare_the_sizes() -> list[dict]:
    """One row per set size, exact cost against sketch cost against error."""
    rows = []
    for count in (1000, 10000, 100000, 500000):
        sketch = _sketched(count)
        estimate = sketch.estimate()
        rows.append(
            {
                "distinct": count,
                "estimate": estimate,
                "error": round(abs(estimate - count) / count, 4),
                "exact_bytes": exact_bytes(count),
                "sketch_bytes": sketch.nbytes,
            }
        )
    return rows


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "the_error_is_banded": the_error_sits_inside_the_promised_band(),
        "duplicates_are_invisible": duplicates_do_not_move_the_estimate(),
        "the_union_is_free": the_union_is_free_and_the_intersection_is_arithmetic(),
        "the_sketch_stays_small": the_sketch_is_thousands_of_times_smaller_at_the_top_size(),
    }
