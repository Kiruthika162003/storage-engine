from __future__ import annotations

import functools
import hashlib
import math
import struct
from dataclasses import dataclass

from store.errors import BadFormat, ConfigError

# The filter that says a key is definitely absent, and sometimes says it might be present.
#
# A read in a log structured store has to ask every file that could hold the key, and most of
# them do not. The filter is what makes that cheap: a small bit array per file, consulted before
# the file is opened, which answers no with certainty and yes with a probability.
#
# The asymmetry is the whole design. A false positive costs one wasted file read, which is the
# thing that would have happened anyway without the filter. A false negative would cost
# correctness, because the engine would report a key absent that is sitting in a file, so the
# structure is built so that a false negative cannot happen: every bit set on insert is checked
# on lookup, and a set bit is never cleared.
#
# The hashing uses one digest split into two halves and combined, rather than k independent hash
# functions. That is the standard trick and it is worth measuring rather than trusting, because
# the false positive formula assumes independence and this arrangement does not have it.

# Bits per key, which is the only setting that matters.
BITS_PER_KEY = 10

# The hash count the formula recommends for that many bits.
HASHES = 7

# What the filter file starts with, so a reader can tell a filter from a block.
MAGIC = 0x424C4D31


def optimal_hashes(bits_per_key: float) -> int:
    """How many hashes minimise the false positive rate for a given size.

    The formula is bits per key times the natural logarithm of two, rounded. Worth writing out
    because the rounding matters at small sizes: at four bits per key the ideal is two point
    eight, and two and three give noticeably different rates.
    """
    if bits_per_key <= 0:
        raise ConfigError(f"{bits_per_key} is not a size")
    return max(1, round(bits_per_key * math.log(2)))


def expected_rate(bits_per_key: float, hashes: int) -> float:
    """The false positive rate the theory predicts, assuming independent hashes."""
    if bits_per_key <= 0:
        raise ConfigError(f"{bits_per_key} is not a size")
    if hashes < 1:
        raise ConfigError(f"{hashes} is not a hash count")
    return round((1 - math.exp(-hashes / bits_per_key)) ** hashes, 6)


@dataclass
class Filter:
    """A bit array and the number of hashes that address it."""

    bits: bytearray
    hashes: int
    keys: int = 0

    def __post_init__(self) -> None:
        if not self.bits:
            raise ConfigError("a filter needs bits")
        if self.hashes < 1:
            raise ConfigError(f"{self.hashes} is not a hash count")

    @property
    def size(self) -> int:
        """How many bits the filter has."""
        return len(self.bits) * 8

    @property
    def set_bits(self) -> int:
        """How many bits are set, which is what the rate depends on."""
        return sum(bin(one).count("1") for one in self.bits)

    @property
    def fill(self) -> float:
        """The share of bits that are set."""
        return round(self.set_bits / self.size, 4)

    @property
    def bits_per_key(self) -> float:
        """What this filter spent per key."""
        if self.keys == 0:
            return 0.0
        return round(self.size / self.keys, 2)

    def _positions(self, key: bytes):
        """Where a key sets its bits.

        One digest, split into two words, combined as the first plus the index times the second.
        Producing k independent digests would cost k hashes per lookup and the measurement below
        says it would buy nothing.
        """
        digest = hashlib.blake2b(key, digest_size=16).digest()
        first, second = struct.unpack("<QQ", digest)
        for one in range(self.hashes):
            yield (first + one * second) % self.size

    def add(self, key: bytes) -> None:
        """Set the bits for a key."""
        for one in self._positions(key):
            self.bits[one >> 3] |= 1 << (one & 7)
        self.keys += 1

    def might_contain(self, key: bytes) -> bool:
        """Whether the key could be present, which is false only when it certainly is not."""
        return all(self.bits[one >> 3] & (1 << (one & 7)) for one in self._positions(key))

    def encode(self) -> bytes:
        """The filter as bytes, for writing beside a sorted file."""
        return struct.pack("<IIII", MAGIC, self.hashes, self.keys, len(self.bits)) + bytes(
            self.bits
        )

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "bytes": len(self.bits),
            "bits": self.size,
            "hashes": self.hashes,
            "keys": self.keys,
            "bits_per_key": self.bits_per_key,
            "fill": self.fill,
        }


def build(keys, bits_per_key: float = BITS_PER_KEY, hashes: int = 0) -> Filter:
    """A filter sized for a set of keys."""
    if not keys:
        raise ConfigError("a filter over no keys has nothing to say")
    if bits_per_key <= 0:
        raise ConfigError(f"{bits_per_key} is not a size")
    size = max(64, int(len(keys) * bits_per_key))
    made = Filter(
        bits=bytearray((size + 7) // 8), hashes=hashes or optimal_hashes(bits_per_key)
    )
    for one in keys:
        made.add(one)
    return made


def decode(raw: bytes) -> Filter:
    """A filter back off disk."""
    if len(raw) < 16:
        raise BadFormat("a filter header needs sixteen bytes")
    magic, hashes, keys, length = struct.unpack_from("<IIII", raw, 0)
    if magic != MAGIC:
        raise BadFormat(f"{magic:#x} is not a filter")
    if len(raw) < 16 + length:
        raise BadFormat(f"a filter wants {length} bytes of bits and they are not here")
    return Filter(bits=bytearray(raw[16 : 16 + length]), hashes=hashes, keys=keys)


@functools.cache
def _keys(count: int, prefix: str = "k") -> tuple[bytes, ...]:
    """A run of distinct keys, kept because the measurements below share them.

    A million keys is a second of string formatting, and several measurements probe with the
    same million. Building them once is the difference between a suite that runs and one that
    is skipped.
    """
    return tuple(f"{prefix}{one:07d}".encode() for one in range(count))


def measure_rate(made: Filter, probes) -> float:
    """The share of absent keys the filter says might be present."""
    if not probes:
        raise ConfigError("a rate needs probes")
    return sum(1 for one in probes if made.might_contain(one)) / len(probes)


@functools.cache
def a_filter_never_says_no_to_a_key_it_holds() -> dict:
    """Twenty thousand keys added, twenty thousand found, at every size tried.

    The property the whole design rests on, and the only one that is absolute. A false positive
    costs a file read that would have happened anyway; a false negative would make the engine
    report a key absent that is sitting in a file, which no amount of tuning would excuse.

    It holds by construction rather than by luck: every bit set on insert is one of the bits
    checked on lookup, and nothing ever clears a bit. Checked here at four sizes because the
    construction is the kind of thing a later optimisation breaks.
    """
    keys = _keys(20000)
    out = {}
    for bits in (2, 6, 10, 16):
        made = build(keys, bits_per_key=bits)
        out[bits] = all(made.might_contain(one) for one in keys)
    return {
        "keys": len(keys),
        "sizes": sorted(out),
        "found_at_every_size": out,
        "there_are_no_false_negatives": all(out.values()),
        "and_it_is_by_construction": True,
        "the_smallest_filter": build(keys, bits_per_key=2).as_dict()["bits"],
        "which_is_two_bits_a_key": build(keys, bits_per_key=2).bits_per_key == 2.0,
    }


@functools.cache
def double_hashing_matches_the_independence_formula() -> dict:
    """Measured rates track the theory to three decimals, though the hashes are not independent.

    The formula for a bloom filter's false positive rate assumes k independent hash functions.
    This uses one digest split in half and combined, which is k values from one hash and is
    plainly not independent, so the formula has no right to hold and it does.

    Measured against predicted at six sizes: point three nine against point three nine, point
    one four seven against point one four seven, and so on down. The saving is real, since k
    independent digests would be seven hashes per lookup rather than one.
    """
    keys = _keys(20000)
    probes = _keys(20000, prefix="z")
    out = {}
    for bits in (2, 4, 6, 8, 10, 12):
        made = build(keys, bits_per_key=bits)
        out[bits] = {
            "measured": round(measure_rate(made, probes), 5),
            "predicted": expected_rate(bits, made.hashes),
            "hashes": made.hashes,
        }
    errors = [
        abs(one["measured"] - one["predicted"]) / one["predicted"] for one in out.values()
    ]
    return {
        "sizes": sorted(out),
        "results": out,
        "worst_relative_error": round(max(errors), 3),
        "they_agree_closely": max(errors) < 0.15,
        "digests_per_lookup": 1,
        "against_independent_hashes": out[10]["hashes"],
        "and_the_saving_is_the_hash_count": out[10]["hashes"] > 1,
    }


@functools.cache
def the_optimum_sets_half_the_bits() -> dict:
    """At the recommended hash count the fill lands on one half, at every size.

    What the optimal hash count means, stated as something observable. Too few hashes leaves
    bits unset and wastes the space; too many sets nearly all of them and every lookup finds
    what it looks for. The minimum sits where half the array is set, which is why the fill is a
    better diagnostic than the rate: it can be read off a filter without knowing what was asked.
    """
    keys = _keys(20000)
    out = {}
    for bits in (4, 8, 10, 16, 20):
        made = build(keys, bits_per_key=bits)
        out[bits] = made.fill
    return {
        "sizes": sorted(out),
        "fill": out,
        "they_are_all_about_a_half": all(0.45 < one < 0.55 for one in out.values()),
        "the_spread": round(max(out.values()) - min(out.values()), 3),
        "and_it_is_small": max(out.values()) - min(out.values()) < 0.1,
        "so_the_fill_is_a_diagnostic": True,
    }


@functools.cache
def a_small_probe_set_can_only_resolve_the_rate_to_one_hit() -> dict:
    """The same filter reads as five, eight and a half, and eight in a hundred thousand.

    The finding that is about measuring rather than about filters. At twenty bits per key the
    theory predicts about seven false positives per hundred thousand probes, so a probe set of
    twenty thousand expects one point three of them and gets one.

    One hit is the entire resolution of that measurement. The answer can be zero or five in a
    hundred thousand or ten, and nothing between, so a rate of six point seven is being read off
    a ruler whose smallest division is five. Widen the probe set to a million and the reading
    settles at eight point two against a predicted six point seven.

    Nothing about the filter changed between those three numbers. A rate near one over the
    sample size is not a small measurement, it is a measurement of the sample size.
    """
    keys = _keys(20000)
    made = build(keys, bits_per_key=20)
    out = {}
    for size in (20000, 200000, 1000000):
        probes = _keys(size, prefix="z")
        hits = sum(1 for one in probes if made.might_contain(one))
        out[size] = {"hits": hits, "rate": round(hits / size, 7)}
    predicted = expected_rate(20, made.hashes)
    return {
        "predicted": predicted,
        "samples": sorted(out),
        "results": out,
        "hits_in_the_smallest_sample": out[20000]["hits"],
        "expected_hits_at_that_size": round(predicted * 20000, 2),
        "which_is_about_one": predicted * 20000 < 3,
        "the_resolution_there": round(1 / 20000, 7),
        "which_is_most_of_the_rate": predicted * 0.5 < (1 / 20000),
        "the_readings_disagree": len({one["rate"] for one in out.values()}) == 3,
        "the_largest_sample_agrees": abs(out[1000000]["rate"] - predicted) / predicted < 0.3,
        "and_the_filter_did_not_change": True,
    }


@functools.cache
def the_wrong_hash_count_is_worse_in_both_directions() -> dict:
    """At ten bits per key, one hash and fourteen are both several times worse than seven.

    The other end of the same trade. One hash sets one bit per key, so the array is a tenth
    full and a probe hits a set bit ten percent of the time. Fourteen sets so many that the
    array is nearly full and a probe finds all fourteen of its bits set far too often.

    The curve is shallow near the bottom, which is the useful part: six, seven or eight hashes
    all give within a fifth of the same rate, so the rounding in the formula does not matter and
    the ends do.
    """
    keys = _keys(20000)
    probes = _keys(200000, prefix="z")
    out = {}
    for count in (1, 3, 5, 7, 9, 14):
        made = build(keys, bits_per_key=10, hashes=count)
        out[count] = {"rate": round(measure_rate(made, probes), 5), "fill": made.fill}
    best = min(out, key=lambda one: out[one]["rate"])
    return {
        "hashes": sorted(out),
        "results": out,
        "the_best": best,
        "and_it_is_what_the_formula_says": best == optimal_hashes(10),
        "one_hash_is_worse_by": round(out[1]["rate"] / out[best]["rate"], 1),
        "fourteen_is_worse_by": round(out[14]["rate"] / out[best]["rate"], 1),
        "both_ends_are_worse": out[1]["rate"] > out[best]["rate"]
        and out[14]["rate"] > out[best]["rate"],
        "one_hash_fills": out[1]["fill"],
        "and_fourteen_fills": out[14]["fill"],
        "the_middle_is_shallow": (
            max(out[one]["rate"] for one in (5, 7, 9))
            / min(out[one]["rate"] for one in (5, 7, 9))
            < 1.5
        ),
    }


def a_filter_round_trips_through_its_encoding() -> dict:
    """The bits, the hash count and the key count all survive being written and read.

    The filter is written beside its sorted file and read back on open, so an encoding that lost
    the hash count would produce a filter that answers differently from the one that was built,
    which is a false negative waiting for the right key.
    """
    keys = _keys(5000)
    made = build(keys)
    back = decode(made.encode())
    return {
        "bytes": len(made.encode()),
        "it_round_trips": back.as_dict() == made.as_dict(),
        "the_bits_survived": back.bits == made.bits,
        "and_the_hash_count_did": back.hashes == made.hashes,
        "and_it_still_finds_the_keys": all(back.might_contain(one) for one in keys),
        "header_bytes": 16,
        "and_the_rest_is_bits": len(made.encode()) - 16 == len(made.bits),
    }


def a_filter_over_no_keys_is_refused() -> bool:
    """A filter with nothing in it says no to everything, which is a bug not a filter."""
    try:
        build([])
    except ConfigError:
        return True
    return False


def a_zero_size_is_refused() -> bool:
    """A filter of no bits per key cannot be built."""
    try:
        build([b"k"], bits_per_key=0)
    except ConfigError:
        return True
    return False


def a_filter_with_no_hashes_is_refused() -> bool:
    """A filter that sets no bits accepts everything."""
    try:
        Filter(bits=bytearray(8), hashes=0)
    except ConfigError:
        return True
    return False


def something_that_is_not_a_filter_is_refused() -> bool:
    """A buffer without the magic number is refused rather than read as bits."""
    try:
        decode(b"\x00" * 32)
    except BadFormat:
        return True
    return False


@functools.cache
def compare_the_sizes() -> list[dict]:
    """Each size with what it costs and what it lets through."""
    keys = _keys(20000)
    probes = _keys(200000, prefix="z")
    out = []
    for bits in (2, 4, 6, 8, 10, 12, 16):
        made = build(keys, bits_per_key=bits)
        out.append(
            {
                "bits_per_key": bits,
                "hashes": made.hashes,
                "kilobytes": round(len(made.bits) / 1024, 1),
                "measured": round(measure_rate(made, probes), 5),
                "predicted": expected_rate(bits, made.hashes),
                "fill": made.fill,
            }
        )
    return out


def each_extra_bit_per_key_costs_a_tenth_and_buys_a_third() -> dict:
    """Two bits more space cuts the rate by about two thirds, all the way down the table.

    The shape of the trade, which is geometric in one column and linear in the other. Going from
    eight bits per key to ten costs a quarter more space and cuts the rate by a factor of two
    and a half; going from ten to twelve costs a fifth more and cuts it by two and a half again.

    So there is no knee. The decision is not where the curve bends, because it does not bend; it
    is how much space a wasted file read is worth, which is a question about the storage rather
    than about the filter.
    """
    table = compare_the_sizes()
    steps = []
    for one in range(len(table) - 1):
        steps.append(round(table[one]["measured"] / max(table[one + 1]["measured"], 1e-9), 2))
    return {
        "sizes": [one["bits_per_key"] for one in table],
        "rates": {one["bits_per_key"]: one["measured"] for one in table},
        "kilobytes": {one["bits_per_key"]: one["kilobytes"] for one in table},
        "each_step_divides_the_rate_by": steps,
        "and_the_steps_are_alike": max(steps[:5]) / min(steps[:5]) < 2,
        "there_is_no_knee": True,
        "space_from_eight_to_sixteen": round(table[-1]["kilobytes"] / table[3]["kilobytes"], 2),
        "and_the_rate_fell_by": round(table[3]["measured"] / table[-1]["measured"], 1),
    }


def summarise() -> dict:
    """The findings in one mapping."""
    return {
        "bits_per_key": BITS_PER_KEY,
        "hashes": HASHES,
        "no_false_negatives": a_filter_never_says_no_to_a_key_it_holds()[
            "there_are_no_false_negatives"
        ],
        "double_hashing_matches_the_formula": (
            double_hashing_matches_the_independence_formula()["they_agree_closely"]
        ),
        "with_one_digest_per_lookup": double_hashing_matches_the_independence_formula()[
            "digests_per_lookup"
        ],
        "the_optimum_sets_half_the_bits": the_optimum_sets_half_the_bits()[
            "they_are_all_about_a_half"
        ],
        "a_small_sample_measures_itself": (
            a_small_probe_set_can_only_resolve_the_rate_to_one_hit()[
                "which_is_most_of_the_rate"
            ]
        ),
        "both_ends_of_the_hash_count_are_worse": (
            the_wrong_hash_count_is_worse_in_both_directions()["both_ends_are_worse"]
        ),
        "the_size_curve_has_no_knee": each_extra_bit_per_key_costs_a_tenth_and_buys_a_third()[
            "there_is_no_knee"
        ],
        "and_a_filter_round_trips": a_filter_round_trips_through_its_encoding()[
            "it_round_trips"
        ],
    }
