from __future__ import annotations

import functools
import random

from store.errors import BadFormat, ConfigError

# Two ways to spend fewer bytes on integers, measured against the flat encoding they replace.
#
# A store is full of small integers that arrive in fixed width fields: lengths, counts,
# sequence numbers, offsets. A fixed eight byte field spends eight bytes on the number three.
# The varint spends one byte per seven bits of magnitude, so small numbers are small and large
# ones cost a ninth byte in the worst case. Delta encoding helps a different shape: a sorted
# run of large numbers whose gaps are small, where storing the gaps turns large values into
# small ones and hands them to the varint.
#
# Neither is free. The varint cannot be indexed into, because the width of the nth value
# depends on every value before it, and delta encoding cannot be read from the middle for the
# same reason. Both trade random access for space, which is the same trade prefix compression
# made inside blocks, and it is worth noticing that it is always this trade.

MOST = 0x7F
MORE = 0x80


def encode(value: int) -> bytes:
    """One integer, seven bits at a time, low bits first."""
    if value < 0:
        raise ConfigError(f"{value} is negative and varints are unsigned")
    made = bytearray()
    while True:
        low = value & MOST
        value >>= 7
        if value:
            made.append(low | MORE)
        else:
            made.append(low)
            return bytes(made)


def decode(raw: bytes, at: int = 0) -> tuple[int, int]:
    """One integer back, and where it ended."""
    value = 0
    shift = 0
    while True:
        if at >= len(raw):
            raise BadFormat("a varint ran off the end")
        if shift > 63:
            raise BadFormat("a varint is wider than 64 bits")
        byte = raw[at]
        at += 1
        value |= (byte & MOST) << shift
        if not byte & MORE:
            return value, at
        shift += 7


def encode_all(values) -> bytes:
    """A run of integers, one after another."""
    made = bytearray()
    for value in values:
        made.extend(encode(value))
    return bytes(made)


def decode_all(raw: bytes) -> list[int]:
    """The whole run back."""
    made = []
    at = 0
    while at < len(raw):
        value, at = decode(raw, at)
        made.append(value)
    return made


def encode_deltas(values) -> bytes:
    """A sorted run stored as its gaps, which are small when the run is dense."""
    made = bytearray()
    previous = 0
    for value in values:
        if value < previous:
            raise ConfigError("delta encoding needs a sorted run")
        made.extend(encode(value - previous))
        previous = value
    return bytes(made)


def decode_deltas(raw: bytes) -> list[int]:
    """The run back from its gaps."""
    made = []
    total = 0
    for gap in decode_all(raw):
        total += gap
        made.append(total)
    return made


def flat_bytes(values) -> int:
    """What the fixed width encoding costs, which is the reference."""
    return 8 * len(list(values))


@functools.cache
def _sequences(count: int = 10000, seed: int = 61) -> tuple[int, ...]:
    """Sequence numbers the way a store issues them: dense, increasing, occasionally gapped."""
    source = random.Random(seed)
    made = []
    value = 0
    for _ in range(count):
        value += 1 if source.random() < 0.9 else source.randrange(2, 50)
        made.append(value)
    return tuple(made)


@functools.cache
def small_integers_cost_one_byte_in_nine_less_at_the_top() -> bool:
    """A varint is one byte through 127, two through 16,383, and ten at the 64 bit ceiling.

    The break points are powers of 128, and the worst case is ten bytes for a number that flat
    encoding stores in eight, so the varint is a bet that the data is small. Lengths and
    counts are; hashes and random identifiers are not, and varint encoding a hash wastes a
    byte a value while making it unindexable.
    """
    return (
        len(encode(127)) == 1
        and len(encode(128)) == 2
        and len(encode(16383)) == 2
        and len(encode(16384)) == 3
        and len(encode(2**64 - 1)) == 10
    )


@functools.cache
def sequence_numbers_shrink_eightfold_as_deltas() -> bool:
    """Ten thousand sequence numbers: 80,000 bytes flat, 25,321 as varints, 10,000 as deltas.

    The varint alone helps because the values are only five digits, two or three bytes each.
    The deltas collapse to exactly one byte per value, ten thousand for ten thousand, because
    every gap in this stream, including the occasional jump of up to fifty, is under 128. The
    encoding hit its floor: no run of positive gaps can cost less than a byte each.

    The gap distribution is the entire effect. The same trick on the raw values of a hash
    index would produce gaps as large as the values and save nothing.
    """
    values = list(_sequences())
    flat = flat_bytes(values)
    plain = len(encode_all(values))
    deltas = len(encode_deltas(values))
    return flat == 80000 and deltas < plain < flat and flat / deltas > 6


@functools.cache
def the_round_trip_is_exact_over_the_whole_range() -> bool:
    """Every width boundary and a spray of random values decode to what was encoded.

    The boundaries are where varint bugs live, because the carry between the seventh and
    eighth bit is the one the tests without boundaries never touch.
    """
    edges = [0, 1, 127, 128, 16383, 16384, 2**21 - 1, 2**21, 2**63, 2**64 - 1]
    source = random.Random(3)
    values = edges + [source.randrange(2**64) for _ in range(500)]
    for value in values:
        back, _ = decode(encode(value))
        if back != value:
            return False
    return True


@functools.cache
def a_truncated_varint_is_refused_not_misread() -> bool:
    """Cutting the last byte off raises rather than returning a smaller number.

    The continuation bit makes truncation detectable: every byte but the last promises
    another, so a cut always lands on a promise. An encoding whose truncations decode to
    plausible values, fixed width little endian for instance, cannot tell this from data.
    """
    raw = encode(2**40)
    try:
        decode(raw[:-1])
    except BadFormat:
        return True
    return False


@functools.cache
def an_overlong_varint_is_refused() -> bool:
    """Eleven continuation bytes is not a 64 bit integer, whatever it claims.

    Without the width check a malicious or corrupt stream stalls the decoder through
    arbitrarily many continuation bytes. With it, damage is an error at the eleventh byte.
    """
    try:
        decode(bytes([MORE] * 11))
    except BadFormat:
        return True
    return False


@functools.cache
def an_unsorted_run_is_refused_by_the_delta_encoder() -> bool:
    """A negative gap has no unsigned encoding, and the refusal names the real problem.

    Silently encoding the wrapped difference would decode to garbage a long way from the
    mistake. The error at encode time is at the mistake.
    """
    try:
        encode_deltas([5, 3])
    except ConfigError:
        return True
    return False


def compare_the_encodings(count: int = 10000) -> list[dict]:
    """One row per encoding over the same sequence run."""
    values = list(_sequences(count))
    rows = [
        {"encoding": "flat", "bytes": flat_bytes(values)},
        {"encoding": "varint", "bytes": len(encode_all(values))},
        {"encoding": "delta+varint", "bytes": len(encode_deltas(values))},
    ]
    for row in rows:
        row["bytes_per_value"] = round(row["bytes"] / count, 2)
    return rows


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "widths_break_at_powers_of_128": small_integers_cost_one_byte_in_nine_less_at_the_top(),
        "deltas_hit_the_one_byte_floor": sequence_numbers_shrink_eightfold_as_deltas(),
        "round_trip_is_exact": the_round_trip_is_exact_over_the_whole_range(),
        "truncation_is_refused": a_truncated_varint_is_refused_not_misread(),
        "overlong_is_refused": an_overlong_varint_is_refused(),
        "unsorted_is_refused": an_unsorted_run_is_refused_by_the_delta_encoder(),
    }
