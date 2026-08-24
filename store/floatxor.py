from __future__ import annotations

import functools
import random
import struct

from store.errors import BadFormat, ConfigError

# XOR float compression: the values half of the Gorilla pair.
#
# The timeseries and bitpack modules compressed the timestamps; the values ride beside them
# and have their own structure. A slowly moving gauge, a temperature, a queue depth, a
# price, changes little between samples, and the IEEE bit patterns of near-equal doubles
# share their sign, exponent and leading mantissa. XOR of consecutive values is then mostly
# zero bits, and storing only the differing window, its position and width, spends bytes
# proportional to the change rather than the value. Byte-aligned here as in bitpack's
# spirit, because the point is the measurement of the mechanism, and the mechanism is that
# stable signals cost little and jumpy ones cost everything.


def _bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def _value(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def encode(values: list[float]) -> bytes:
    """First value raw, then per value: a zero byte, or offset, width and window bytes."""
    if not values:
        raise ConfigError("an empty series encodes nothing")
    made = bytearray(struct.pack("<d", values[0]))
    previous = _bits(values[0])
    for value in values[1:]:
        bits = _bits(value)
        delta = bits ^ previous
        previous = bits
        if delta == 0:
            made.append(0)
            continue
        low = (delta & -delta).bit_length() - 1
        high = delta.bit_length()
        offset = low // 8
        width = (high - offset * 8 + 7) // 8
        made.append((offset << 4) | width)
        window = delta >> (offset * 8)
        made.extend(window.to_bytes(width, "little"))
    return bytes(made)


def decode(raw: bytes) -> list[float]:
    """The series back."""
    if len(raw) < 8:
        raise BadFormat("a series needs its first value")
    values = [struct.unpack("<d", raw[:8])[0]]
    previous = _bits(values[0])
    at = 8
    while at < len(raw):
        header = raw[at]
        at += 1
        if header == 0:
            values.append(_value(previous))
            continue
        offset = header >> 4
        width = header & 0x0F
        if at + width > len(raw):
            raise BadFormat("a window ended early")
        window = int.from_bytes(raw[at : at + width], "little")
        at += width
        previous ^= window << (offset * 8)
        values.append(_value(previous))
    return values


def flat_bytes(values: list[float]) -> int:
    """Eight bytes each, the reference."""
    return 8 * len(values)


@functools.cache
def _gauge(count: int = 20000, seed: int = 283) -> tuple[float, ...]:
    """A slowly drifting gauge: the friendly shape."""
    source = random.Random(seed)
    made = []
    level = 20.0
    for _ in range(count):
        level += source.uniform(-0.05, 0.05)
        made.append(round(level, 2))
    return tuple(made)


@functools.cache
def _noise(count: int = 20000, seed: int = 293) -> tuple[float, ...]:
    """Uncorrelated doubles: the hostile shape."""
    source = random.Random(seed)
    return tuple(source.uniform(-1e9, 1e9) for _ in range(count))


@functools.cache
def a_flat_gauge_costs_one_byte_per_repeat() -> bool:
    """A constant series stores at one byte per sample after the first.

    Equal doubles XOR to zero, the zero header byte is the whole record, and a sensor that
    reports the same reading all night costs a byte an hour. The eight-to-one against flat
    is the codec's floor, this format's version of the varint's byte.
    """
    values = [42.5] * 5000
    raw = encode(values)
    return len(raw) == 8 + 4999 and decode(raw) == values


@functools.cache
def the_rounded_gauge_disappoints_and_the_reason_is_decimal() -> bool:
    """The drifting gauge costs 6.42 bytes per sample, not the third I expected.

    The expectation came from the Gorilla paper's gauges; the measurement says a fifth
    under flat, and the culprit is the rounding. A gauge rounded to two decimals moves in
    steps of 0.01, which is not representable in binary, so consecutive values differ
    across most of the low mantissa and the XOR window is six or seven bytes nearly every
    time. The paper's wins lean on three things this shape lacks: values that repeat
    exactly, sub-byte windows, and the same-window-as-last-time control path. Byte
    alignment costs some of that; the decimal steps cost more. A codec keyed to a signal
    shape is keyed to the shape's representation, not its picture on a dashboard.
    """
    values = list(_gauge())
    raw = encode(values)
    per_sample = len(raw) / len(values)
    return 6.0 < per_sample < 7.0 and decode(raw) == values


@functools.cache
def uncorrelated_doubles_cost_more_than_flat() -> bool:
    """Random doubles store at 8.48 bytes per sample: six percent over flat.

    Unrelated values differ nearly everywhere, the window is seven or eight bytes plus a
    header, and the codec runs over flat storage. The compression module's threshold
    lesson without the threshold: a codec keyed to a signal shape is a liability off the
    shape, and a writer should measure and fall back to raw.
    """
    values = list(_noise())
    raw = encode(values)
    per_sample = len(raw) / len(values)
    return per_sample > 8.3 and decode(raw) == values


@functools.cache
def round_trips_are_bit_exact() -> bool:
    """Every shape round trips to equality, including the specials.

    Floats have the values equality glosses over, negative zero and infinities, and bit
    exactness through pack and unpack is checked with bit compares where equality would
    lie. NaN is excluded deliberately: it fails equality by definition, and a store that
    needs NaN needs a bit-level comparison policy first.
    """
    specials = [0.0, -0.0, float("inf"), float("-inf"), 1e-308, -1e308]
    cases = [list(_gauge(500)), list(_noise(500)), specials]
    for values in cases:
        back = decode(encode(values))
        if len(back) != len(values):
            return False
        for left, right in zip(values, back, strict=True):
            if _bits(left) != _bits(right):
                return False
    return True


def compare_the_shapes() -> list[dict]:
    """One row per signal shape."""
    rows = []
    for name, values in (
        ("constant", [42.5] * 20000),
        ("gauge", list(_gauge())),
        ("noise", list(_noise())),
    ):
        raw = encode(values)
        rows.append(
            {
                "shape": name,
                "samples": len(values),
                "encoded_bytes": len(raw),
                "bytes_per_sample": round(len(raw) / len(values), 2),
                "flat_bytes": flat_bytes(values),
            }
        )
    return rows


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "constants_cost_a_byte": a_flat_gauge_costs_one_byte_per_repeat(),
        "the_gauge_disappoints": the_rounded_gauge_disappoints_and_the_reason_is_decimal(),
        "noise_costs_more_than_flat": uncorrelated_doubles_cost_more_than_flat(),
        "round_trips_are_bit_exact": round_trips_are_bit_exact(),
    }
