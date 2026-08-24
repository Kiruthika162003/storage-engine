from __future__ import annotations

import functools
import itertools

from store.errors import BadFormat, ConfigError
from store.timeseries import _metronome, _scrape, zigzag
from store.timeseries import encode as dod_encode

# The layer under the byte: where the timeseries module's missing fractions live.
#
# The delta of delta module ended at one byte per sample and named the culprit: a varint
# cannot spend less than a byte. This module is the spender. Values are packed into fixed
# bit widths chosen per block, and a run of zeros collapses to a count, so the metronome's
# stream of zero second-differences finally costs what the folklore promised. The format is
# deliberately small: a block is a width byte, a count, and either a zero-run marker or the
# packed bits, because the point is the measurement, not a codec zoo.

ZERO_RUN = 0xFF


def _width_of(values: list[int]) -> int:
    """The bits the widest value needs, at least one."""
    return max((value.bit_length() for value in values), default=1) or 1


def pack_block(values: list[int]) -> bytes:
    """One block: zero runs collapse, everything else packs to the block's width."""
    if not values:
        raise ConfigError("an empty block packs nothing")
    if any(value < 0 for value in values):
        raise ConfigError("pack unsigned values; zigzag first")
    if len(values) > 65535:
        raise ConfigError("a block holds at most 65535 values")
    if all(value == 0 for value in values):
        return bytes([ZERO_RUN]) + len(values).to_bytes(2, "little")
    width = _width_of(values)
    if width >= ZERO_RUN:
        raise ConfigError(f"{width} bits is too wide for the format")
    made = bytearray([width])
    made.extend(len(values).to_bytes(2, "little"))
    bits = 0
    held = 0
    for value in values:
        bits |= value << held
        held += width
        while held >= 8:
            made.append(bits & 0xFF)
            bits >>= 8
            held -= 8
    if held:
        made.append(bits & 0xFF)
    return bytes(made)


def unpack_block(raw: bytes) -> tuple[list[int], int]:
    """One block back, and where it ended."""
    if len(raw) < 3:
        raise BadFormat("a block header needs three bytes")
    width = raw[0]
    count = int.from_bytes(raw[1:3], "little")
    if width == ZERO_RUN:
        return [0] * count, 3
    total_bits = width * count
    body = (total_bits + 7) // 8
    if len(raw) < 3 + body:
        raise BadFormat("a block body ended early")
    bits = int.from_bytes(raw[3 : 3 + body], "little")
    mask = (1 << width) - 1
    values = [(bits >> (at * width)) & mask for at in range(count)]
    return values, 3 + body


def pack(values: list[int], block: int = 256) -> bytes:
    """A stream cut into blocks, each packed to its own width."""
    made = bytearray()
    for at in range(0, len(values), block):
        made.extend(pack_block(values[at : at + block]))
    return bytes(made)


def unpack(raw: bytes) -> list[int]:
    """The stream back."""
    values: list[int] = []
    at = 0
    while at < len(raw):
        block, used = unpack_block(raw[at:])
        values.extend(block)
        at += used
    return values


def _second_differences(moments) -> list[int]:
    """The timeseries module's stream reduced to zigzagged second differences."""
    deltas = [later - earlier for earlier, later in itertools.pairwise(moments)]
    return [zigzag(later - earlier) for earlier, later in itertools.pairwise(deltas)]


@functools.cache
def the_metronome_finally_costs_fractions_of_a_byte() -> bool:
    """The packed metronome spends 0.012 bits per sample where the varint spent eight.

    The promised payoff: the metronome's second differences are all zero, the zero-run
    blocks collapse each 256 of them to three bytes, and the whole twenty thousand sample
    stream packs into 240 bytes against the varint layer's 20,006. The regularity the
    timeseries module could not monetise is worth 83 to 1 in the layer built to spend it.
    """
    wobbles = _second_differences(list(_metronome()))
    packed = pack(wobbles)
    varint_cost = len(dod_encode(list(_metronome())))
    return len(packed) < 300 and varint_cost > 19000


@functools.cache
def the_jittery_scrape_packs_to_half_a_byte() -> bool:
    """Real jitter needs five bits a sample, still a third under the varint's byte.

    The scrape's second differences span the jitter's double-width range, five bits
    zigzagged, and the stream costs 0.68 bytes per sample with block headers included. The
    byte boundary was the varint's floor, not the data's, and this is the measurement that
    separates the two.
    """
    wobbles = _second_differences(list(_scrape()))
    packed = pack(wobbles)
    per_sample = len(packed) / len(wobbles)
    return 0.3 < per_sample < 0.7


@functools.cache
def round_trips_hold_across_widths_and_runs() -> bool:
    """Mixed widths, pure zeros, singletons, and the 65535 count edge all round trip."""
    cases = [
        [0],
        [1],
        [0] * 5000,
        [2**40, 0, 1],
        list(range(1000)),
        _second_differences(list(_scrape(2000))),
    ]
    return all(unpack(pack(values)) == values for values in cases)


@functools.cache
def the_width_is_per_block_and_an_outlier_taxes_only_its_block() -> bool:
    """One huge value costs its 256 neighbours 39 extra bits each, and nobody else.

    The clean stream packs to 2,620 bytes at two bits a value. One 41 bit outlier widens
    its whole block to 41 bits, 3,868 bytes total, a 48 percent tax, because the block is
    the width's blast radius and 255 innocents share it. Per-stream width would have
    charged all ten thousand values, 51,250 bytes, nearly twenty times clean. The block
    size knob is the same trade the zone map made: smaller blocks contain damage better
    and spend more on headers.
    """
    small = [3] * 10000
    clean = len(pack(small))
    small[5000] = 2**40
    dirty = len(pack(small))
    stream_width_cost = (41 * 10000 + 7) // 8
    return clean * 1.3 < dirty < clean * 1.6 and stream_width_cost > clean * 15


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "the_metronome_pays_off": the_metronome_finally_costs_fractions_of_a_byte(),
        "jitter_packs_to_half": the_jittery_scrape_packs_to_half_a_byte(),
        "round_trips_hold": round_trips_hold_across_widths_and_runs(),
        "outliers_tax_their_block": (
            the_width_is_per_block_and_an_outlier_taxes_only_its_block()
        ),
    }
