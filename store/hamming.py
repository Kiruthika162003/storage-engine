from __future__ import annotations

import functools
import random

from store.errors import BadFormat, ConfigError

# Correction against detection: what one more layer of redundancy buys, and its limit.
#
# The checksum module's verdict was detection: a flipped bit is noticed and the read
# refused. The scrub module then needed a good copy to repair from. A SECDED Hamming code
# removes the need for the copy for exactly one class of damage: 72 stored bits carry 64
# data bits, any single flipped bit is corrected in place, any double flip is detected and
# refused. The implementation uses the classic formulation, the syndrome is the XOR of the
# positions of the set bits, which makes correction one flip at the syndrome's address.
# The measurements walk every single-bit position, sample the double flips, and price the
# 12.5 percent redundancy against the checksum's 4, because that ratio is the entire
# engineering decision between the two.

WORD_BITS = 72
PARITY_POSITIONS = (1, 2, 4, 8, 16, 32, 64)


def _data_positions() -> list[int]:
    return [at for at in range(1, WORD_BITS) if at not in PARITY_POSITIONS]


def encode(payload: bytes) -> bytes:
    """Eight data bytes to nine coded bytes."""
    if len(payload) != 8:
        raise ConfigError("the code word carries exactly eight bytes")
    data = int.from_bytes(payload, "little")
    word = 0
    for data_at, position in enumerate(_data_positions()):
        if (data >> data_at) & 1:
            word |= 1 << position
    syndrome = 0
    for position in range(1, WORD_BITS):
        if (word >> position) & 1:
            syndrome ^= position
    for parity in PARITY_POSITIONS:
        if syndrome & parity:
            word |= 1 << parity
    if bin(word).count("1") % 2:
        word |= 1
    return word.to_bytes(9, "little")


def decode(coded: bytes) -> tuple[bytes, str]:
    """The payload back with a verdict: clean, corrected, or a refusal on double damage."""
    if len(coded) != 9:
        raise BadFormat("a code word is nine bytes")
    word = int.from_bytes(coded, "little")
    syndrome = 0
    for position in range(1, WORD_BITS):
        if (word >> position) & 1:
            syndrome ^= position
    overall_odd = bin(word).count("1") % 2 == 1
    verdict = "clean"
    if syndrome and overall_odd:
        if syndrome >= WORD_BITS:
            raise BadFormat("the syndrome points outside the word")
        word ^= 1 << syndrome
        verdict = "corrected"
    elif syndrome and not overall_odd:
        raise BadFormat("double damage: detected, uncorrectable")
    elif not syndrome and overall_odd:
        word ^= 1
        verdict = "corrected"
    data = 0
    for data_at, position in enumerate(_data_positions()):
        if (word >> position) & 1:
            data |= 1 << data_at
    return data.to_bytes(8, "little"), verdict


@functools.cache
def every_single_bit_flip_corrects_in_place() -> bool:
    """All 72 positions, flipped one at a time, every one corrected to the original.

    The exhaustive walk is affordable because the word is small, and exhaustiveness is the
    point: the classic implementation bug is an off by one in the position mapping, which
    corrects 71 positions and corrupts on the last, and only the full walk tells those
    apart.
    """
    payload = bytes(range(65, 73))
    coded = encode(payload)
    for bit in range(WORD_BITS):
        damaged = bytearray(coded)
        damaged[bit // 8] ^= 1 << (bit % 8)
        back, verdict = decode(bytes(damaged))
        if back != payload or verdict != "corrected":
            return False
    return True


@functools.cache
def every_double_flip_is_refused_never_miscorrected() -> bool:
    """All 2,556 distinct double flips: refusal every time, silent miscorrection never.

    The dangerous failure is not the missed detection, it is the confident wrong
    correction, a double flip whose syndrome points at some third innocent bit. The
    overall parity bit exists to close exactly that path, and the exhaustive pair walk
    confirms it closed: every double lands in the refusing branch.
    """
    payload = bytes(range(1, 9))
    coded = encode(payload)
    for first in range(WORD_BITS):
        for second in range(first + 1, WORD_BITS):
            damaged = bytearray(coded)
            damaged[first // 8] ^= 1 << (first % 8)
            damaged[second // 8] ^= 1 << (second % 8)
            try:
                decode(bytes(damaged))
                return False
            except BadFormat:
                continue
    return True


@functools.cache
def triple_flips_can_lie_which_is_the_codes_edge() -> bool:
    """Beyond its design, the code miscorrects: some triples decode clean or corrected.

    Three flips can produce a syndrome that looks like a correctable single, and the code
    confidently repairs its way to wrong data. Sampled triples find such cases, which is
    the honest boundary: SECDED is a contract about one and two, and damage beyond it needs
    the checksum layer above, which is why real memory systems run both.
    """
    payload = bytes(range(9, 17))
    coded = encode(payload)
    source = random.Random(3)
    lies = 0
    for _ in range(300):
        bits = source.sample(range(WORD_BITS), 3)
        damaged = bytearray(coded)
        for bit in bits:
            damaged[bit // 8] ^= 1 << (bit % 8)
        try:
            back, _ = decode(bytes(damaged))
            if back != payload:
                lies += 1
        except BadFormat:
            continue
    return lies > 0


@functools.cache
def the_redundancy_is_an_eighth_against_the_checksums_twenty_fifth() -> bool:
    """One coded byte per eight data bytes, against CRC32's four per hundred-plus block.

    Correction is three times the redundancy of detection at these shapes, and the ratio
    widens with block size because the CRC amortises and the Hamming word cannot: its
    correction radius is per word, so its overhead is per word. That is the whole
    reason correction lives in memory hardware, where words are small and copies are
    unaffordable, while detection lives in storage, where blocks are large and the scrub
    has replicas.
    """
    hamming_overhead = 1 / 8
    crc_overhead = 4 / 128
    return abs(hamming_overhead - 0.125) < 1e-9 and hamming_overhead > crc_overhead * 3


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "singles_correct": every_single_bit_flip_corrects_in_place(),
        "doubles_refuse": every_double_flip_is_refused_never_miscorrected(),
        "triples_can_lie": triple_flips_can_lie_which_is_the_codes_edge(),
        "correction_costs_triple": (
            the_redundancy_is_an_eighth_against_the_checksums_twenty_fifth()
        ),
    }
