from __future__ import annotations

import functools
import random
import zlib

# Three checksums, and what each one actually detects.
#
# The package uses CRC32 everywhere a frame needs guarding, and this module is the receipt for
# that choice. The candidates are the ones that turn up in real systems: a plain sum, which is
# what people write first; a Fletcher style position weighted sum, which is what people write
# second; and CRC32, which is what everyone converges on. They differ not in whether they
# detect corruption, all three catch most of it, but in which corruptions they are blind to,
# and blindness is the only property that matters, because the common case of no corruption is
# free for everyone.

WIDTH = 32
MASK = 0xFFFFFFFF


def plain_sum(raw: bytes) -> int:
    """Add the bytes, wrap at 32 bits: blind to reordering and to balanced changes."""
    return sum(raw) & MASK


def fletcher(raw: bytes) -> int:
    """Two running sums, the second weighting position: catches reordering, still linear."""
    low = 0
    high = 0
    for byte in raw:
        low = (low + byte) % 65535
        high = (high + low) % 65535
    return (high << 16) | low


def crc32(raw: bytes) -> int:
    """The polynomial checksum every store settles on."""
    return zlib.crc32(raw) & MASK


CHECKSUMS = {"sum": plain_sum, "fletcher": fletcher, "crc32": crc32}


def _payload(size: int = 256, seed: int = 71) -> bytes:
    """One random payload."""
    return random.Random(seed).randbytes(size)


@functools.cache
def a_swap_is_invisible_to_the_sum_and_visible_to_the_others() -> bool:
    """Swapping two bytes changes nothing the plain sum can see.

    Addition commutes, so any reordering of the payload has the same sum, and a disk that
    writes two sectors in the wrong order produces exactly this damage. Fletcher weights each
    byte by its position and catches it; so does the CRC.
    """
    raw = bytearray(_payload())
    raw[10], raw[200] = raw[200], raw[10]
    swapped = bytes(raw)
    original = _payload()
    return (
        plain_sum(swapped) == plain_sum(original)
        and fletcher(swapped) != fletcher(original)
        and crc32(swapped) != crc32(original)
    )


@functools.cache
def a_balanced_pair_of_flips_is_invisible_to_the_sum() -> bool:
    """Add one somewhere, subtract one somewhere else, and the sum balances.

    This is not exotic damage: a bit flipping downward in one byte and upward in another is
    two independent single bit events, and memory produces those. The sum is blind to every
    corruption whose byte deltas cancel, which is a large and structured class.
    """
    raw = bytearray(_payload())
    raw[5] = (raw[5] + 1) & 0xFF
    raw[6] = (raw[6] - 1) & 0xFF
    damaged = bytes(raw)
    original = _payload()
    return plain_sum(damaged) == plain_sum(original) and crc32(damaged) != crc32(original)


@functools.cache
def every_single_bit_flip_is_caught_by_all_three() -> bool:
    """One flipped bit changes every checksum, everywhere in the payload.

    Single bit detection is the easy property, which is the point of measuring it: the sum
    looks equal to the CRC on the easy damage, and choosing by the easy damage chooses wrong.
    """
    original = _payload(64)
    for at in range(64):
        for bit in range(8):
            raw = bytearray(original)
            raw[at] ^= 1 << bit
            damaged = bytes(raw)
            for check in CHECKSUMS.values():
                if check(damaged) == check(original):
                    return False
    return True


@functools.cache
def the_sum_does_not_even_use_its_32_bits() -> bool:
    """I claimed all three behave as uniform 32 bit tags under random damage. The sum does not.

    Two thousand trials of heavy random corruption and the plain sum collided once, which a
    32 bit tag should do once in four billion trials, not once in two thousand. The reason is
    the range: the sum of 256 bytes lives between 0 and 65,280, a 16 bit space wearing a 32
    bit type, so its collision rate against unstructured damage is 2 to the minus 16 at this
    payload size and gets worse as payloads shrink.

    Fletcher and the CRC produced no collision in the same trials, consistent with tags that
    actually occupy their width. So the plain sum fails both ways: blind to structured damage
    by algebra, and weak against random damage by range. The measurement that was supposed to
    say the candidates are equal on unstructured damage says the opposite.
    """
    source = random.Random(9)
    original = _payload()
    tags = {name: check(original) for name, check in CHECKSUMS.items()}
    collided = dict.fromkeys(CHECKSUMS, 0)
    for _ in range(2000):
        raw = bytearray(original)
        for _ in range(source.randrange(1, 40)):
            raw[source.randrange(len(raw))] = source.randrange(256)
        damaged = bytes(raw)
        if damaged == original:
            continue
        for name, check in CHECKSUMS.items():
            if check(damaged) == tags[name]:
                collided[name] += 1
    top = max(plain_sum(bytes([255] * 256)), 1)
    return (
        collided["sum"] >= 1
        and collided["fletcher"] == 0
        and collided["crc32"] == 0
        and top < 1 << 17
    )


def blindness_table(trials: int = 3000, seed: int = 17) -> list[dict]:
    """Structured damage kinds against each checksum, missed counts."""
    source = random.Random(seed)
    kinds = ("swap", "balanced", "zero_run", "random")
    missed = {name: dict.fromkeys(kinds, 0) for name in CHECKSUMS}
    original = _payload(256, 71)
    tags = {name: check(original) for name, check in CHECKSUMS.items()}
    for _ in range(trials):
        kind = source.choice(kinds)
        raw = bytearray(original)
        if kind == "swap":
            a, b = source.randrange(256), source.randrange(256)
            raw[a], raw[b] = raw[b], raw[a]
        elif kind == "balanced":
            a, b = source.randrange(256), source.randrange(256)
            raw[a] = (raw[a] + 7) & 0xFF
            raw[b] = (raw[b] - 7) & 0xFF
        elif kind == "zero_run":
            start = source.randrange(200)
            for at in range(start, start + 8):
                raw[at] = 0
        else:
            raw[source.randrange(256)] = source.randrange(256)
        damaged = bytes(raw)
        if damaged == original:
            continue
        for name, check in CHECKSUMS.items():
            if check(damaged) == tags[name]:
                missed[name][kind] += 1
    return [{"checksum": name, **counts} for name, counts in missed.items()]


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "a_swap_blinds_the_sum": a_swap_is_invisible_to_the_sum_and_visible_to_the_others(),
        "balance_blinds_the_sum": a_balanced_pair_of_flips_is_invisible_to_the_sum(),
        "single_bits_catch_everywhere": every_single_bit_flip_is_caught_by_all_three(),
        "the_sum_wastes_its_width": the_sum_does_not_even_use_its_32_bits(),
    }
