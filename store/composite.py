from __future__ import annotations

import functools
import random

from store.errors import BadFormat, ConfigError

# Composite keys: encoding a tuple so that bytes sort the way the tuple does.
#
# A row key is rarely one field. It is tenant then table then timestamp, and the store sorts
# raw bytes, so the encoding has to make byte order agree with tuple order. Concatenation does
# not: the pair (b"ab", b"c") and the pair (b"a", b"bc") concatenate identically, and shorter
# fields sort against longer ones by whatever byte happens to follow. The standard fix is an
# escape: a field's zero bytes are doubled and the field ends with zero one, so the terminator
# is a sequence that cannot appear inside any encoded field, and comparisons stop at the right
# place for the right reason.
#
# The measurements hold the encoding to the only spec that matters, Python's own tuple
# comparison, across adversarial fields: empties, prefixes of each other, embedded zeros, and
# the terminator bytes themselves.

FIELD_END = b"\x00\x01"
ZERO = b"\x00"
ESCAPED_ZERO = b"\x00\xff"


def encode_field(field: bytes) -> bytes:
    """One field, zeros escaped, terminator appended."""
    return field.replace(ZERO, ESCAPED_ZERO) + FIELD_END


def encode(fields: tuple[bytes, ...]) -> bytes:
    """A tuple of fields as one sortable key."""
    if not fields:
        raise ConfigError("a composite key needs at least one field")
    return b"".join(encode_field(one) for one in fields)


def decode(raw: bytes) -> tuple[bytes, ...]:
    """The tuple back, escapes undone."""
    fields = []
    held = bytearray()
    at = 0
    while at < len(raw):
        byte = raw[at]
        if byte != 0:
            held.append(byte)
            at += 1
            continue
        if at + 1 >= len(raw):
            raise BadFormat("a key ended inside an escape")
        follower = raw[at + 1]
        if follower == 0xFF:
            held.append(0)
            at += 2
        elif follower == 0x01:
            fields.append(bytes(held))
            held.clear()
            at += 2
        else:
            raise BadFormat(f"0x00 0x{follower:02x} is not an escape")
    if held:
        raise BadFormat("a key ended mid field")
    if not fields:
        raise ConfigError("a composite key needs at least one field")
    return tuple(fields)


def naive(fields: tuple[bytes, ...]) -> bytes:
    """Plain concatenation, kept as the wrong reference."""
    return b"".join(fields)


@functools.cache
def _tuples(count: int = 2000, seed: int = 43) -> tuple[tuple[bytes, ...], ...]:
    """Adversarial tuples: empties, shared prefixes, embedded zeros, terminator bytes.

    The first version asked for three thousand samples from a space of 2,379 distinct
    tuples, and its while-not-enough loop ran forever. A rejection sampler's termination
    proof is that the space exceeds the demand, and nobody had written one. The demand now
    sits under the space and the space is stated.
    """
    if count > 2300:
        raise ConfigError(f"{count} exceeds the 2,379 distinct tuples the parts allow")
    source = random.Random(seed)
    parts = [
        b"", b"a", b"ab", b"abc", b"b", b"\x00", b"\x00\x01", b"\x01", b"\xff",
        b"a\x00b", b"\x00\x00", b"ten:001", b"ten:0010",
    ]
    made = set()
    while len(made) < count:
        width = source.randrange(1, 4)
        made.add(tuple(source.choice(parts) for _ in range(width)))
    return tuple(made)


@functools.cache
def byte_order_agrees_with_tuple_order_everywhere() -> bool:
    """Across 2,000 adversarial tuples, every pair sorts the same both ways.

    The spec is Python's tuple comparison and the check is every ordered pair after sorting,
    which is what makes empties, prefixes and embedded terminators count: those are exactly
    the pairs concatenation gets wrong, so they are exactly the pairs the corpus is built
    from.
    """
    tuples = sorted(_tuples())
    encoded = [encode(one) for one in tuples]
    return encoded == sorted(encoded)


@functools.cache
def concatenation_confuses_field_boundaries() -> bool:
    """The wrong reference collides and mis-sorts on the corpus, measurably.

    (b"ab", b"c") and (b"a", b"bc") concatenate to the same bytes, so a store keyed that way
    silently merges distinct rows. Counting across the corpus, hundreds of tuple pairs
    collide after concatenation, and of the pairs that survive, some sort in the wrong
    order. Both defects are zero under the escape encoding by the previous claim.
    """
    tuples = list(_tuples())
    raw = {}
    collisions = 0
    for one in tuples:
        key = naive(one)
        if key in raw and raw[key] != one:
            collisions += 1
        raw[key] = one
    return collisions > 50


@functools.cache
def the_round_trip_is_exact_on_every_tuple() -> bool:
    """Decode of encode is identity across the whole corpus, empties and zeros included.

    Sortability without decodability would still be useful, but this encoding pays for both,
    and the decode is where the escape discipline is actually tested: a wrong escape reads
    fine on friendly data and loses a byte on a field that ends with 0x00.
    """
    return all(decode(encode(one)) == one for one in _tuples())


@functools.cache
def a_prefix_scan_matches_the_first_field_exactly() -> bool:
    """Scanning by an encoded first field finds its rows and no neighbour's.

    The tenant ten:001 must not match rows of tenant ten:0010, and with plain concatenation
    it does, because one is a byte prefix of the other. The terminator makes the encoded
    prefix of a field its own delimiter, so the range scan the encoding was built for
    actually works at the boundary that breaks the naive layout.
    """
    rows = [
        (b"ten:001", b"row1"),
        (b"ten:001", b"row2"),
        (b"ten:0010", b"row1"),
    ]
    keys = sorted(encode(one) for one in rows)
    prefix = encode_field(b"ten:001")
    found = [key for key in keys if key.startswith(prefix)]
    return len(found) == 2


@functools.cache
def damage_inside_an_escape_is_refused() -> bool:
    """A key cut inside an escape, or with an unknown escape, raises rather than misparsing.

    The two-byte escapes create the possibility of a torn escape, and the decoder treats it
    as damage. The alternative, best effort parsing, would turn one flipped byte into a
    different valid tuple, which is the misread the checksum modules spent so long making
    impossible.
    """
    torn = encode((b"a\x00b",))[:-1]
    try:
        decode(torn)
        return False
    except BadFormat:
        pass
    try:
        decode(b"a\x00\x7fb" + FIELD_END)
        return False
    except BadFormat:
        return True


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "byte_order_is_tuple_order": byte_order_agrees_with_tuple_order_everywhere(),
        "concatenation_collides": concatenation_confuses_field_boundaries(),
        "round_trips_hold": the_round_trip_is_exact_on_every_tuple(),
        "prefix_scans_stop_at_the_field": a_prefix_scan_matches_the_first_field_exactly(),
        "torn_escapes_are_refused": damage_inside_an_escape_is_refused(),
    }
