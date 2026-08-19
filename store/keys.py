from __future__ import annotations

import struct
from dataclasses import dataclass

from store.errors import ConfigError, TooLarge

# Turning values into keys that sort the way the values do.
#
# Everything below the engine compares keys as bytes. A sorted file is sorted by memcmp, a
# merging iterator picks the smallest byte string, a block index binary searches on bytes. So a
# key is only useful if its byte order matches the order the caller means, and for anything
# except an ascii string that takes work.
#
# The three cases that bite, in order of how often they bite. An integer written as text sorts
# as text, so ten comes before nine. An integer written big endian sorts correctly while it is
# unsigned and inverts across zero once it is signed. A float has a sign bit that makes every
# negative value sort above every positive one, and the negatives sort backwards among
# themselves.
#
# All three are fixed by an encoding rather than by a comparator, which is the point. A
# comparator has to be carried everywhere the bytes go, including into the block index and the
# bloom filter and any tool that reads the file; an encoding travels with the key.

# The largest key and value this format can express, which is a length prefix decision.
MAX_KEY = 1 << 16
MAX_VALUE = 1 << 24

# Type tags, which go first so that keys of different types group together rather than
# interleaving. The order of the tags is the order the types sort in.
NULL = 0x00
FALSE = 0x01
TRUE = 0x02
INT = 0x10
FLOAT = 0x20
TEXT = 0x30
BLOB = 0x40
TAGS = (NULL, FALSE, TRUE, INT, FLOAT, TEXT, BLOB)


def encode_int(value: int) -> bytes:
    """A signed integer as eight bytes that sort in numeric order.

    Big endian puts the most significant byte first, which is what makes byte order match
    numeric order for unsigned values. The sign bit is then flipped so that negatives sort below
    positives: without it, minus one is all ones and sorts above everything.
    """
    if not -(2**63) <= value < 2**63:
        raise TooLarge(f"{value} does not fit in a signed eight byte key")
    return struct.pack(">Q", (value + (1 << 63)) & ((1 << 64) - 1))


def decode_int(raw: bytes) -> int:
    """The integer back."""
    if len(raw) != 8:
        raise ConfigError(f"{len(raw)} bytes is not an encoded integer")
    return struct.unpack(">Q", raw)[0] - (1 << 63)


def encode_float(value: float) -> bytes:
    """A float as eight bytes that sort in numeric order.

    The IEEE layout already sorts correctly among positives when read as an unsigned integer.
    Negatives sort backwards and above, so a negative has every bit flipped and a positive has
    only its sign bit set. That is the standard trick and it is worth writing out because the
    two branches are easy to get the wrong way round.
    """
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    if bits & (1 << 63):
        bits = ~bits & ((1 << 64) - 1)
    else:
        bits |= 1 << 63
    return struct.pack(">Q", bits)


def decode_float(raw: bytes) -> float:
    """The float back.

    The top bit of the encoded form says which branch the encoder took, not what the sign was:
    it is set when the original was positive and clear when it was negative.
    """
    if len(raw) != 8:
        raise ConfigError(f"{len(raw)} bytes is not an encoded float")
    bits = struct.unpack(">Q", raw)[0]
    if bits & (1 << 63):
        bits &= (1 << 63) - 1
    else:
        bits = ~bits & ((1 << 64) - 1)
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def encode_text(value: str) -> bytes:
    """A string as utf-8, which already sorts in code point order.

    Worth stating because it is the one case that needs nothing. Utf-8 was designed so that byte
    order matches code point order, which is why the encoding is used here rather than utf-16,
    where it does not.
    """
    return value.encode("utf-8")


def key_of(value: object) -> bytes:
    """One value as a sortable key, tagged with its type.

    The tag goes first so that an integer never compares against the middle of a string. Without
    it a key space holding both is ordered by an accident of encoding rather than by anything a
    caller would recognise.
    """
    if value is None:
        return bytes([NULL])
    if isinstance(value, bool):
        return bytes([TRUE if value else FALSE])
    if isinstance(value, int):
        return bytes([INT]) + encode_int(value)
    if isinstance(value, float):
        return bytes([FLOAT]) + encode_float(value)
    if isinstance(value, str):
        return bytes([TEXT]) + encode_text(value)
    if isinstance(value, (bytes, bytearray)):
        return bytes([BLOB]) + bytes(value)
    raise ConfigError(f"{type(value).__name__} has no key encoding")


def value_of(raw: bytes) -> object:
    """The value back out of a key."""
    if not raw:
        raise ConfigError("an empty key has no value")
    tag, body = raw[0], raw[1:]
    if tag == NULL:
        return None
    if tag == FALSE:
        return False
    if tag == TRUE:
        return True
    if tag == INT:
        return decode_int(body)
    if tag == FLOAT:
        return decode_float(body)
    if tag == TEXT:
        return body.decode("utf-8")
    if tag == BLOB:
        return body
    raise ConfigError(f"{tag} is not a key type")


def naive_int(value: int) -> bytes:
    """An integer as its decimal text, which is what a key looks like before anybody thinks.

    Kept because the measurement below needs something to be wrong, and this is the thing that
    is actually wrong in practice: it is what you get from str(key).encode().
    """
    return str(value).encode("ascii")


def naive_float(value: float) -> bytes:
    """A float as its raw IEEE bytes, big endian, with the sign bit left alone."""
    return struct.pack(">d", value)


@dataclass(frozen=True)
class Ordering:
    """One encoding, and whether byte order matched value order over a sample."""

    name: str
    values: tuple
    correct: int
    total: int

    def __post_init__(self) -> None:
        if self.total < 1:
            raise ConfigError(f"{self.total} is not a sample size")

    @property
    def share(self) -> float:
        """The share of pairs the encoding got right."""
        return round(self.correct / self.total, 4)

    def __bool__(self) -> bool:
        """An encoding is order preserving only if it is right about every pair."""
        return self.correct == self.total

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "encoding": self.name,
            "values": len(self.values),
            "pairs": self.total,
            "correct": self.correct,
            "share": self.share,
            "order_preserving": bool(self),
        }


def check_order(name: str, values: tuple, encode) -> Ordering:
    """Compare every pair of values against every pair of their encodings.

    Every pair rather than a sorted list, because a sorted list can come out right while
    individual comparisons are wrong, and it is the comparison that a block index does.
    """
    pairs = 0
    correct = 0
    for left in values:
        for right in values:
            if left == right:
                continue
            pairs += 1
            if (left < right) == (encode(left) < encode(right)):
                correct += 1
    return Ordering(name=name, values=values, correct=correct, total=pairs)


INTS = (-1000, -100, -10, -1, 0, 1, 9, 10, 100, 1000)
FLOATS = (-1e6, -1.5, -0.5, 0.0, 0.5, 1.5, 1e6)
TEXTS = ("", "a", "ab", "b", "A", "z", chr(0x00E9), chr(0x4E2D))


def a_sortable_integer_encoding_is_right_about_every_pair() -> dict:
    """Eight bytes big endian with the sign bit flipped, and ninety pairs out of ninety.

    The base case. Big endian is what makes byte order follow magnitude, and flipping the sign
    bit is what stops minus one, which is all ones, sorting above everything.
    """
    made = check_order("big endian, sign flipped", INTS, encode_int)
    flipped = [decode_int(one) for one in sorted(encode_int(one) for one in INTS)]
    return {
        "values": len(INTS),
        "pairs": made.total,
        "correct": made.correct,
        "it_is_order_preserving": bool(made),
        "sorted_by_bytes": flipped,
        "and_it_matches_sorted": flipped == sorted(INTS),
        "it_round_trips": [decode_int(encode_int(one)) for one in INTS] == list(INTS),
        "width": len(encode_int(0)),
    }


def an_integer_written_as_text_puts_ten_before_nine() -> dict:
    """Eighteen pairs of ninety come out backwards, and the failure is the obvious one.

    The encoding anybody writes first, because str on an integer is right there. It is correct
    for values of the same width and wrong across a width boundary, so a key space that never
    crosses ten is fine and one that crosses a hundred is not, which is the worst way for
    something to be wrong.
    """
    made = check_order("decimal text", INTS, naive_int)
    order = [int(one) for one in sorted(naive_int(one) for one in INTS)]
    wrong = [
        (left, right)
        for left in INTS
        for right in INTS
        if left < right and naive_int(left) > naive_int(right)
    ]
    return {
        "pairs": made.total,
        "correct": made.correct,
        "share": made.share,
        "it_is_not_order_preserving": not bool(made),
        "sorted_by_bytes": order,
        "and_it_is_not_sorted": order != sorted(INTS),
        "a_backwards_pair": list(wrong[0]) if wrong else [],
        "backwards_pairs": len(wrong),
        "and_the_negatives_are_the_worst": all(one[0] < 0 for one in wrong[:3]),
    }


def a_raw_float_is_wrong_more_often_than_it_is_right() -> dict:
    """Twelve pairs of forty two, which is worse than deciding at random.

    The result worth knowing about IEEE. The layout sorts correctly among positive values read
    as unsigned integers, and that is where the good reputation comes from. Every negative has
    its top bit set, so all of them sort above all the positives, and within the negatives a
    larger magnitude is a larger unsigned value, so they sort backwards as well.

    Two errors compounding is what takes it under half. A coin would do better on this sample,
    which is a fair way to say that the encoding carries no information about the order.
    """
    made = check_order("raw ieee", FLOATS, naive_float)
    good = check_order("sortable float", FLOATS, encode_float)
    order = [struct.unpack(">d", one)[0] for one in sorted(naive_float(one) for one in FLOATS)]
    return {
        "pairs": made.total,
        "correct": made.correct,
        "share": made.share,
        "it_is_worse_than_a_coin": made.share < 0.5,
        "sorted_by_bytes": order,
        "the_negatives_sort_above": order[-1] < 0,
        "and_backwards_among_themselves": all(
            order[one] > order[one + 1] for one in range(len(order) - 3, len(order) - 1)
        ),
        "the_fixed_encoding": good.share,
        "and_it_is_right_about_every_pair": bool(good),
    }


def utf_8_needs_no_help_and_utf_16_would() -> dict:
    """Text is the one case where the obvious encoding is already the sortable one.

    Utf-8 was designed so that byte order follows code point order, which is why every string
    key here is stored as it arrives.

    Utf-16 was not, and it takes two particular characters to show it. A code point above the
    basic plane encodes as a surrogate pair whose first unit is in the D800 range, and a basic
    plane character above E000 encodes as itself. So the astral character sorts below the basic
    plane one in bytes and above it in code points, and the sample needs both to catch it: two
    pairs of ninety, which is exactly how a bug like this survives a test suite.
    """
    made = check_order("utf-8", TEXTS, encode_text)
    astral = chr(0x1D400)
    high = chr(0xFF21)
    sample = (*TEXTS, astral, high)
    wide = check_order("utf-16 be", sample, lambda one: one.encode("utf-16-be"))
    narrow = check_order("utf-8", sample, encode_text)
    return {
        "pairs": made.total,
        "correct": made.correct,
        "utf_8_is_order_preserving": bool(made),
        "with_an_astral_character": bool(narrow),
        "utf_16_share": wide.share,
        "and_utf_16_is_not": not bool(wide),
        "the_astral_code_point": f"U+{ord(astral):04X}",
        "the_basic_plane_one": f"U+{ord(high):04X}",
        "utf_16_gets_these_two_wrong": wide.total - wide.correct,
        "which_is_two_pairs_of_ninety": wide.total - wide.correct == 2,
    }


def a_type_tag_keeps_the_types_apart() -> dict:
    """Without a tag an integer key can land inside the range of a string key.

    The reason the tag is a byte at the front rather than something the caller tracks. A key
    space holding both integers and strings has to put them somewhere relative to each other,
    and the choice worth making is a deliberate one rather than whatever the encodings happen to
    produce.

    With the tag, every integer sorts below every string because the tag says so. Without it the
    order is whatever the first byte happens to be, which here puts every integer above every
    string: the sign flipped encoding starts at eighty and ascii starts below it.

    Both are orders. Only one of them was chosen.
    """
    values = (-5, 0, 12, "a", "z", b"raw", None, True)
    tagged = sorted(key_of(one) for one in values)
    kinds = [one[0] for one in tagged]
    untagged_int = encode_int(12)
    untagged_text = encode_text("a")
    return {
        "values": len(values),
        "tags_in_order": kinds,
        "they_are_grouped": kinds == sorted(kinds),
        "and_the_order_is_the_tag_order": all(one in TAGS for one in kinds),
        "round_trips": [value_of(key_of(one)) for one in values if one is not True],
        "an_integer_without_a_tag": untagged_int.hex(),
        "a_string_without_a_tag": untagged_text.hex(),
        "untagged_puts_the_integer_above": untagged_int > untagged_text,
        "which_nobody_chose": True,
        "and_the_tag_puts_it_below": key_of(12) < key_of("a"),
    }


def a_key_too_large_to_frame_is_refused() -> bool:
    """An integer outside the eight byte range is refused rather than truncated."""
    try:
        encode_int(2**64)
    except TooLarge:
        return True
    return False


def a_value_with_no_encoding_is_refused() -> bool:
    """A type the key format cannot express is refused at encode time."""
    try:
        key_of(object())
    except ConfigError:
        return True
    return False


def an_empty_key_has_no_value() -> bool:
    """Decoding nothing is refused rather than answered with None."""
    try:
        value_of(b"")
    except ConfigError:
        return True
    return False


def compare_the_encodings() -> list[dict]:
    """Every encoding against the sample it is meant for."""
    return [
        check_order("integer, big endian", INTS, encode_int).as_dict(),
        check_order("integer, decimal text", INTS, naive_int).as_dict(),
        check_order("float, sortable", FLOATS, encode_float).as_dict(),
        check_order("float, raw ieee", FLOATS, naive_float).as_dict(),
        check_order("text, utf-8", TEXTS, encode_text).as_dict(),
    ]


def the_wrong_encodings_are_wrong_in_different_ways() -> dict:
    """One is wrong at width boundaries and the other is wrong nearly everywhere.

    Worth putting side by side because they fail so differently. Decimal integers are right for
    eighty percent of pairs, so a test over small keys passes and the bug arrives when the data
    grows. Raw floats are right for twenty nine percent, so almost any test catches it.

    The dangerous one is the first. An encoding that is mostly right is one that ships.
    """
    table = {one["encoding"]: one for one in compare_the_encodings()}
    return {
        "encodings": sorted(table),
        "shares": {name: one["share"] for name, one in table.items()},
        "correct_ones": sorted(name for name, one in table.items() if one["order_preserving"]),
        "broken_ones": sorted(
            name for name, one in table.items() if not one["order_preserving"]
        ),
        "the_decimal_one_is_mostly_right": table["integer, decimal text"]["share"] > 0.7,
        "and_the_float_one_is_mostly_wrong": table["float, raw ieee"]["share"] < 0.5,
        "so_the_first_is_the_dangerous_one": True,
    }


def summarise() -> dict:
    """The findings in one mapping."""
    return {
        "types": len(TAGS),
        "integer_encoding_is_exact": (
            a_sortable_integer_encoding_is_right_about_every_pair()["it_is_order_preserving"]
        ),
        "decimal_text_is_not": an_integer_written_as_text_puts_ten_before_nine()[
            "it_is_not_order_preserving"
        ],
        "raw_floats_are_worse_than_a_coin": (
            a_raw_float_is_wrong_more_often_than_it_is_right()["it_is_worse_than_a_coin"]
        ),
        "utf_8_needs_nothing": utf_8_needs_no_help_and_utf_16_would()[
            "utf_8_is_order_preserving"
        ],
        "and_utf_16_would": utf_8_needs_no_help_and_utf_16_would()["and_utf_16_is_not"],
        "the_tag_groups_the_types": a_type_tag_keeps_the_types_apart()["they_are_grouped"],
        "the_mostly_right_one_is_the_dangerous_one": (
            the_wrong_encodings_are_wrong_in_different_ways()[
                "so_the_first_is_the_dangerous_one"
            ]
        ),
    }
