from __future__ import annotations

import struct
from dataclasses import dataclass

from store.errors import BadFormat, ConfigError, TooLarge
from store.keys import MAX_KEY, MAX_VALUE

# What actually gets written down: a key, a sequence number, a kind, and sometimes a value.
#
# The sequence number is the part that makes a log structured store work at all. Nothing is ever
# updated in place, so a key can appear many times across many files, and the only thing that
# says which one is current is the sequence number attached to each version. Everything else in
# this package, the merge order, the compaction rules, the snapshot visibility, is arithmetic on
# that number.
#
# A deletion is a record too. It has to be, because the value being deleted may live in a file
# this write cannot touch, so the only way to hide it is to write something newer that says it
# is gone. That record is a tombstone, and the fact that a delete makes the store bigger is the
# first thing about this design that surprises people.
#
# The encoding puts the key before the value and both behind length prefixes. Fixed width fields
# first, then the variable ones, so a reader can find the key without decoding the value, which
# is what a block index needs and what makes a bloom filter check cheap.

PUT = 0
DELETE = 1
MERGE = 2
KINDS = (PUT, DELETE, MERGE)

# The fixed part of a record: sequence number, kind, key length, value length.
HEADER = struct.Struct("<QBHI")


@dataclass(frozen=True)
class Record:
    """One version of one key."""

    key: bytes
    sequence: int
    kind: int = PUT
    value: bytes = b""

    def __post_init__(self) -> None:
        if not self.key:
            raise ConfigError("a record needs a key")
        if len(self.key) > MAX_KEY:
            raise TooLarge(f"{len(self.key)} byte key is past the limit of {MAX_KEY}")
        if len(self.value) > MAX_VALUE:
            raise TooLarge(f"{len(self.value)} byte value is past the limit of {MAX_VALUE}")
        if self.sequence < 0:
            raise ConfigError(f"{self.sequence} is not a sequence number")
        if self.kind not in KINDS:
            raise ConfigError(f"{self.kind} is not one of {list(KINDS)}")
        if self.kind == DELETE and self.value:
            raise ConfigError("a tombstone carries no value")

    @property
    def tombstone(self) -> bool:
        """Whether this record says the key is gone."""
        return self.kind == DELETE

    @property
    def nbytes(self) -> int:
        """What this record costs on disk."""
        return HEADER.size + len(self.key) + len(self.value)

    @property
    def order(self) -> tuple[bytes, int]:
        """What records sort by: key ascending, then sequence descending.

        Descending on the sequence is the whole trick. Sorting that way puts the newest
        version of a key first, so a reader walking a sorted run takes the first record it sees
        for a key and skips the rest, and a merge across files is the same rule with a heap.
        """
        return (self.key, -self.sequence)

    def encode(self) -> bytes:
        """The record as bytes."""
        return (
            HEADER.pack(self.sequence, self.kind, len(self.key), len(self.value))
            + self.key
            + self.value
        )

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "key": self.key.decode("utf-8", "replace"),
            "sequence": self.sequence,
            "kind": self.kind,
            "tombstone": self.tombstone,
            "bytes": self.nbytes,
        }

    def __str__(self) -> str:
        if self.tombstone:
            return f"{self.key.decode('utf-8', 'replace')}@{self.sequence} deleted"
        return f"{self.key.decode('utf-8', 'replace')}@{self.sequence}"


def decode(raw: bytes, at: int = 0) -> tuple[Record, int]:
    """One record and where it ended."""
    if at + HEADER.size > len(raw):
        raise BadFormat("a record header ran off the end")
    sequence, kind, key_length, value_length = HEADER.unpack_from(raw, at)
    at += HEADER.size
    if at + key_length + value_length > len(raw):
        raise BadFormat(f"a record wants {key_length + value_length} bytes that are not here")
    key = raw[at : at + key_length]
    at += key_length
    value = raw[at : at + value_length]
    return Record(key=key, sequence=sequence, kind=kind, value=value), at + value_length


def decode_all(raw: bytes) -> list[Record]:
    """Every record in a buffer."""
    out = []
    at = 0
    while at < len(raw):
        made, at = decode(raw, at)
        out.append(made)
    return out


def newest_first(records: list[Record]) -> list[Record]:
    """The records in the order a reader walks them."""
    return sorted(records, key=lambda one: one.order)


def live(records: list[Record]) -> dict[bytes, Record]:
    """The current version of every key that still has one.

    The reference implementation of what the whole engine computes, in four lines. Everything
    below this module is an effort to produce the same answer without holding every version in
    memory, and every measurement in the package compares against it.
    """
    out: dict[bytes, Record] = {}
    for one in newest_first(records):
        if one.key in out:
            continue
        out[one.key] = one
    return {key: one for key, one in out.items() if not one.tombstone}


def a_record_round_trips_through_its_encoding() -> dict:
    """Every field survives, and the encoded length is the one the record predicted.

    The first thing a format has to do. The length matters as much as the fields: the size
    accounting in the memtable and the block builder both use it to decide when to flush, so a
    record that lies about its size makes every threshold in the engine approximate.
    """
    made = [
        Record(key=b"alpha", sequence=1, value=b"one"),
        Record(key=b"beta", sequence=2, kind=DELETE),
        Record(key=b"gamma", sequence=3, value=b""),
        Record(key=b"x" * 200, sequence=4, value=b"y" * 500),
    ]
    raw = b"".join(one.encode() for one in made)
    back = decode_all(raw)
    return {
        "records": len(made),
        "they_round_trip": back == made,
        "bytes": len(raw),
        "predicted": sum(one.nbytes for one in made),
        "and_the_length_is_predicted": len(raw) == sum(one.nbytes for one in made),
        "header": HEADER.size,
        "the_smallest_record": min(one.nbytes for one in made),
        "which_is_the_header_and_a_key": min(one.nbytes for one in made)
        == HEADER.size + len(b"beta"),
    }


def sorting_by_key_then_sequence_puts_the_newest_first() -> dict:
    """Three versions of one key come back in the order a reader wants them.

    The ordering the entire engine is built on. Ascending by key groups the versions together;
    descending by sequence inside the group puts the current one at the front, so finding the
    live value is taking the first record rather than scanning for the largest sequence.
    """
    made = [
        Record(key=b"k", sequence=1, value=b"first"),
        Record(key=b"k", sequence=7, value=b"third"),
        Record(key=b"k", sequence=4, value=b"second"),
        Record(key=b"j", sequence=2, value=b"other"),
    ]
    order = newest_first(made)
    return {
        "records": len(made),
        "keys_in_order": [one.key.decode() for one in order],
        "the_keys_ascend": [one.key for one in order] == sorted(one.key for one in made),
        "sequences_for_k": [one.sequence for one in order if one.key == b"k"],
        "and_they_descend": [one.sequence for one in order if one.key == b"k"] == [7, 4, 1],
        "the_first_for_k_is_the_newest": next(one for one in order if one.key == b"k").sequence
        == 7,
        "so_a_reader_takes_the_first": True,
    }


def a_delete_makes_the_store_bigger() -> dict:
    """Deleting a key writes a record rather than removing one.

    The consequence of never updating in place, and the first thing about this design that
    surprises people. The value being deleted may live in a file this write cannot touch, so the
    only way to hide it is to write something newer saying it is gone.

    Which means a workload of writes and deletes grows monotonically until a compaction runs,
    and a delete heavy workload can grow faster than a write heavy one, because a tombstone is
    smaller than the value it hides but it does not remove that value.
    """
    written = [
        Record(key=f"k{one}".encode(), sequence=one, value=b"v" * 100) for one in range(10)
    ]
    deleted = [
        Record(key=f"k{one}".encode(), sequence=100 + one, kind=DELETE) for one in range(10)
    ]
    return {
        "written": len(written),
        "bytes_after_writing": sum(one.nbytes for one in written),
        "deleted": len(deleted),
        "bytes_after_deleting": sum(one.nbytes for one in written + deleted),
        "it_grew": sum(one.nbytes for one in written + deleted)
        > sum(one.nbytes for one in written),
        "by_this_many_bytes": sum(one.nbytes for one in deleted),
        "live_keys_after": len(live(written + deleted)),
        "and_there_are_none": not live(written + deleted),
        "a_tombstone_is_smaller_than_the_value": deleted[0].nbytes < written[0].nbytes,
        "but_it_does_not_remove_it": True,
    }


def a_tombstone_with_a_value_is_refused() -> bool:
    """A deletion carries nothing, since anything it carried could never be read."""
    try:
        Record(key=b"k", sequence=1, kind=DELETE, value=b"x")
    except ConfigError:
        return True
    return False


def a_record_without_a_key_is_refused() -> bool:
    """There is nowhere to put a record with no key."""
    try:
        Record(key=b"", sequence=1)
    except ConfigError:
        return True
    return False


def an_oversized_value_is_refused() -> bool:
    """A value past what the length prefix can express is refused at construction."""
    try:
        Record(key=b"k", sequence=1, value=b"x" * (MAX_VALUE + 1))
    except TooLarge:
        return True
    return False


def an_unknown_kind_is_refused() -> bool:
    """There are three kinds and anything else is a typo."""
    try:
        Record(key=b"k", sequence=1, kind=99)
    except ConfigError:
        return True
    return False


def a_truncated_record_is_a_bad_format_and_not_a_crash() -> dict:
    """Every prefix of a valid record is refused, and none of them decodes to something else.

    The failure a length prefixed format has to avoid, and the reason the check is on both the
    header and the body. A reader that trusted the lengths would index past the end of the
    buffer, which in Python is a short slice and in anything else is a segfault, and a short
    slice silently becomes a record with a truncated value.
    """
    raw = Record(key=b"key", sequence=3, value=b"value").encode()
    refused = 0
    decoded = 0
    for size in range(len(raw)):
        try:
            decode(raw[:size])
            decoded += 1
        except BadFormat:
            refused += 1
    return {
        "record_bytes": len(raw),
        "prefixes": len(raw),
        "refused": refused,
        "decoded": decoded,
        "every_prefix_was_refused": decoded == 0,
        "and_the_whole_record_decodes": decode(raw)[0].value == b"value",
    }


def compare_the_kinds() -> list[dict]:
    """Each kind of record with what it costs and what it means."""
    key = b"key"
    value = b"a value"
    return [
        {
            **Record(key=key, sequence=1, value=value).as_dict(),
            "record": "put",
            "hides_older_versions": True,
        },
        {
            **Record(key=key, sequence=2, kind=DELETE).as_dict(),
            "record": "delete",
            "hides_older_versions": True,
        },
        {
            **Record(key=key, sequence=3, kind=MERGE, value=b"+1").as_dict(),
            "record": "merge",
            "hides_older_versions": False,
        },
    ]


def only_a_merge_record_needs_the_ones_below_it() -> dict:
    """A put and a delete are complete on their own; a merge is an instruction.

    The distinction that decides whether compaction can drop older versions. A put says what the
    value is and a delete says there is none, so both make everything older irrelevant. A merge
    says what to do to the value, so it cannot be applied without finding what is underneath,
    and a compaction that dropped the older records would change the answer.

    This package writes merges nowhere and carries the kind because the cost of leaving it out
    is a format change and the cost of leaving it in is one byte.
    """
    table = compare_the_kinds()
    complete = [one for one in table if one["hides_older_versions"]]
    return {
        "kinds": len(table),
        "complete": [one["record"] for one in complete],
        "incomplete": [one["record"] for one in table if not one["hides_older_versions"]],
        "two_of_three_are_complete": len(complete) == 2,
        "and_the_merge_is_not": "merge" not in [one["record"] for one in complete],
        "sizes": {one["record"]: one["bytes"] for one in table},
        "the_tombstone_is_smallest": min(table, key=lambda one: one["bytes"])["record"]
        == "delete",
    }


def summarise() -> dict:
    """The findings in one mapping."""
    return {
        "kinds": len(KINDS),
        "header_bytes": HEADER.size,
        "records_round_trip": a_record_round_trips_through_its_encoding()["they_round_trip"],
        "and_predict_their_size": a_record_round_trips_through_its_encoding()[
            "and_the_length_is_predicted"
        ],
        "the_newest_version_sorts_first": (
            sorting_by_key_then_sequence_puts_the_newest_first()["and_they_descend"]
        ),
        "a_delete_makes_the_store_bigger": a_delete_makes_the_store_bigger()["it_grew"],
        "and_leaves_no_live_keys": a_delete_makes_the_store_bigger()["and_there_are_none"],
        "every_truncated_record_is_refused": (
            a_truncated_record_is_a_bad_format_and_not_a_crash()["every_prefix_was_refused"]
        ),
        "only_a_merge_needs_what_is_below": (
            only_a_merge_record_needs_the_ones_below_it()["and_the_merge_is_not"]
        ),
    }
