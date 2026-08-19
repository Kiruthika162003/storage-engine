from __future__ import annotations

import functools
import random
import struct
from dataclasses import dataclass, field

from store.errors import BadFormat, ConfigError
from store.record import Record

# The unit a sorted file is read in, and the two things that make it small.
#
# A sorted file is a sequence of blocks, each a few kilobytes, because that is the unit a disk
# hands over and the unit a cache holds. Inside a block the records are sorted, which makes two
# savings available that a random arrangement would not.
#
# The first is prefix compression. Adjacent keys in sorted order share a prefix far more often
# than adjacent keys in any other order, so each record stores how many bytes it shares with the
# one before and only the bytes that differ. That is free space on sorted data and nothing
# at all on data that happens not to share prefixes, which the measurement below quantifies.
#
# The second is the restart point. Prefix compression makes a record unreadable without the
# one before it, so a block that used it everywhere could only be scanned from the start.
# Every so many records the compression is reset and the full key written, and those positions
# are recorded at the end of the block, so a search can binary search the restarts and then
# scan a short run. The interval is the trade and it is measured rather than assumed.

# How many records share a prefix run before the key is written in full.
RESTART_INTERVAL = 16

# The size a block is filled to before it is closed.
BLOCK_BYTES = 4096

# One record inside a block: shared prefix length, the rest of the key, the value.
ENTRY = struct.Struct("<HHIQB")


@dataclass
class Block:
    """A sorted run of records, prefix compressed, with restart points at the end."""

    payload: bytes
    restarts: tuple[int, ...]
    count: int

    def __post_init__(self) -> None:
        if not self.restarts:
            raise ConfigError("a block needs at least one restart point")
        if self.count < 1:
            raise ConfigError(f"{self.count} is not a record count")

    @property
    def nbytes(self) -> int:
        """What the block costs, payload and restart array together."""
        return len(self.payload) + len(self.restarts) * 4 + 4

    @property
    def interval(self) -> float:
        """How many records there are per restart point."""
        return round(self.count / len(self.restarts), 2)

    def records(self) -> list[Record]:
        """Everything in the block, in order."""
        return list(self.scan())

    def scan(self, start: bytes = b""):
        """Every record from a key onwards.

        The scan begins at the restart point below the key rather than at the start of the
        block, which is the whole reason the restarts exist. Without them a lookup near the end
        of a block decodes every record before it.
        """
        at = self.restart_below(start) if start else self.restarts[0]
        previous = b""
        while at < len(self.payload):
            record, previous, at = self.decode_at(at, previous)
            if record.key >= start:
                yield record

    def restart_below(self, key: bytes) -> int:
        """The last restart point whose key is at or below the given key."""
        low, high = 0, len(self.restarts) - 1
        best = self.restarts[0]
        while low <= high:
            middle = (low + high) // 2
            found, _, _ = self.decode_at(self.restarts[middle], b"")
            if found.key <= key:
                best = self.restarts[middle]
                low = middle + 1
            else:
                high = middle - 1
        return best

    def decode_at(self, at: int, previous: bytes) -> tuple[Record, bytes, int]:
        """One record, the key it leaves behind, and where it ended."""
        if at + ENTRY.size > len(self.payload):
            raise BadFormat("a block entry ran off the end")
        shared, rest, value_length, sequence, kind = ENTRY.unpack_from(self.payload, at)
        at += ENTRY.size
        if shared > len(previous):
            raise BadFormat(f"an entry shares {shared} bytes of a {len(previous)} byte key")
        key = previous[:shared] + self.payload[at : at + rest]
        at += rest
        value = self.payload[at : at + value_length]
        at += value_length
        return Record(key=key, sequence=sequence, kind=kind, value=value), key, at

    def get(self, key: bytes) -> Record | None:
        """The record for an exact key, or nothing."""
        for one in self.scan(key):
            if one.key == key:
                return one
            if one.key > key:
                return None
        return None

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "records": self.count,
            "bytes": self.nbytes,
            "payload": len(self.payload),
            "restarts": len(self.restarts),
            "interval": self.interval,
            "bytes_per_record": round(self.nbytes / self.count, 1),
        }


class Builder:
    """Builds one block, resetting the prefix compression at each restart."""

    def __init__(self, interval: int = RESTART_INTERVAL):
        if interval < 1:
            raise ConfigError(f"{interval} is not a restart interval")
        self.interval = interval
        self.payload = bytearray()
        self.restarts: list[int] = []
        self.count = 0
        self.previous = b""
        self.shared_bytes = 0
        self.key_bytes = 0

    def add(self, record: Record) -> None:
        """Append a record, sharing with the one before unless this is a restart."""
        if self.count and record.key <= self.previous:
            raise ConfigError(f"{record.key!r} does not follow {self.previous!r}")
        restart = self.count % self.interval == 0
        if restart:
            self.restarts.append(len(self.payload))
            shared = 0
        else:
            shared = _shared(self.previous, record.key)
        rest = record.key[shared:]
        self.payload.extend(
            ENTRY.pack(shared, len(rest), len(record.value), record.sequence, record.kind)
        )
        self.payload.extend(rest)
        self.payload.extend(record.value)
        self.previous = record.key
        self.count += 1
        self.shared_bytes += shared
        self.key_bytes += len(record.key)

    @property
    def full(self) -> bool:
        """Whether the block has reached the size it is closed at."""
        return len(self.payload) >= BLOCK_BYTES

    def finish(self) -> Block:
        """Close the block."""
        if not self.count:
            raise ConfigError("an empty block has nothing to close")
        return Block(
            payload=bytes(self.payload), restarts=tuple(self.restarts), count=self.count
        )


def _shared(left: bytes, right: bytes) -> int:
    """How many leading bytes two keys have in common."""
    limit = min(len(left), len(right))
    at = 0
    while at < limit and left[at] == right[at]:
        at += 1
    return at


def build(records: list[Record], interval: int = RESTART_INTERVAL) -> Block:
    """One block from a sorted run of records."""
    made = Builder(interval=interval)
    for one in records:
        made.add(one)
    return made.finish()


@dataclass
class Cost:
    """What a block costs to hold and what it costs to reach a key inside it."""

    interval: int
    bytes: int
    saved: int
    decoded: int

    @property
    def ratio(self) -> float:
        """How much of the key bytes the prefix sharing removed."""
        return round(self.saved / max(self.bytes + self.saved, 1), 4)

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "interval": self.interval,
            "bytes": self.bytes,
            "saved": self.saved,
            "ratio": self.ratio,
            "decoded": self.decoded,
        }


@dataclass
class Reader:
    """A block that counts what a lookup decodes, so the restart trade can be measured."""

    block: Block
    decoded: int = field(default=0)

    def get(self, key: bytes) -> Record | None:
        """The record for a key, counting entries decoded on the way."""
        at = self.block.restart_below(key)
        previous = b""
        while at < len(self.block.payload):
            record, previous, at = self.block.decode_at(at, previous)
            self.decoded += 1
            if record.key == key:
                return record
            if record.key > key:
                return None
        return None


@functools.cache
def _sorted_keys(count: int, prefix: bytes = b"user:") -> tuple[bytes, ...]:
    """Keys that share a prefix and a numeric run, which is what real keys look like."""
    return tuple(prefix + f"{one:012d}".encode() for one in range(count))


@functools.cache
def _random_keys(count: int, seed: int = 7) -> tuple[bytes, ...]:
    """Keys with nothing in common, which is what hashed keys look like."""
    source = random.Random(seed)
    made: set[bytes] = set()
    while len(made) < count:
        made.add(source.randbytes(17))
    return tuple(sorted(made))


def _records(keys: tuple[bytes, ...], value: int = 8) -> list[Record]:
    """Records over a run of keys, all puts, sequences in order."""
    return [
        Record(key=key, sequence=one + 1, value=bytes(value)) for one, key in enumerate(keys)
    ]


@functools.cache
def measure(count: int, interval: int, kind: str = "sorted") -> Cost:
    """Build a block at one interval and probe every key in it."""
    keys = _sorted_keys(count) if kind == "sorted" else _random_keys(count)
    made = Builder(interval=interval)
    for record in _records(keys):
        made.add(record)
    block = made.finish()
    reader = Reader(block=block)
    for key in keys:
        reader.get(key)
    return Cost(
        interval=interval,
        bytes=block.nbytes,
        saved=made.shared_bytes,
        decoded=reader.decoded,
    )


@functools.cache
def prefix_sharing_pays_on_sorted_keys_and_not_on_hashed_ones() -> bool:
    """Sorted keys share nearly all of their bytes and random keys share none.

    Ten thousand keys of the form user:000000000000 give a block of 273,459 bytes, of which the
    sharing removed 149,045, so 35 percent of the block never had to exist. The same count of
    random seventeen byte keys gives 412,698 bytes and the sharing removed 9,806, which is 2
    percent and is entirely the accident of two keys starting with the same byte.

    That is the argument against hashing a key before storing it. A hash spreads the keys
    evenly, which sounds like it helps, and it destroys every prefix in the process, which
    costs the space the spreading was supposed to save.
    """
    ordered = measure(10000, RESTART_INTERVAL, "sorted")
    hashed = measure(10000, RESTART_INTERVAL, "random")
    return ordered.ratio > 0.2 and hashed.ratio < 0.05


@functools.cache
def the_restart_interval_trades_space_against_the_scan() -> bool:
    """The two ends of the interval are the two things a block can be bad at.

    At an interval of one every key is stored in full, nothing is shared, the block is 460,004
    bytes and every lookup decodes exactly one entry. At one thousand the block is 261,284,
    which is 43 percent smaller, and probing every key decodes 5,005,000 entries not 10,000.
    Both are blocks, both answer correctly, and the two ends differ by 500 times on read work.

    Sixteen is not a derived number and nothing here says it is optimal. What the measurement
    says is that the curve is steep at the left and flat in the middle, so the cost of being
    somewhat wrong about the interval is small and the cost of being at the wrong end is not.
    """
    tight = measure(10000, 1, "sorted")
    loose = measure(10000, 1000, "sorted")
    return (
        tight.saved == 0
        and tight.decoded == 10000
        and loose.bytes < tight.bytes
        and loose.decoded > tight.decoded * 300
    )


@functools.cache
def the_middle_of_the_interval_curve_is_flat() -> bool:
    """Four through thirty two are close on space and far apart on read work.

    From four to thirty two the block goes 310,504 to 267,295 bytes, a spread of 14 percent
    across an eightfold change, while the decode count goes 25,000 to 164,872, a factor of six
    and a half. The interval is a knob that mostly controls read work and barely controls size.

    That is the useful shape. If someone tunes the interval expecting to save space they will
    move it a long way and find nothing, and they will have made every lookup slower.
    """
    sizes = [measure(10000, one, "sorted").bytes for one in (4, 8, 16, 32)]
    reads = [measure(10000, one, "sorted").decoded for one in (4, 8, 16, 32)]
    return max(sizes) < min(sizes) * 1.2 and max(reads) > min(reads) * 5


@functools.cache
def a_binary_search_over_restarts_beats_the_scan_it_replaces() -> bool:
    """Finding the last key in a block costs a binary search, not a walk of the block.

    Without restart points a lookup for the last of ten thousand keys decodes ten thousand
    entries because prefix compression makes every earlier entry a prerequisite. With restarts
    every sixteen it decodes at most sixteen, after a binary search over six hundred and twenty
    five positions that costs ten comparisons.

    The restart array is a small part of the block. That is what it buys.
    """
    keys = _sorted_keys(10000)
    block = build(_records(keys), interval=RESTART_INTERVAL)
    reader = Reader(block=block)
    reader.get(keys[-1])
    return reader.decoded <= RESTART_INTERVAL


@functools.cache
def a_block_never_shares_across_a_restart() -> bool:
    """Every restart position decodes on its own, which is what makes the search legal.

    A restart entry is decoded with an empty previous key, so if the builder ever shared across
    one the key read back would be a suffix of the real key and the search would land in the
    wrong place. Checking it directly rather than trusting the modulo: every restart offset
    decodes to the key the full scan has at that position.
    """
    keys = _sorted_keys(2000)
    block = build(_records(keys), interval=RESTART_INTERVAL)
    for one, at in enumerate(block.restarts):
        found, _, _ = block.decode_at(at, b"")
        if found.key != keys[one * RESTART_INTERVAL]:
            return False
    return True


@functools.cache
def the_value_dominates_once_it_is_large() -> bool:
    """Prefix compression is a key optimisation and keys are usually the small half.

    The same ten thousand keys with an eight byte value give 273,459 bytes, of which the sharing
    removed 149,045. With a two hundred byte value the block is ten times that and the sharing
    removed the same 149,045, which has gone from 35 percent of the block to under 7.

    So the compression is worth most on the workload that needs it least, small values, and
    nearly nothing on large ones. That is not a reason to drop it. It is a reason not to count
    on it.
    """
    keys = _sorted_keys(10000)
    small = build(_records(keys, value=8))
    large = build(_records(keys, value=200))
    return large.nbytes > small.nbytes * 5


def compare_the_intervals(count: int = 10000) -> list[dict]:
    """A row per restart interval, space and decode work together."""
    return [
        measure(count, one, "sorted").as_dict() for one in (1, 2, 4, 8, 16, 32, 64, 256, 1000)
    ]


def compare_the_key_shapes(count: int = 10000) -> list[dict]:
    """A row per key shape, showing what the sharing is worth on each."""
    return [
        {"keys": kind, **measure(count, RESTART_INTERVAL, kind).as_dict()}
        for kind in ("sorted", "random")
    ]


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "sharing_pays_on_sorted_keys": (
            prefix_sharing_pays_on_sorted_keys_and_not_on_hashed_ones()
        ),
        "interval_trades_space_for_scan": the_restart_interval_trades_space_against_the_scan(),
        "middle_of_the_curve_is_flat": the_middle_of_the_interval_curve_is_flat(),
        "binary_search_beats_the_walk": (
            a_binary_search_over_restarts_beats_the_scan_it_replaces()
        ),
        "no_sharing_across_a_restart": a_block_never_shares_across_a_restart(),
        "the_value_dominates_when_large": the_value_dominates_once_it_is_large(),
    }
