from __future__ import annotations

from dataclasses import dataclass, field

from store.errors import ConfigError

# A file that can lose a crash, modelled rather than mocked.
#
# Everything about durability turns on one distinction: bytes that have been handed to the
# operating system and bytes that have reached the medium. A write puts bytes in the first
# place and a sync moves them to the second, and a crash keeps the second and loses the first.
#
# Real files hide that distinction well enough that it is easy to write code which is correct
# only because the machine did not fall over. This file makes it explicit: a Disk holds durable
# bytes and pending bytes separately, crash drops the pending ones, and nothing else in the
# package is allowed to reach past it.
#
# The model is deliberately harsh in one way and generous in another. Harsh: a crash loses every
# pending byte, where a real crash often keeps a prefix of them. Generous: a sync is atomic and
# a durable byte never changes, where a real medium can tear a sector. The torn write is covered
# separately, so this file can stay simple and say what it assumes.

# The size of a sector, which is the unit a real disk writes atomically.
SECTOR = 512


@dataclass
class Disk:
    """One file: what has reached the medium, and what is only in the buffer."""

    name: str
    durable: bytearray = field(default_factory=bytearray)
    pending: bytearray = field(default_factory=bytearray)
    writes: int = 0
    syncs: int = 0
    bytes_written: int = 0
    bytes_synced: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("a file needs a name")

    @property
    def size(self) -> int:
        """Everything written, whether or not it has reached the medium."""
        return len(self.durable) + len(self.pending)

    @property
    def at_risk(self) -> int:
        """Bytes a crash would lose right now."""
        return len(self.pending)

    def append(self, raw: bytes) -> int:
        """Write bytes, which reach the buffer and no further."""
        self.pending.extend(raw)
        self.writes += 1
        self.bytes_written += len(raw)
        return len(raw)

    def sync(self) -> int:
        """Move everything pending to the medium, and say how much moved."""
        moved = len(self.pending)
        self.durable.extend(self.pending)
        self.pending.clear()
        self.syncs += 1
        self.bytes_synced += moved
        return moved

    def crash(self) -> int:
        """Lose everything that has not been synced, and say how much was lost."""
        lost = len(self.pending)
        self.pending.clear()
        return lost

    def read(self) -> bytes:
        """What a reader sees now, which includes the buffer because the page cache does."""
        return bytes(self.durable + self.pending)

    def read_durable(self) -> bytes:
        """What a reader would see after a crash."""
        return bytes(self.durable)

    def truncate(self, size: int) -> None:
        """Cut the file back, which recovery does after finding a bad record."""
        if size < 0:
            raise ConfigError(f"{size} is not a size")
        if size > self.size:
            raise ConfigError(f"{size} is past the end at {self.size}")
        self.pending.clear()
        del self.durable[size:]

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "file": self.name,
            "size": self.size,
            "durable": len(self.durable),
            "at_risk": self.at_risk,
            "writes": self.writes,
            "syncs": self.syncs,
            "bytes_written": self.bytes_written,
            "bytes_synced": self.bytes_synced,
        }


def a_write_reaches_the_buffer_and_a_sync_reaches_the_medium() -> dict:
    """Five bytes written are at risk; five bytes synced are not.

    The distinction the whole file exists to make explicit. Code that never asks which of the
    two it has is code whose durability depends on nothing crashing, and it looks identical to
    code that asked.
    """
    made = Disk(name="wal")
    made.append(b"hello")
    before = made.as_dict()
    made.sync()
    after = made.as_dict()
    return {
        "after_writing": before["at_risk"],
        "it_is_all_at_risk": before["at_risk"] == 5,
        "and_none_of_it_is_durable": before["durable"] == 0,
        "after_syncing": after["at_risk"],
        "nothing_is_at_risk": after["at_risk"] == 0,
        "and_it_is_all_durable": after["durable"] == 5,
        "the_size_did_not_change": before["size"] == after["size"],
        "which_is_why_a_reader_cannot_tell": True,
    }


def a_reader_cannot_tell_the_difference_and_a_crash_can() -> dict:
    """Reading returns the buffered bytes, so a read after a write always finds them.

    The reason this mistake is so easy to make. A write followed by a read returns what was
    written, whether or not it is durable, because the page cache answers the read. Every test
    that writes and then reads passes, and the only thing that distinguishes the two states is
    a crash, which no test causes by accident.
    """
    made = Disk(name="wal")
    made.append(b"acknowledged")
    seen = made.read()
    lost = made.crash()
    after = made.read()
    return {
        "read_before_the_crash": seen.decode(),
        "it_was_there": seen == b"acknowledged",
        "bytes_lost": lost,
        "read_after_the_crash": after.decode() or "nothing",
        "and_it_is_gone": after == b"",
        "the_durable_view_agreed_all_along": made.read_durable() == b"",
        "so_only_a_crash_separates_them": True,
    }


def syncing_costs_a_call_and_not_a_byte() -> dict:
    """Ten writes and one sync move the same bytes as ten writes and ten syncs.

    What a sync policy actually trades. The bytes are the same either way; what changes is the
    number of calls, and a call is where the cost is, because it waits for the medium.

    This model charges nothing for a sync, which is exactly why the measurements in store.wal
    count syncs rather than timing them. A count is a fact about the policy and a duration would
    be a fact about the machine.
    """
    lazy = Disk(name="lazy")
    eager = Disk(name="eager")
    for _ in range(10):
        lazy.append(b"x" * 100)
        eager.append(b"x" * 100)
        eager.sync()
    lazy.sync()
    return {
        "lazy": lazy.as_dict(),
        "eager": eager.as_dict(),
        "the_same_bytes": lazy.bytes_synced == eager.bytes_synced,
        "lazy_syncs": lazy.syncs,
        "eager_syncs": eager.syncs,
        "and_ten_times_the_calls": eager.syncs == lazy.syncs * 10,
        "the_same_durable_content": lazy.read_durable() == eager.read_durable(),
        "so_the_difference_is_the_window": True,
    }


def truncating_removes_durable_bytes_too() -> dict:
    """Recovery cuts the file back, and the cut has to reach the medium.

    What happens after recovery finds a bad record. The bytes past that point are not merely
    ignored, they are removed, because leaving them means the next writer appends after garbage
    and the next recovery finds a hole in the middle rather than at the end.
    """
    made = Disk(name="wal")
    made.append(b"good record")
    made.sync()
    made.append(b"torn")
    made.sync()
    before = made.size
    made.truncate(len(b"good record"))
    return {
        "before": before,
        "after": made.size,
        "it_shrank": made.size < before,
        "content": made.read().decode(),
        "and_the_good_part_survived": made.read() == b"good record",
        "at_risk": made.at_risk,
        "and_nothing_is_pending": made.at_risk == 0,
    }


def a_truncation_past_the_end_is_refused() -> bool:
    """Cutting a file to a size larger than it has is a bug in the caller."""
    made = Disk(name="wal")
    made.append(b"abc")
    try:
        made.truncate(100)
    except ConfigError:
        return True
    return False


def a_negative_truncation_is_refused() -> bool:
    """A file cannot be cut to less than nothing."""
    try:
        Disk(name="wal").truncate(-1)
    except ConfigError:
        return True
    return False


def a_file_without_a_name_is_refused() -> bool:
    """Every file is named, because a manifest refers to it by name."""
    try:
        Disk(name="")
    except ConfigError:
        return True
    return False


def summarise() -> dict:
    """The findings in one mapping."""
    return {
        "sector": SECTOR,
        "a_write_is_not_durable": a_write_reaches_the_buffer_and_a_sync_reaches_the_medium()[
            "it_is_all_at_risk"
        ],
        "and_a_sync_is": a_write_reaches_the_buffer_and_a_sync_reaches_the_medium()[
            "and_it_is_all_durable"
        ],
        "a_reader_cannot_tell": a_reader_cannot_tell_the_difference_and_a_crash_can()[
            "it_was_there"
        ],
        "and_a_crash_can": a_reader_cannot_tell_the_difference_and_a_crash_can()[
            "and_it_is_gone"
        ],
        "syncing_moves_the_same_bytes": syncing_costs_a_call_and_not_a_byte()["the_same_bytes"],
        "and_differs_in_calls": syncing_costs_a_call_and_not_a_byte()[
            "and_ten_times_the_calls"
        ],
        "truncation_reaches_the_medium": truncating_removes_durable_bytes_too()["it_shrank"],
    }
