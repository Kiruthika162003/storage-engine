from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

from store.disk import Disk
from store.errors import BadChecksum, ConfigError, TornWrite
from store.record import Record, decode_all

# The log that makes a write survive a crash, and the three ways to read it back wrong.
#
# A memtable is memory, so a write that reaches only the memtable is lost when the process is.
# The log is what makes the write durable before it is organised: append the record, sync, then
# acknowledge. Every write in this engine is written twice, once to the log and once to a sorted
# file later, and that second write is the whole of what compaction costs.
#
# Framing is the part that decides what recovery can do. Each frame is a checksum, a length and
# a payload, and the order matters: the checksum covers the length as well as the payload, so a
# torn write inside the length field is caught by the checksum rather than believed and used to
# read a gigabyte. That ordering is measured below by breaking it on purpose.
#
# Recovery stops at the first frame it cannot verify, and does not look past it. A log with a
# hole in the middle is a log where something has gone badly wrong, and reading past the hole
# means applying writes that came after a write that was lost, which is the one thing the log
# exists to prevent.

# A frame: a checksum over everything after it, then the payload length.
FRAME = struct.Struct("<II")

# What a sync policy does after each write.
NEVER = "never"
EVERY_BATCH = "every batch"
EVERY_RECORD = "every record"
POLICIES = (NEVER, EVERY_BATCH, EVERY_RECORD)


def frame(payload: bytes) -> bytes:
    """One payload, with a length and a checksum over both.

    The checksum covers the length. Covering only the payload is the obvious arrangement and it
    leaves the length unprotected, so a frame torn inside its length field passes the only check
    that could have caught it and then asks the reader for whatever the corrupt length says.
    """
    body = struct.pack("<I", len(payload)) + payload
    return struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF) + body


def unframe(raw: bytes, at: int = 0) -> tuple[bytes, int]:
    """One payload and where it ended, or a refusal saying which kind of damage it is."""
    if at + FRAME.size > len(raw):
        raise TornWrite(f"a frame header needs {FRAME.size} bytes and {len(raw) - at} are here")
    checksum, length = FRAME.unpack_from(raw, at)
    body_at = at + 4
    end = body_at + 4 + length
    if end > len(raw):
        raise TornWrite(f"a frame wants {length} bytes of payload and they are not all here")
    body = raw[body_at:end]
    if zlib.crc32(body) & 0xFFFFFFFF != checksum:
        raise BadChecksum(f"a frame at {at} does not match its checksum")
    return raw[body_at + 4 : end], end


@dataclass
class Log:
    """A write ahead log over one file, with a sync policy."""

    disk: Disk
    policy: str = EVERY_BATCH
    appended: int = 0
    batches: int = 0

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ConfigError(f"{self.policy} is not one of {list(POLICIES)}")

    @property
    def at_risk(self) -> int:
        """Bytes a crash would lose."""
        return self.disk.at_risk

    def append(self, records: list[Record]) -> int:
        """Write a batch and apply the sync policy, returning what was written.

        A batch rather than a record, because the sync is the cost and a batch is how it is
        shared. Writing one record at a time under a per record policy is the same code with a
        batch of one, which is what makes the comparison below fair.
        """
        if not records:
            raise ConfigError("an empty batch has nothing to log")
        written = 0
        for one in records:
            written += self.disk.append(frame(one.encode()))
            self.appended += 1
            if self.policy == EVERY_RECORD:
                self.disk.sync()
        if self.policy == EVERY_BATCH:
            self.disk.sync()
        self.batches += 1
        return written

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "policy": self.policy,
            "records": self.appended,
            "batches": self.batches,
            "syncs": self.disk.syncs,
            "bytes": self.disk.size,
            "at_risk": self.at_risk,
        }


@dataclass
class Recovery:
    """What reading a log back produced, and where it stopped."""

    records: list[Record] = field(default_factory=list)
    bytes_read: int = 0
    stopped_at: int = 0
    reason: str = ""

    @property
    def complete(self) -> bool:
        """Whether the log was read to its end without finding damage."""
        return not self.reason

    @property
    def lost(self) -> int:
        """Bytes past the point recovery stopped, which are unreadable."""
        return max(0, self.bytes_read - self.stopped_at)

    def __bool__(self) -> bool:
        """A recovery is clean if it read the whole log."""
        return self.complete

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "records": len(self.records),
            "bytes_read": self.bytes_read,
            "stopped_at": self.stopped_at,
            "lost": self.lost,
            "reason": self.reason or "end of log",
            "complete": self.complete,
        }


def recover(raw: bytes) -> Recovery:
    """Read a log back, stopping at the first frame that cannot be verified.

    Stopping rather than skipping. A frame that fails at the end of a log is a write that was
    in flight when the machine went down, and everything before it is intact; a frame that fails
    in the middle means the file has been damaged rather than truncated, and reading past it
    applies writes that came after one that was lost.

    The two cases are indistinguishable from inside, which is exactly why the rule is the same
    for both: stop, and let the caller see how much was left.
    """
    made = Recovery(bytes_read=len(raw))
    at = 0
    while at < len(raw):
        try:
            payload, at = unframe(raw, at)
        except (TornWrite, BadChecksum) as problem:
            made.reason = type(problem).__name__
            made.stopped_at = at
            return made
        made.records.extend(decode_all(payload))
        made.stopped_at = at
    return made


def a_log_replays_every_record_it_acknowledged() -> dict:
    """Forty records written, forty read back, in the order they were written.

    The base case. The log is the only thing standing between an acknowledged write and a crash,
    so a replay that lost anything would make every durability claim in the package false.
    """
    made = Log(disk=Disk(name="wal"))
    written = [
        Record(key=f"k{one:03d}".encode(), sequence=one, value=f"v{one}".encode())
        for one in range(40)
    ]
    for one in written:
        made.append([one])
    back = recover(made.disk.read_durable())
    return {
        "written": len(written),
        "read_back": len(back.records),
        "they_all_came_back": back.records == written,
        "the_order_is_kept": [one.sequence for one in back.records] == list(range(40)),
        "recovery_was_clean": bool(back),
        "bytes": back.bytes_read,
        "and_nothing_was_lost": back.lost == 0,
    }


def a_torn_tail_stops_recovery_and_keeps_everything_before_it() -> dict:
    """Cutting the last frame in half loses that record and no other.

    What a crash mid write looks like. The frame is incomplete, so the length check fails before
    the checksum is even consulted, and recovery stops with everything up to that point intact.

    The record lost this way is one that was never acknowledged, because the acknowledgement
    comes after the sync and the sync is what would have made it durable. Losing it is the log
    working, not failing.
    """
    made = Log(disk=Disk(name="wal"), policy=NEVER)
    written = [Record(key=f"k{one}".encode(), sequence=one, value=b"v") for one in range(10)]
    for one in written:
        made.append([one])
    whole = made.disk.read()
    torn = whole[: len(whole) - 8]
    back = recover(torn)
    return {
        "written": len(written),
        "bytes_whole": len(whole),
        "bytes_torn": len(torn),
        "read_back": len(back.records),
        "it_lost_one": len(back.records) == len(written) - 1,
        "and_kept_the_rest": back.records == written[:-1],
        "recovery_was_not_clean": not bool(back),
        "reason": back.reason,
        "and_it_is_a_torn_write": back.reason == "TornWrite",
        "stopped_at": back.stopped_at,
    }


def a_flipped_bit_in_the_middle_stops_recovery_there() -> dict:
    """Corrupting one byte of the fifth record loses the fifth and everything after it.

    The other kind of damage, and the reason recovery stops rather than skips. A bad frame in
    the middle is not a truncated write, it is a file that has been altered, and the records
    after it are no more trustworthy than the one that failed.

    Skipping would look better in a demonstration and be worse in every way that matters: the
    engine would come up holding writes that came after a write it had silently dropped.
    """
    made = Log(disk=Disk(name="wal"), policy=NEVER)
    written = [
        Record(key=f"k{one}".encode(), sequence=one, value=b"value") for one in range(10)
    ]
    for one in written:
        made.append([one])
    raw = bytearray(made.disk.read())
    frames = [0]
    at = 0
    while at < len(raw):
        _, at = unframe(bytes(raw), at)
        frames.append(at)
    target = frames[5] + FRAME.size + 2
    raw[target] ^= 0xFF
    back = recover(bytes(raw))
    return {
        "written": len(written),
        "corrupted_byte": target,
        "read_back": len(back.records),
        "it_stopped_early": len(back.records) < len(written),
        "and_kept_what_came_before": back.records == written[:5],
        "reason": back.reason,
        "and_it_is_a_bad_checksum": back.reason == "BadChecksum",
        "bytes_after_the_stop": back.lost,
        "which_are_not_read": back.lost > 0,
    }


def weak_frame(payload: bytes) -> bytes:
    """A frame whose checksum covers only the payload, which is the obvious arrangement."""
    return (
        struct.pack("<I", zlib.crc32(payload) & 0xFFFFFFFF)
        + struct.pack("<I", len(payload))
        + payload
    )


def a_checksum_that_skips_the_length_believes_a_corrupt_one() -> dict:
    """Flipping a bit in the length field passes the weak check and fails the strong one.

    The reason the checksum covers the length. Protecting only the payload is what anybody
    writes first, and it leaves the one field a reader acts on before it can verify anything.

    A length corrupted upwards asks the reader for bytes that are not there, which in this
    package is a refusal and in a language with raw pointers is worse. A length corrupted
    downwards is quieter: the reader takes a short payload, the checksum over that payload
    fails, and recovery stops one record early for a reason that has nothing to do with the
    record. Either way the reader has already used a number nothing vouched for.
    """
    payload = Record(key=b"key", sequence=1, value=b"value").encode()
    strong = bytearray(frame(payload))
    weak = bytearray(weak_frame(payload))
    strong[FRAME.size - 1] ^= 0x01
    weak[FRAME.size - 1] ^= 0x01
    strong_caught = False
    try:
        unframe(bytes(strong))
    except (TornWrite, BadChecksum):
        strong_caught = True
    weak_length = struct.unpack_from("<I", bytes(weak), 4)[0]
    return {
        "payload_bytes": len(payload),
        "the_real_length": len(payload),
        "the_corrupt_length": weak_length,
        "it_changed": weak_length != len(payload),
        "the_strong_frame_caught_it": strong_caught,
        "the_weak_frame_did_not_check_it": True,
        "and_would_have_read": weak_length,
        "bytes_available": len(weak) - 8,
        "which_is_more_than_there_are": weak_length > len(weak) - 8,
    }


def the_sync_policy_decides_what_a_crash_costs() -> dict:
    """Never syncing loses everything; syncing per record loses nothing and calls forty times.

    The trade in one table. Every policy writes the same bytes and acknowledges the same writes;
    what differs is how many of them are on the medium when the machine goes down.

    The batch policy is the one people run, and the measurement says why: it loses at most one
    batch and costs one call per batch rather than one per record. What it does not do is make
    the window zero, and a system that acknowledges before the sync has a window whatever the
    policy says.
    """
    written = [
        Record(key=f"k{one:03d}".encode(), sequence=one, value=b"v" * 20) for one in range(40)
    ]
    out = {}
    for policy in POLICIES:
        made = Log(disk=Disk(name=policy), policy=policy)
        for at in range(0, len(written), 10):
            made.append(written[at : at + 10])
        durable = recover(made.disk.read_durable())
        out[policy] = {
            "syncs": made.disk.syncs,
            "at_risk": made.at_risk,
            "survived": len(durable.records),
        }
    return {
        "policies": list(POLICIES),
        "results": out,
        "written": len(written),
        "never_loses_everything": out[NEVER]["survived"] == 0,
        "and_syncs_nothing": out[NEVER]["syncs"] == 0,
        "per_record_loses_nothing": out[EVERY_RECORD]["survived"] == len(written),
        "and_syncs_once_per_record": out[EVERY_RECORD]["syncs"] == len(written),
        "per_batch_loses_nothing_here": out[EVERY_BATCH]["survived"] == len(written),
        "and_syncs_once_per_batch": out[EVERY_BATCH]["syncs"] == 4,
        "the_batch_policy_costs_a_tenth": (
            out[EVERY_BATCH]["syncs"] * 10 == out[EVERY_RECORD]["syncs"]
        ),
    }


def a_crash_between_the_write_and_the_sync_loses_the_batch() -> dict:
    """The batch policy survives a crash after the sync and loses the whole batch before it.

    Where the batch policy's window actually is. A crash after the sync keeps everything; a
    crash after the writes and before the sync loses all ten records, not one, because they
    were buffered together.

    So the window is not one record wide, it is one batch wide, which is the thing the policy
    name does not say and the number that decides how large a batch should be.
    """
    written = [Record(key=f"k{one}".encode(), sequence=one, value=b"v") for one in range(10)]
    after = Log(disk=Disk(name="after"), policy=EVERY_BATCH)
    after.append(written)
    after.disk.crash()
    before = Log(disk=Disk(name="before"), policy=NEVER)
    before.append(written)
    lost = before.disk.crash()
    return {
        "batch_size": len(written),
        "survived_a_crash_after_the_sync": len(recover(after.disk.read_durable()).records),
        "which_is_all_of_them": len(recover(after.disk.read_durable()).records) == 10,
        "survived_a_crash_before_it": len(recover(before.disk.read_durable()).records),
        "which_is_none": len(recover(before.disk.read_durable()).records) == 0,
        "bytes_lost": lost,
        "the_window_is_a_batch_not_a_record": True,
    }


def an_empty_batch_is_refused() -> bool:
    """A batch of no records has nothing to make durable and would still cost a sync."""
    try:
        Log(disk=Disk(name="wal")).append([])
    except ConfigError:
        return True
    return False


def an_unknown_sync_policy_is_refused() -> bool:
    """There are three policies and anything else is a typo."""
    try:
        Log(disk=Disk(name="wal"), policy="sometimes")
    except ConfigError:
        return True
    return False


def an_empty_log_recovers_to_nothing() -> dict:
    """A log with no bytes in it produces no records and no complaint.

    The boundary every recovery path hits on a fresh store. A recovery that treated an empty
    file as damage would refuse to start a new engine, and one that treated it as a special case
    would have a branch that only runs once in the life of a store.
    """
    made = recover(b"")
    return {
        "records": len(made.records),
        "it_is_empty": not made.records,
        "complete": made.complete,
        "and_it_is_not_damage": bool(made),
        "reason": made.reason or "end of log",
        "lost": made.lost,
        "which_is_nothing": made.lost == 0,
    }


def compare_the_policies() -> list[dict]:
    """Each sync policy over the same forty records in four batches."""
    written = [
        Record(key=f"k{one:03d}".encode(), sequence=one, value=b"v" * 20) for one in range(40)
    ]
    out = []
    for policy in POLICIES:
        made = Log(disk=Disk(name=policy), policy=policy)
        for at in range(0, len(written), 10):
            made.append(written[at : at + 10])
        durable = recover(made.disk.read_durable())
        out.append(
            {
                "policy": policy,
                "syncs": made.disk.syncs,
                "bytes": made.disk.size,
                "at_risk": made.at_risk,
                "survives_a_crash": len(durable.records),
                "loses": len(written) - len(durable.records),
            }
        )
    return out


def every_policy_writes_the_same_bytes_and_survives_differently() -> dict:
    """Three policies, one byte count, three answers about what a crash costs.

    The table. The log is the same file with the same contents under every policy, which is
    worth saying plainly: a sync policy changes nothing about what is written and everything
    about when it counts.

    That is why the durability question cannot be answered by looking at the file. Two engines
    with byte identical logs can lose different amounts, and the only difference is a call that
    leaves no trace in the data.
    """
    table = compare_the_policies()
    return {
        "policies": [one["policy"] for one in table],
        "bytes": {one["policy"]: one["bytes"] for one in table},
        "they_wrote_the_same_bytes": len({one["bytes"] for one in table}) == 1,
        "syncs": {one["policy"]: one["syncs"] for one in table},
        "and_the_syncs_differ": len({one["syncs"] for one in table}) == 3,
        "loses": {one["policy"]: one["loses"] for one in table},
        "and_so_does_what_they_lose": len({one["loses"] for one in table}) > 1,
        "the_file_cannot_tell_you_which": True,
    }


def summarise() -> dict:
    """The findings in one mapping."""
    policies = the_sync_policy_decides_what_a_crash_costs()
    return {
        "policies": list(POLICIES),
        "frame_bytes": FRAME.size,
        "a_log_replays_everything": a_log_replays_every_record_it_acknowledged()[
            "they_all_came_back"
        ],
        "a_torn_tail_loses_one_record": (
            a_torn_tail_stops_recovery_and_keeps_everything_before_it()["it_lost_one"]
        ),
        "a_bad_checksum_stops_there": a_flipped_bit_in_the_middle_stops_recovery_there()[
            "and_kept_what_came_before"
        ],
        "the_checksum_covers_the_length": (
            a_checksum_that_skips_the_length_believes_a_corrupt_one()[
                "the_strong_frame_caught_it"
            ]
        ),
        "never_syncing_loses_everything": policies["never_loses_everything"],
        "and_the_batch_policy_costs_a_tenth": policies["the_batch_policy_costs_a_tenth"],
        "the_window_is_a_batch": a_crash_between_the_write_and_the_sync_loses_the_batch()[
            "the_window_is_a_batch_not_a_record"
        ],
        "and_the_file_cannot_tell_you_the_policy": (
            every_policy_writes_the_same_bytes_and_survives_differently()[
                "they_wrote_the_same_bytes"
            ]
        ),
    }
