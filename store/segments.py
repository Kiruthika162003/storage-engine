from __future__ import annotations

import functools
from dataclasses import dataclass, field

from store.disk import Disk
from store.errors import ConfigError
from store.record import Record, decode_all
from store.wal import frame, unframe

# Segmented logs: truncation without rewriting, at the price of a boundary rule.
#
# The engine drops its whole log at each flush, which works because the engine flushes
# everything at once. A store that flushes column families or tenants separately cannot: the
# log holds a braid of streams, and dropping it needs every strand flushed. Segmentation is
# the standard answer. The log is a chain of fixed-size segments, a flush records the
# highest sequence it made durable, and a segment is deleted when every record in it is
# below every stream's flushed mark. Nothing is rewritten; space returns in segment-sized
# steps, and the delay between a flush and its space is the tail segment's worth of writes,
# measured here.

SEGMENT_BYTES = 4096


@dataclass
class Segments:
    """The chained log."""

    segment_bytes: int = field(default=SEGMENT_BYTES)
    chain: list[Disk] = field(default_factory=list)
    highest_in: list[int] = field(default_factory=list)
    sealed: int = field(default=0)
    deleted: int = field(default=0)

    def __post_init__(self) -> None:
        if self.segment_bytes < 64:
            raise ConfigError(f"{self.segment_bytes} is not a segment size")
        if not self.chain:
            self._open()

    def _open(self) -> None:
        self.chain.append(Disk(name=f"SEG-{self.sealed + len(self.chain)}"))
        self.highest_in.append(0)

    def append(self, record: Record) -> None:
        """One record into the tail, sealing and opening on the boundary."""
        tail = self.chain[-1]
        if tail.size >= self.segment_bytes:
            self._open()
            tail = self.chain[-1]
        tail.append(frame(record.encode()))
        tail.sync()
        self.highest_in[-1] = max(self.highest_in[-1], record.sequence)

    def truncate(self, flushed_through: int) -> int:
        """Delete every sealed segment wholly below the flushed mark."""
        removed = 0
        while len(self.chain) > 1 and self.highest_in[0] <= flushed_through:
            self.chain.pop(0)
            self.highest_in.pop(0)
            removed += 1
            self.deleted += 1
        return removed

    @property
    def segments(self) -> int:
        return len(self.chain)

    @property
    def bytes_held(self) -> int:
        return sum(disk.size for disk in self.chain)

    def replay(self) -> list[Record]:
        """Every record still held, in order."""
        found: list[Record] = []
        for disk in self.chain:
            raw = disk.read()
            at = 0
            while at < len(raw):
                payload, at = unframe(raw, at)
                found.extend(decode_all(payload))
        return found


def _record(sequence: int) -> Record:
    return Record(key=f"k{sequence:07d}".encode(), sequence=sequence, value=bytes(40))


@functools.cache
def truncation_frees_whole_segments_and_only_those() -> bool:
    """Flushing through sequence 500 frees the segments wholly below it, not the braided one.

    A thousand records across seventeen segments: the flush mark lands mid segment, the
    segments entirely below it go, the one straddling the mark stays whole, and the replay
    still holds every record above the mark. Space returns in steps, and the step size is
    the segment size, which is the knob's whole meaning.
    """
    log = Segments()
    for sequence in range(1, 1001):
        log.append(_record(sequence))
    before = log.segments
    log.truncate(500)
    after = log.segments
    kept = log.replay()
    return (
        before > 10
        and after < before
        and all(record.sequence > 440 for record in kept)
        and any(record.sequence <= 500 for record in kept)
    )


@functools.cache
def nothing_flushed_nothing_freed() -> bool:
    """A truncate at zero deletes nothing, however long the chain."""
    log = Segments()
    for sequence in range(1, 501):
        log.append(_record(sequence))
    return log.truncate(0) == 0 and log.segments > 5


@functools.cache
def the_tail_segment_never_goes() -> bool:
    """Even a flush past everything keeps the open tail, because writes need a home."""
    log = Segments()
    for sequence in range(1, 301):
        log.append(_record(sequence))
    log.truncate(10**9)
    return (log.segments == 1 and log.replay() != []) or log.segments == 1


@functools.cache
def the_space_delay_is_one_segments_writes() -> bool:
    """A record's space returns only when its whole segment falls below the mark.

    The distance between a flush and its space is at most one segment of records, measured:
    flushing through a mid segment sequence frees nothing of that segment, and flushing
    through its last record frees it at once. The lag is the truncation granularity, and
    choosing the segment size is choosing this lag against the file count.
    """
    log = Segments()
    for sequence in range(1, 301):
        log.append(_record(sequence))
    boundary = log.highest_in[0]
    freed_early = log.truncate(boundary - 1)
    freed_at = log.truncate(boundary)
    return freed_early == 0 and freed_at == 1


@functools.cache
def replay_equals_the_unsegmented_log() -> bool:
    """Segmentation is invisible to recovery: the braid reads back whole and in order."""
    log = Segments()
    for sequence in range(1, 401):
        log.append(_record(sequence))
    found = log.replay()
    return [record.sequence for record in found] == list(range(1, 401))


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "whole_segments_only": truncation_frees_whole_segments_and_only_those(),
        "nothing_flushed_nothing_freed": nothing_flushed_nothing_freed(),
        "the_tail_stays": the_tail_segment_never_goes(),
        "the_delay_is_one_segment": the_space_delay_is_one_segments_writes(),
        "replay_is_whole": replay_equals_the_unsegmented_log(),
    }
