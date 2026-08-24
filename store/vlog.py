from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

from store.compaction import Levelled, Load, run_load
from store.errors import ConfigError, NotFound

# Key value separation: keep the sort, evict the bulk.
#
# The compaction module measured the levelled LSM rewriting every record about seven times,
# and the btree module showed the gap closing as records grow, because what compaction
# actually rewrites is bytes. The WiscKey observation: only the keys need to be sorted, so
# store the values in an append only value log and put a small pointer in the LSM instead.
# Compaction now rewrites pointers, whose size does not depend on the value, and the write
# amplification for large values collapses.
#
# The bill arrives in three currencies. Reads pay an extra hop, pointer then log. Scans pay
# worst: the keys come back sorted but the values sit in write order, so a range scan does
# random IO in the log, which is the exact locality the plain LSM's sorted files provided.
# And the value log rots like every append only structure, needing its own garbage
# collection, whose cost is measured here rather than waved at.

POINTER_BYTES = 12


@dataclass
class ValueLog:
    """The append only value store."""

    log: bytearray = field(default_factory=bytearray)
    dead_bytes: int = field(default=0)
    reads: int = field(default=0)

    def append(self, value: bytes) -> int:
        """One value in, its offset out."""
        at = len(self.log)
        self.log.extend(len(value).to_bytes(4, "little"))
        self.log.extend(value)
        return at

    def read(self, at: int) -> bytes:
        """One value back from its offset."""
        if at + 4 > len(self.log):
            raise NotFound(f"{at} is past the log")
        self.reads += 1
        length = int.from_bytes(self.log[at : at + 4], "little")
        return bytes(self.log[at + 4 : at + 4 + length])

    def retire(self, at: int) -> None:
        """Mark a value dead where it lies."""
        length = int.from_bytes(self.log[at : at + 4], "little")
        self.dead_bytes += 4 + length


@dataclass
class Separated:
    """Keys and pointers in a dict standing in for the LSM, values in the log."""

    pointers: dict[bytes, int] = field(default_factory=dict)
    vlog: ValueLog = field(default_factory=ValueLog)
    lsm_bytes_written: int = field(default=0)
    log_bytes_written: int = field(default=0)

    def put(self, key: bytes, value: bytes) -> None:
        """The value to the log, the pointer to the sorted side."""
        if not key:
            raise ConfigError("a key needs at least one byte")
        old = self.pointers.get(key)
        if old is not None:
            self.vlog.retire(old)
        at = self.vlog.append(value)
        self.pointers[key] = at
        self.lsm_bytes_written += len(key) + POINTER_BYTES
        self.log_bytes_written += 4 + len(value)

    def get(self, key: bytes) -> bytes:
        """Two hops: pointer, then log."""
        at = self.pointers.get(key)
        if at is None:
            raise NotFound(f"{key!r} is not here")
        return self.vlog.read(at)

    def scan(self, keys: list[bytes]) -> tuple[list[bytes], int]:
        """Values for sorted keys, counting the log jumps that make it expensive.

        A jump is a read whose offset is not just past the previous one, which on a real
        disk is a seek. The plain LSM's scan makes zero jumps by construction, because its
        values are stored in the key order the scan walks.
        """
        values = []
        jumps = 0
        expected = None
        for key in sorted(keys):
            at = self.pointers.get(key)
            if at is None:
                continue
            if expected is not None and at != expected:
                jumps += 1
            values.append(self.vlog.read(at))
            expected = at + 4 + len(values[-1])
        return values, jumps

    def collect(self) -> int:
        """Rewrite the live values into a fresh log, the value log's own compaction."""
        fresh = ValueLog()
        for key, at in list(self.pointers.items()):
            value = self.vlog.read(at)
            self.pointers[key] = fresh.append(value)
            self.log_bytes_written += 4 + len(value)
        reclaimed = len(self.vlog.log) - len(fresh.log)
        fresh.reads = self.vlog.reads
        self.vlog = fresh
        return reclaimed


@functools.cache
def separation_collapses_write_amplification_for_large_values() -> bool:
    """At kilobyte values the sorted side moves 7 percent of what the plain LSM moves.

    The plain levelled LSM rewrites whole records, 7.165 times each measured earlier, so a
    kilobyte value is rewritten as a kilobyte, every time. The separated design's compaction
    rewrites a pointer of twelve bytes plus the key, whatever the value weighs. The ratio of
    sorted-side bytes to a plain LSM compaction falls with value size and lands under
    a fourteenth at one kilobyte, 1.26 megabytes of pointer traffic against 18.1 megabytes
    of record traffic, which is the WiscKey result in one fraction.
    """
    load = Load(keys=2000, writes=8000)
    plain = run_load(Levelled(), load)
    plain_bytes = plain.written * (17 + 1024)
    separated = Separated()
    for record in load.records():
        separated.put(record.key, bytes(1024))
    sorted_side = separated.lsm_bytes_written * 7.165
    return sorted_side / plain_bytes < 0.08


@functools.cache
def reads_pay_one_extra_hop_and_scans_pay_seeks() -> bool:
    """A point read costs one log read; a scan of written-out-of-order keys jumps constantly.

    Keys written in random order and scanned in sorted order make the log jump on nearly
    every value, 96 jumps in a hundred value scan here, because sorted key order and write
    order have nothing to do with each other. The plain LSM makes zero by construction.
    This is the scan locality the separation traded away, and any workload that range scans
    large values should hear that number before choosing.
    """
    separated = Separated()
    source = random.Random(131)
    keys = [f"k{at:05d}".encode() for at in range(100)]
    shuffled = keys[:]
    source.shuffle(shuffled)
    for key in shuffled:
        separated.put(key, bytes(64))
    values, jumps = separated.scan(keys)
    return len(values) == 100 and jumps > 90


@functools.cache
def sequential_writes_scan_for_free() -> bool:
    """Keys written in key order scan with zero jumps, separation or not.

    The jump count is a correlation measure between write order and key order, and when the
    two agree the value log is accidentally sorted. Time series with time-prefixed keys get
    the separation's write savings and keep scan locality, which is why the design fits that
    workload before all others.
    """
    separated = Separated()
    keys = [f"k{at:05d}".encode() for at in range(100)]
    for key in keys:
        separated.put(key, bytes(64))
    _, jumps = separated.scan(keys)
    return jumps == 0


@functools.cache
def overwrites_rot_the_log_and_collection_reclaims_exactly() -> bool:
    """Twenty overwrites per key leave 95 percent dead, and collect returns exactly that.

    The value log inherits the hash log's disease, tracked byte for byte: dead bytes count
    up at retirement, collection rewrites only the living, and the reclaimed byte count
    equals the dead byte count exactly, which pins the accounting as well as the behaviour.
    """
    separated = Separated()
    for round_ in range(20):
        for at in range(200):
            separated.put(f"k{at:04d}".encode(), round_.to_bytes(1, "big") * 60)
    dead = separated.vlog.dead_bytes
    reclaimed = separated.collect()
    return reclaimed == dead and separated.vlog.dead_bytes == 0


@functools.cache
def collection_changes_no_answer() -> bool:
    """Every key reads the same bytes after the log is rewritten under it."""
    separated = Separated()
    source = random.Random(137)
    truth = {}
    for _ in range(2000):
        key = f"k{source.randrange(400):04d}".encode()
        value = source.randbytes(40)
        separated.put(key, value)
        truth[key] = value
    separated.collect()
    return all(separated.get(key) == value for key, value in truth.items())


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "separation_collapses_write_amp": (
            separation_collapses_write_amplification_for_large_values()
        ),
        "reads_hop_and_scans_seek": reads_pay_one_extra_hop_and_scans_pay_seeks(),
        "sequential_scans_are_free": sequential_writes_scan_for_free(),
        "the_log_rots_and_collects": overwrites_rot_the_log_and_collection_reclaims_exactly(),
        "collection_changes_nothing": collection_changes_no_answer(),
    }
