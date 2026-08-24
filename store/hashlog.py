from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

from store.errors import ConfigError, NotFound

# The bitcask shape: a hash index over an append only log, which is not an LSM.
#
# Everything in this package so far keeps keys sorted, and paid for it in compaction. The
# hash log refuses the sort. Writes append to a log, an in memory table maps each key to the
# offset of its newest record, a point read is one seek at a known offset, and the design is
# complete. The three consequences are the module: point reads cost exactly one log read at
# any size, the whole keyset must fit in memory because the index cannot spill without
# becoming a different design, and range scans do not exist, because the log has no order to
# scan in. Compaction still happens, but for space alone, never for read shape.

HEADER = 16


@dataclass
class HashLog:
    """The log, the offsets, and the meters."""

    log: bytearray = field(default_factory=bytearray)
    offsets: dict[bytes, int] = field(default_factory=dict)
    log_reads: int = field(default=0)
    appended: int = field(default=0)
    dead_bytes: int = field(default=0)

    def put(self, key: bytes, value: bytes) -> int:
        """Append and point the index at the new record."""
        if not key:
            raise ConfigError("a key needs at least one byte")
        at = len(self.log)
        old = self.offsets.get(key)
        if old is not None:
            self.dead_bytes += self._record_size(old)
        self.log.extend(len(key).to_bytes(8, "little"))
        self.log.extend(len(value).to_bytes(8, "little"))
        self.log.extend(key)
        self.log.extend(value)
        self.offsets[key] = at
        self.appended += 1
        return at

    def _record_size(self, at: int) -> int:
        key_length = int.from_bytes(self.log[at : at + 8], "little")
        value_length = int.from_bytes(self.log[at + 8 : at + 16], "little")
        return HEADER + key_length + value_length

    def _read_at(self, at: int) -> tuple[bytes, bytes]:
        """One record from one offset, which is the whole read path."""
        self.log_reads += 1
        key_length = int.from_bytes(self.log[at : at + 8], "little")
        value_length = int.from_bytes(self.log[at + 8 : at + 16], "little")
        key_start = at + HEADER
        value_start = key_start + key_length
        return (
            bytes(self.log[key_start:value_start]),
            bytes(self.log[value_start : value_start + value_length]),
        )

    def get(self, key: bytes) -> bytes:
        """One index probe, one log read, no other outcome."""
        at = self.offsets.get(key)
        if at is None:
            raise NotFound(f"{key!r} is not here")
        _, value = self._read_at(at)
        return value

    def delete(self, key: bytes) -> None:
        """Forget the offset; the bytes become garbage where they lie."""
        old = self.offsets.pop(key, None)
        if old is None:
            raise NotFound(f"{key!r} is not here")
        self.dead_bytes += self._record_size(old)

    def compact(self) -> int:
        """Rewrite only the live records, for space and nothing else."""
        fresh = bytearray()
        offsets = {}
        for key, at in self.offsets.items():
            _, value = self._read_at(at)
            new_at = len(fresh)
            fresh.extend(len(key).to_bytes(8, "little"))
            fresh.extend(len(value).to_bytes(8, "little"))
            fresh.extend(key)
            fresh.extend(value)
            offsets[key] = new_at
        reclaimed = len(self.log) - len(fresh)
        self.log = fresh
        self.offsets = offsets
        self.dead_bytes = 0
        return reclaimed

    @property
    def keys(self) -> int:
        """Live keys."""
        return len(self.offsets)

    def index_bytes(self) -> int:
        """What the index costs: every live key in memory, plus an offset each."""
        return sum(len(key) + 8 for key in self.offsets)

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "keys": self.keys,
            "log_bytes": len(self.log),
            "dead_bytes": self.dead_bytes,
            "index_bytes": self.index_bytes(),
            "log_reads": self.log_reads,
        }


@functools.cache
def a_point_read_costs_one_log_read_at_any_size() -> bool:
    """A thousand reads cost a thousand log reads, in a store of any population.

    No filter, no level walk, no binary search: the index knows the offset and the read goes
    there. The LSM's read path exists because its index is sparse; this index is dense, and
    the density is paid for in the next claim.
    """
    small = HashLog()
    large = HashLog()
    for at in range(500):
        small.put(f"k{at:06d}".encode(), bytes(20))
    for at in range(20000):
        large.put(f"k{at:06d}".encode(), bytes(20))
    for at in range(0, 500, 5):
        small.get(f"k{at:06d}".encode())
    for at in range(0, 20000, 200):
        large.get(f"k{at:06d}".encode())
    return small.log_reads == 100 and large.log_reads == 100


@functools.cache
def the_index_holds_every_key_which_is_the_designs_wall() -> bool:
    """The index costs the keys themselves, in memory, always.

    Twenty thousand keys of eight bytes cost 320,000 index bytes, and the number scales with
    the keyset forever. The sstable's sparse index cost under one percent of the data; this
    one costs all of the keys, and that is the wall: a hash log whose keys outgrow memory is
    not slow, it is over.
    """
    made = HashLog()
    for at in range(20000):
        made.put(f"k{at:07d}".encode(), bytes(50))
    return made.index_bytes() == 20000 * (8 + 8)


@functools.cache
def overwrites_rot_the_log_until_compaction() -> bool:
    """A thousand keys overwritten twenty times leave 95 percent of the log dead.

    Every overwrite abandons its predecessor in place, dead bytes are tracked as they are
    made, and one compaction rewrites the five percent that is alive. The LSM told this
    story with levels; the hash log tells it with one flat file, and the moral does not
    change: append only designs owe their space back, on a schedule someone must choose.
    """
    made = HashLog()
    for round_ in range(20):
        for at in range(1000):
            made.put(f"k{at:05d}".encode(), round_.to_bytes(1, "big") * 40)
    before = len(made.log)
    reclaimed = made.compact()
    return made.dead_bytes == 0 and reclaimed > before * 0.9


@functools.cache
def compaction_changes_no_answer() -> bool:
    """Every live key reads the same bytes after the rewrite.

    The same bar every compaction in this package has to clear, cleared the same way, by
    reading everything on both sides.
    """
    made = HashLog()
    source = random.Random(97)
    truth = {}
    for _ in range(3000):
        key = f"k{source.randrange(600):04d}".encode()
        value = source.randbytes(12)
        made.put(key, value)
        truth[key] = value
    made.compact()
    return all(made.get(key) == value for key, value in truth.items())


@functools.cache
def there_is_no_scan_and_that_is_the_price_of_the_flat_read() -> bool:
    """The design has no ordered anything, stated as an interface fact.

    The offsets dictionary iterates in insertion order, the log is write order, and neither
    is key order. The LSM paid compaction to keep order; the hash log pockets that cost and
    loses range queries outright, not slowly. A design comparison that omits the missing
    operation flatters the design that dropped it.
    """
    made = HashLog()
    for key in (b"c", b"a", b"b"):
        made.put(key, b"v")
    stored = list(made.offsets)
    return stored == [b"c", b"a", b"b"] and stored != sorted(stored)


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "point_reads_are_flat": a_point_read_costs_one_log_read_at_any_size(),
        "the_index_is_the_wall": the_index_holds_every_key_which_is_the_designs_wall(),
        "overwrites_rot_the_log": overwrites_rot_the_log_until_compaction(),
        "compaction_changes_nothing": compaction_changes_no_answer(),
        "there_is_no_scan": there_is_no_scan_and_that_is_the_price_of_the_flat_read(),
    }
