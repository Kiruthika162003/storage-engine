"""A log-with-keydir design measured against the levelled engine's shape.

The alternative to sorting is remembering: append every record to a log
and hold an in-memory directory from key to file offset. Reads cost one
seek always, writes cost one append always, and the price is a directory
entry per live key plus a merge that rewrites the log to evict garbage.
The numbers here weigh that trade against the levelled design's.
"""

from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

ENTRY = 32
KEY = 16
VALUE = 100


@dataclass
class Keydir:
    """Key to (file, offset); memory charged at ENTRY bytes per live key."""

    offsets: dict[bytes, tuple[int, int]] = field(default_factory=dict)

    def memory(self) -> int:
        return len(self.offsets) * ENTRY


@dataclass
class Cask:
    keydir: Keydir = field(default_factory=Keydir)
    files: list[list[tuple[bytes, bytes | None]]] = field(default_factory=lambda: [[]])
    appended_bytes: int = 0
    seeks: int = 0

    def put(self, key: bytes, value: bytes) -> None:
        active = self.files[-1]
        self.keydir.offsets[key] = (len(self.files) - 1, len(active))
        active.append((key, value))
        self.appended_bytes += KEY + len(value)

    def delete(self, key: bytes) -> None:
        active = self.files[-1]
        active.append((key, None))
        self.appended_bytes += KEY
        self.keydir.offsets.pop(key, None)

    def get(self, key: bytes) -> bytes | None:
        place = self.keydir.offsets.get(key)
        if place is None:
            return None
        self.seeks += 1
        file_at, offset = place
        return self.files[file_at][offset][1]

    def roll(self) -> None:
        self.files.append([])

    def live_bytes(self) -> int:
        return sum(
            KEY + len(self.files[file_at][offset][1])
            for file_at, offset in self.keydir.offsets.values()
        )

    def stored_bytes(self) -> int:
        return sum(
            KEY + (len(value) if value is not None else 0)
            for chunk in self.files
            for _, value in chunk
        )

    def merge(self) -> int:
        """Rewrite every file into one holding only live records.

        Returns the bytes read plus written, the merge's cost.
        """
        cost = self.stored_bytes()
        survivors = []
        for key, (file_at, offset) in sorted(
            self.keydir.offsets.items(), key=lambda item: item[1]
        ):
            survivors.append((key, self.files[file_at][offset][1]))
        self.files = [[]]
        self.keydir = Keydir()
        for key, value in survivors:
            self.put(key, value)
        cost += self.live_bytes()
        self.appended_bytes = 0
        return cost


def _worked(seed: int, value_size: int = VALUE) -> Cask:
    source = random.Random(seed)
    cask = Cask()
    keys = [f"k{number:06d}".encode() for number in range(20000)]
    for step in range(100000):
        cask.put(source.choice(keys), source.randbytes(value_size))
        if step % 10000 == 9999:
            cask.roll()
    return cask


@functools.cache
def every_present_read_is_one_seek() -> bool:
    """4962 present keys read, 4962 seeks, five overwrites deep.

    The keydir points at the newest copy directly, so read cost is flat at
    one seek no matter how many stale copies the log holds. The levelled
    engine's read walks tables until one answers; this design's walk is a
    dictionary lookup.
    """
    cask = _worked(9)
    keys = [f"k{number:06d}".encode() for number in range(5000)]
    found = sum(1 for key in keys if cask.get(key) is not None)
    return cask.seeks == found == 4962


@functools.cache
def absent_keys_cost_no_seek_at_all() -> bool:
    """A miss is answered by the keydir in memory: zero disk seeks.

    The levelled engine buys this property with a bloom filter per table
    and still pays the probes. Here absence is exact and free because the
    directory is complete: 38 of the first 5000 keys were never drawn by
    the workload, and reading them touched the disk zero times.
    """
    cask = _worked(9)
    keys = [f"k{number:06d}".encode() for number in range(5000)]
    misses = sum(1 for key in keys if cask.get(key) is None)
    return misses == 38 and cask.seeks == 5000 - misses


@functools.cache
def the_log_stores_five_times_the_live_data() -> bool:
    """100000 puts over 20000 keys: 11.6 MB stored, 2.3 MB live, 5.03x.

    Write amplification is exactly one, every byte is written once, and
    the bill arrives as space instead: each key averages five copies and
    only the newest matters. Sorting pays during the write; logging pays
    with a warehouse of garbage.
    """
    cask = _worked(9)
    return 5.0 < cask.stored_bytes() / cask.live_bytes() < 5.1


@functools.cache
def the_merge_reads_the_garbage_to_evict_it() -> bool:
    """The merge costs 13.9 MB of io to shrink 11.6 MB stored to 2.3 MB.

    It reads every stored byte, garbage included, and writes the live set
    back out: cost equals stored plus live. Afterwards stored equals live
    exactly. This is compaction under another name, deferred and paid in
    one lump instead of levelled instalments.
    """
    cask = _worked(9)
    before = cask.stored_bytes()
    cost = cask.merge()
    return cost == before + cask.live_bytes() and cask.stored_bytes() == cask.live_bytes()


@functools.cache
def the_keydir_rent_scales_with_count_not_size() -> bool:
    """Keydir memory is 27.6 percent of live data at 100 byte values and
    3.1 percent at 1000 byte values, exactly 32 over key plus value.

    The directory charges 32 bytes per live key regardless of value size,
    so the design suits few large values and punishes many small ones.
    At 100 byte values more than a quarter of the dataset must fit in
    memory just to know where things are.
    """
    small = _worked(9, value_size=100)
    large = _worked(9, value_size=1000)
    small_rent = small.keydir.memory() / small.live_bytes()
    large_rent = large.keydir.memory() / large.live_bytes()
    return small_rent == ENTRY / (KEY + 100) and large_rent == ENTRY / (KEY + 1000)


@functools.cache
def summarise() -> dict:
    return {
        "module": "store.bitcask",
        "every_present_read_is_one_seek": every_present_read_is_one_seek(),
        "absent_keys_cost_no_seek_at_all": absent_keys_cost_no_seek_at_all(),
        "the_log_stores_five_times_the_live_data": (
            the_log_stores_five_times_the_live_data()
        ),
        "the_merge_reads_the_garbage_to_evict_it": (
            the_merge_reads_the_garbage_to_evict_it()
        ),
        "the_keydir_rent_scales_with_count_not_size": (
            the_keydir_rent_scales_with_count_not_size()
        ),
    }
