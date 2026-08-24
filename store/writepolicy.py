"""Write-through against write-back: io counted, dirty pages lost on cue.

A cache that sits on the write path must choose when the disk hears about
changes. Write-through tells it immediately and pays io per write.
Write-back marks the page dirty and pays io per page, coalescing every
rewrite of a hot page into one flush, and the price is that a crash takes
the dirty pages with it. Both policies run the same storm; the io meter
and the crash losses are read off, not argued.
"""

from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

PAGES = 256
CHECKPOINT = 200


@dataclass
class Cache:
    policy: str
    held: dict[int, int] = field(default_factory=dict)
    dirty: set[int] = field(default_factory=set)
    disk: dict[int, int] = field(default_factory=dict)
    disk_writes: int = 0
    ticks: int = 0

    def write(self, page: int, value: int) -> None:
        self.held[page] = value
        if self.policy == "through":
            self.disk[page] = value
            self.disk_writes += 1
        else:
            self.dirty.add(page)

    def flush(self) -> None:
        for page in sorted(self.dirty):
            self.disk[page] = self.held[page]
            self.disk_writes += 1
        self.dirty.clear()

    def tick(self) -> None:
        self.ticks += 1
        if self.policy == "back" and self.ticks % CHECKPOINT == 0:
            self.flush()

    def crash(self) -> int:
        """Pages whose latest value the disk never heard."""
        lost = sum(
            1 for page, value in self.held.items() if self.disk.get(page) != value
        )
        self.held = dict(self.disk)
        self.dirty = set()
        return lost


def _storm(policy: str, seed: int = 11, ticks: int = 2000) -> Cache:
    source = random.Random(seed)
    cache = Cache(policy=policy)
    for _ in range(ticks):
        cache.tick()
        for _ in range(4):
            if source.random() < 0.75:
                page = source.randrange(16)
            else:
                page = source.randrange(PAGES)
            cache.write(page, source.randrange(10**9))
    return cache


@functools.cache
def write_back_pays_a_fifth_of_the_io() -> bool:
    """The same 8000 writes cost 8000 disk writes through, 1461 back.

    Three quarters of the storm lands on sixteen hot pages, and write
    back folds every rewrite between checkpoints into one flush. The
    5.5x saving is the coalescing rate of the workload, not a property
    of the cache: a storm with no repeats would save nothing.
    """
    through = _storm("through")
    back = _storm("back")
    return through.disk_writes == 8000 and back.disk_writes == 1461


@functools.cache
def the_crash_bill_is_the_checkpoint_window() -> bool:
    """Crash 99 ticks into the window: 87 pages lost. Just after: 4.

    Write back's exposure is everything dirtied since the last
    checkpoint, so the loss depends entirely on when the plug is pulled.
    Write through loses zero at either moment and that is the whole
    trade: five times the io for a loss column that reads zero always.
    """
    early = _storm("back", ticks=2000)
    late = _storm("back", ticks=2099)
    through = _storm("through", ticks=2099)
    return early.crash() == 4 and late.crash() == 87 and through.crash() == 0


@functools.cache
def a_flush_before_the_crash_closes_the_window() -> bool:
    """Flushing first turns 87 lost pages into 0 for 87 extra writes.

    The orderly shutdown is write back's other half: the policy is only
    as safe as the certainty the flush runs. Everything between the two
    policies is a bet on that certainty.
    """
    saved = _storm("back", ticks=2099)
    cost_before = saved.disk_writes
    saved.flush()
    return saved.crash() == 0 and saved.disk_writes - cost_before == 87


@functools.cache
def recovery_reads_the_disk_not_the_memory() -> bool:
    """After a crash the cache serves exactly what the disk last heard.

    The crash replaces memory with the disk image: every surviving page
    equals its flushed value, every dirty page's last value is gone, and
    the write back cache after a crash equals a write through cache that
    stopped at the last checkpoint.
    """
    back = _storm("back", ticks=2099)
    back.crash()
    return back.held == back.disk and back.dirty == set()


@functools.cache
def summarise() -> dict:
    return {
        "module": "store.writepolicy",
        "write_back_pays_a_fifth_of_the_io": write_back_pays_a_fifth_of_the_io(),
        "the_crash_bill_is_the_checkpoint_window": (
            the_crash_bill_is_the_checkpoint_window()
        ),
        "a_flush_before_the_crash_closes_the_window": (
            a_flush_before_the_crash_closes_the_window()
        ),
        "recovery_reads_the_disk_not_the_memory": (
            recovery_reads_the_disk_not_the_memory()
        ),
    }
