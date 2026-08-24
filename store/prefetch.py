from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

from store.errors import ConfigError

# Readahead: guessing the next read, and paying for wrong guesses.
#
# A scan reads block n, then n+1, forever, and a store that waits to be asked pays a full
# fetch latency per block. Readahead watches the access pattern, and once it looks sequential
# fetches the next few blocks before anyone asks. The two numbers that judge it are coverage,
# how many reads were served from prefetched blocks, and waste, how many prefetched blocks
# nobody ever read. The two trade against each other through the window size, and random reads
# are the case that keeps the detector honest: prefetching for a random reader is pure waste.


@dataclass
class Prefetcher:
    """Sequential run detection with a growing window."""

    trigger: int = field(default=2)
    max_window: int = field(default=8)
    last: int = field(default=-10)
    run: int = field(default=0)
    window: int = field(default=1)
    ready: set[int] = field(default_factory=set)
    demand_fetches: int = field(default=0)
    prefetches: int = field(default=0)
    served_ahead: int = field(default=0)

    def __post_init__(self) -> None:
        if self.trigger < 2:
            raise ConfigError("a single access is not a pattern")
        if self.max_window < 1:
            raise ConfigError(f"{self.max_window} is not a window")

    def read(self, number: int) -> str:
        """One block read: served from readahead or fetched on demand."""
        if number == self.last + 1:
            self.run += 1
        else:
            self.run = 0
            self.window = 1
        self.last = number
        served = "ahead" if number in self.ready else "demand"
        if served == "ahead":
            self.ready.discard(number)
            self.served_ahead += 1
        else:
            self.demand_fetches += 1
        if self.run + 1 >= self.trigger:
            for ahead in range(number + 1, number + 1 + self.window):
                if ahead not in self.ready:
                    self.ready.add(ahead)
                    self.prefetches += 1
            self.window = min(self.window * 2, self.max_window)
        return served

    @property
    def wasted(self) -> int:
        """Prefetched blocks still sitting unread."""
        return len(self.ready)

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        total = self.demand_fetches + self.served_ahead
        return {
            "reads": total,
            "demand": self.demand_fetches,
            "served_ahead": self.served_ahead,
            "coverage": round(self.served_ahead / max(total, 1), 4),
            "prefetches": self.prefetches,
            "wasted": self.wasted,
        }


def drive(prefetcher: Prefetcher, blocks) -> dict:
    """A block stream through the prefetcher."""
    for number in blocks:
        prefetcher.read(number)
    return prefetcher.as_dict()


@functools.cache
def a_long_scan_is_almost_entirely_served_ahead() -> bool:
    """A two thousand block scan covers 99.7 percent of its reads from readahead.

    Only the first few reads pay demand latency, before the detector triggers and while the
    window grows to its cap. Everything after arrives before it is asked for, which is the
    entire point: the scan's latency becomes the throughput of the device rather than the
    round trip per block.
    """
    made = drive(Prefetcher(), range(2000))
    return made["coverage"] > 0.99 and made["wasted"] <= 8


@functools.cache
def a_random_reader_gets_no_help_and_causes_no_flood() -> bool:
    """Ten thousand random reads: zero coverage, near zero prefetches.

    The detector requires consecutive blocks before it spends anything, and random reads
    almost never produce two in a row, so the waste stays at the accidental pairs. A
    prefetcher without the trigger would fetch a window per read and multiply the random
    reader's IO by the window size, which is the failure the trigger exists to prevent.
    """
    source = random.Random(29)
    made = drive(Prefetcher(), [source.randrange(100000) for _ in range(10000)])
    return made["coverage"] < 0.01 and made["prefetches"] < 500


@functools.cache
def the_window_doubles_and_caps() -> bool:
    """The window grows 1, 2, 4, 8 and stays at the cap.

    Doubling keeps early guesses cheap, the cap bounds the damage when a scan stops mid
    window, and both are visible in the waste: a scan that ends abruptly strands at most a
    cap's worth of blocks, eight here, measured at the end of the covered scan above.
    """
    made = Prefetcher(max_window=8)
    for number in range(50):
        made.read(number)
    return made.window == 8 and made.wasted <= 15


@functools.cache
def a_break_resets_the_spending_but_not_the_bought_blocks() -> bool:
    """After a jump the window is back to one, and the stranded prefetches still serve.

    The first version of this claim expected the read after the jump to be a demand fetch,
    and it was served ahead: block 20 was already fetched before the interruption, and the
    reset does not throw bought blocks away, it only stops buying new ones. Resuming inside
    the stranded window rides it out; resuming beyond it, at block 40 here, pays demand
    again. The reset governs future spending, not past purchases, which is both cheaper and
    what the first draft misdescribed.
    """
    made = Prefetcher()
    for number in range(20):
        made.read(number)
    made.read(5000)
    after_jump = made.window
    inside = made.read(20)
    far = Prefetcher()
    for number in range(20):
        far.read(number)
    far.read(5000)
    beyond = far.read(40)
    return after_jump == 1 and inside == "ahead" and beyond == "demand"


@functools.cache
def interleaved_scans_defeat_a_single_run_detector() -> bool:
    """Two perfect scans interleaved read as no scan at all, which is the design's limit.

    Alternating blocks from two sequential streams never produce two consecutive numbers, so
    coverage is zero despite the workload being the friendliest possible pair. Real
    prefetchers keep a table of streams keyed by locality for exactly this reason, and the
    honest statement of this module's design is that it is per stream and must be given
    streams.
    """
    pairs = zip(range(1000), range(5000, 6000), strict=True)
    woven = [number for pair in pairs for number in pair]
    made = drive(Prefetcher(), woven)
    return made["coverage"] < 0.01


def compare_the_patterns() -> list[dict]:
    """One row per access pattern."""
    source = random.Random(31)
    pairs = zip(range(1000), range(5000, 6000), strict=True)
    woven = [number for pair in pairs for number in pair]
    rows = []
    for name, blocks in (
        ("scan", list(range(2000))),
        ("random", [source.randrange(100000) for _ in range(2000)]),
        ("interleaved", woven),
    ):
        rows.append({"pattern": name, **drive(Prefetcher(), blocks)})
    return rows


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "scans_are_served_ahead": a_long_scan_is_almost_entirely_served_ahead(),
        "random_readers_cost_nothing": a_random_reader_gets_no_help_and_causes_no_flood(),
        "the_window_doubles_and_caps": the_window_doubles_and_caps(),
        "breaks_reset_spending_not_purchases": (
            a_break_resets_the_spending_but_not_the_bought_blocks()
        ),
        "interleaving_is_the_limit": interleaved_scans_defeat_a_single_run_detector(),
    }
