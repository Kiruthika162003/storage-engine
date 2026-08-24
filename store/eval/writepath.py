"""One write path, swept over sync policy and memtable size, costs counted.

Two knobs dominate a write path: how often the log reaches the platter and
how large the memtable grows before flushing. The first trades charges for
the durability window, the number of acknowledged writes a crash may lose.
The second trades flush frequency against recovery replay length. Both
trades are swept and counted; the durability window is measured by crashing
at every position, not by argument.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

SYNC = 30
APPEND = 1
FLUSH = 100
WRITES = 2000


@dataclass
class WritePath:
    """Counts charges and tracks which acknowledged writes a crash loses."""

    sync_every: int
    flush_at: int
    charges: int = 0
    synced: int = 0
    appended: int = 0
    flushed_through: int = 0
    flushes: int = 0
    acknowledged: int = 0

    def put(self) -> None:
        self.appended += 1
        self.charges += APPEND
        if self.appended - self.synced >= self.sync_every:
            self.charges += SYNC
            self.synced = self.appended
        self.acknowledged += 1
        if self.appended - self.flushed_through >= self.flush_at:
            self.flush()

    def flush(self) -> None:
        if self.appended == self.flushed_through:
            return
        self.charges += FLUSH
        self.charges += SYNC
        self.synced = self.appended
        self.flushed_through = self.appended
        self.flushes += 1

    def lost_in_a_crash(self) -> int:
        return self.acknowledged - self.synced

    def replay_length(self) -> int:
        return self.synced - self.flushed_through


def run(sync_every: int, flush_at: int) -> WritePath:
    path = WritePath(sync_every=sync_every, flush_at=flush_at)
    for _ in range(WRITES):
        path.put()
    return path


@functools.cache
def sweep() -> tuple:
    rows = []
    for sync_every in (1, 8, 64):
        for flush_at in (128, 1024):
            path = run(sync_every, flush_at)
            rows.append(
                {
                    "sync_every": sync_every,
                    "flush_at": flush_at,
                    "charges": path.charges,
                    "at_risk": path.lost_in_a_crash(),
                    "replay": path.replay_length(),
                    "flushes": path.flushes,
                }
            )
    return tuple(rows)


def _row(sync_every: int, flush_at: int) -> dict:
    for row in sweep():
        if row["sync_every"] == sync_every and row["flush_at"] == flush_at:
            return row
    raise KeyError(sync_every)


def worst_case_lost(sync_every: int, flush_at: int) -> int:
    path = WritePath(sync_every=sync_every, flush_at=flush_at)
    worst = 0
    for _ in range(WRITES):
        path.put()
        worst = max(worst, path.lost_in_a_crash())
    return worst


@functools.cache
def every_write_synced_costs_thirteen_times() -> bool:
    """Sync every write: 63950 charges. Sync every 64th: 4880. 13.1x.

    The log append is one charge; the sync is thirty. Acknowledging each
    write only after its own sync spends the sync price two thousand times
    where the grouped log spends it thirty one times.
    """
    return _row(1, 128)["charges"] / _row(64, 128)["charges"] > 12


@functools.cache
def the_first_grouping_buys_almost_everything() -> bool:
    """Grouping 1 to 8 saves 52500 charges; 8 to 64 saves only 6570 more.

    Sync cost falls as 1/group, so the first factor of eight captures 89
    percent of everything grouping can ever save. Chasing the last factor
    of eight widens the durability window eightfold for a 13 percent gain.
    """
    first = _row(1, 128)["charges"] - _row(8, 128)["charges"]
    rest = _row(8, 128)["charges"] - _row(64, 128)["charges"]
    return first > rest * 7


@functools.cache
def the_window_is_the_group_minus_one() -> bool:
    """Crash at every position: worst loss is exactly sync_every - 1.

    The end-of-run number lied at first: 2000 is divisible by 8, so the
    final write landed synced and the naive at_risk column read zero. The
    worst case over all crash positions is 0, 7 and 63 acknowledged writes
    for groups of 1, 8 and 64, exactly the group minus one.
    """
    return all(
        worst_case_lost(group, 128) == group - 1 for group in (1, 8, 64)
    )


@functools.cache
def a_bigger_memtable_trades_replay_for_flushes() -> bool:
    """flush_at 128: 15 flushes, replay 80. flush_at 1024: 1 flush, replay 976.

    Fourteen saved flushes cost a recovery that replays twelve times as
    much log. The knob moves work between the write path and the restart.
    """
    small = _row(8, 128)
    large = _row(8, 1024)
    return (
        small["flushes"] - large["flushes"] == 14
        and large["replay"] > small["replay"] * 12
    )


@functools.cache
def the_knobs_do_not_touch() -> bool:
    """Flush count ignores sync policy; the at-risk window ignores flush size.

    Every sync_every column shows 15 flushes at flush_at 128 and 1 flush at
    1024; every flush_at row shows the same worst-case loss per group. The
    two trades can be tuned independently.
    """
    same_flushes = all(_row(group, 128)["flushes"] == 15 for group in (1, 8, 64))
    same_window = all(
        worst_case_lost(64, flush_at) == 63 for flush_at in (128, 1024)
    )
    return same_flushes and same_window


def render() -> str:
    lines = ["sync_every  flush_at  charges  worst_lost  replay  flushes"]
    for row in sweep():
        worst = worst_case_lost(row["sync_every"], row["flush_at"])
        lines.append(
            f"{row['sync_every']:<11} {row['flush_at']:<9} {row['charges']:<8} "
            f"{worst:<11} {row['replay']:<7} {row['flushes']}"
        )
    return "\n".join(lines)


@functools.cache
def summarise() -> dict:
    return {
        "module": "store.eval.writepath",
        "every_write_synced_costs_thirteen_times": (
            every_write_synced_costs_thirteen_times()
        ),
        "the_first_grouping_buys_almost_everything": (
            the_first_grouping_buys_almost_everything()
        ),
        "the_window_is_the_group_minus_one": the_window_is_the_group_minus_one(),
        "a_bigger_memtable_trades_replay_for_flushes": (
            a_bigger_memtable_trades_replay_for_flushes()
        ),
        "the_knobs_do_not_touch": the_knobs_do_not_touch(),
    }
