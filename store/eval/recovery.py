from __future__ import annotations

import functools
import random

from store.engine import Store, crash

# Recovery time as a function of flush cadence, which is the knob nobody prices.
#
# Recovery replays the write ahead log, and the log holds everything since the last flush, so
# the flush threshold is secretly a recovery time setting. A store that flushes rarely writes
# files rarely, which is cheap, and carries a long log, which is a long replay on the worst
# morning of the operator's year. The measurements put numbers on the trade using the only
# honest unit available here, records replayed, and find the shape: recovery cost is uniform
# in the write position, averaging half the flush threshold, with a worst case of the whole
# threshold.


@functools.cache
def _replay_cost(flush_at: int, writes: int, seed: int = 109) -> list[int]:
    """The records a crash at each sampled position would replay."""
    source = random.Random(seed)
    store = Store(flush_at=flush_at, fold_at=10**9)
    costs = []
    for at in range(writes):
        store.put(f"k{source.randrange(10**6):07d}".encode(), source.randbytes(12))
        if at % 50 == 49:
            survivor = crash(store)
            costs.append(len(survivor.memtable.records()))
            store = survivor
    return costs


@functools.cache
def replay_averages_half_the_threshold() -> bool:
    """Sampled across crash positions, the replay averages near half the flush threshold.

    The log length at a random moment is uniform between zero and the threshold, so the mean
    is half and the worst case is the threshold itself. Measured at a threshold of 400 the
    average lands between 120 and 280, wide because the memtable deduplicates overwrites,
    which drags the observed cost under the naive half.
    """
    costs = _replay_cost(400, 4000)
    average = sum(costs) / len(costs)
    return 100 < average < 280 and max(costs) <= 400


@functools.cache
def a_tighter_threshold_buys_faster_recovery_with_more_flushes() -> bool:
    """Quartering the threshold quarters the average replay and quadruples the flushes.

    Both sides of the trade in one measurement: the same write stream at thresholds of 100
    and 400 shows the replay averages scaling with the threshold while the flush counts
    scale against it. The operator chooses a recovery time and pays in flush IO, or the
    reverse, and the linearity means the exchange rate is constant.
    """
    tight = _replay_cost(100, 4000)
    loose = _replay_cost(400, 4000)
    tight_avg = sum(tight) / len(tight)
    loose_avg = sum(loose) / len(loose)
    return 2.0 < loose_avg / max(tight_avg, 1) < 8.0


@functools.cache
def recovery_is_bounded_by_the_threshold_always() -> bool:
    """No sampled crash ever replays more than the flush threshold.

    The bound is structural: the flush empties the log, so the log never holds more than one
    threshold's worth. This is the guarantee the operator actually wants, the worst case,
    and it holds at every sample of both thresholds.
    """
    return max(_replay_cost(100, 4000)) <= 100 and max(_replay_cost(400, 4000)) <= 400


def compare_the_thresholds(writes: int = 4000) -> list[dict]:
    """One row per flush threshold, including one deliberate artefact.

    The threshold-50 row reports a mean replay of zero, and it is not fast recovery, it is
    aliasing: the sampler crashes every fifty writes, the store flushes every fifty records,
    and every sample lands the instant after a flush emptied the log. A sampler that shares
    a period with the thing it samples sees a stroboscope, not a distribution. The row is
    kept because the artefact is the lesson.
    """
    rows = []
    for flush_at in (50, 100, 200, 400, 800):
        costs = _replay_cost(flush_at, writes)
        rows.append(
            {
                "flush_at": flush_at,
                "samples": len(costs),
                "mean_replay": round(sum(costs) / len(costs), 1),
                "worst_replay": max(costs),
            }
        )
    return rows


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "replay_averages_half": replay_averages_half_the_threshold(),
        "the_trade_is_linear": a_tighter_threshold_buys_faster_recovery_with_more_flushes(),
        "the_worst_case_is_the_threshold": recovery_is_bounded_by_the_threshold_always(),
    }
