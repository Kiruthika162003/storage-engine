from __future__ import annotations

import random
from dataclasses import dataclass, field

from store.engine import Store, crash
from store.verify.invariants import check as check_invariants

# Crashes injected at random moments, with the acknowledgement rule as the oracle.
#
# The durability contract is narrow and precise: a write the store acknowledged after a sync
# must survive, and a write that never reached a sync may vanish. Everything else, the flush
# timing, the compaction state, the number of files, is allowed to differ after a crash. The
# fuzzer drives a store with random writes, crashes it at a random moment, and checks exactly
# that contract, no more and no less.
#
# Checking less misses lost writes. Checking more, say asserting the file count, fails on
# behaviour the contract permits and trains people to ignore the fuzzer.


@dataclass
class Run:
    """One crash scenario and what the survivor still knew."""

    writes: int
    crashed_at: int
    acknowledged: int
    survived: int
    lost_acknowledged: list[bytes] = field(default_factory=list)
    invariant_breaks: int = field(default=0)

    def __bool__(self) -> bool:
        """A run is clean if nothing acknowledged was lost and the structure held."""
        return not self.lost_acknowledged and not self.invariant_breaks

    def as_dict(self) -> dict:
        """Flat mapping for reports."""
        return {
            "writes": self.writes,
            "crashed_at": self.crashed_at,
            "acknowledged": self.acknowledged,
            "survived": self.survived,
            "lost": [repr(one) for one in self.lost_acknowledged],
            "invariant_breaks": self.invariant_breaks,
            "clean": bool(self),
        }


def run(writes: int = 1500, keys: int = 300, seed: int = 0) -> Run:
    """One store, one random crash point, the contract checked on the survivor.

    The oracle tracks what the store has acknowledged: every completed put or delete, since
    the engine syncs per batch and a batch here is one record. At the crash, everything
    acknowledged must be readable in the survivor at its acknowledged value.
    """
    source = random.Random(seed)
    store = Store(flush_at=200, fold_at=3)
    acknowledged: dict[bytes, bytes | None] = {}
    crash_at = source.randrange(writes // 2, writes)
    made = 0
    for at in range(writes):
        key = f"k{source.randrange(keys):04d}".encode()
        if source.random() < 0.12:
            store.delete(key)
            acknowledged[key] = None
        else:
            value = source.randbytes(8)
            store.put(key, value)
            acknowledged[key] = value
        made += 1
        if at == crash_at:
            break
    survivor = crash(store)
    lost = [
        key
        for key, value in acknowledged.items()
        if survivor.get(key) != value
    ]
    breaks = len(check_invariants(survivor))
    survived = sum(
        1 for key, value in acknowledged.items() if survivor.get(key) == value
    )
    return Run(
        writes=made,
        crashed_at=crash_at,
        acknowledged=len(acknowledged),
        survived=survived,
        lost_acknowledged=lost,
        invariant_breaks=breaks,
    )


def sweep(runs: int = 30, writes: int = 1200) -> dict:
    """Many crash points, summarised for a test to assert on."""
    outcomes = [run(writes=writes, seed=seed) for seed in range(runs)]
    failed = [one for one in outcomes if not one]
    return {
        "runs": runs,
        "clean": len(outcomes) - len(failed),
        "failed": len(failed),
        "first_failure": failed[0].as_dict() if failed else None,
    }


def double_crash(writes: int = 1000, seed: int = 3) -> bool:
    """Crash, recover, write more, crash again, and the contract still holds.

    Recovery code is the least exercised code in any store, and recovery code that runs on the
    output of recovery code is the least exercised of all. The second crash is where a survivor
    that looked fine turns out to have recovered into a state that cannot recover.
    """
    source = random.Random(seed)
    store = Store(flush_at=150, fold_at=3)
    acknowledged: dict[bytes, bytes | None] = {}
    for _ in range(writes):
        key = f"k{source.randrange(200):04d}".encode()
        value = source.randbytes(8)
        store.put(key, value)
        acknowledged[key] = value
    survivor = crash(store)
    for _ in range(writes // 2):
        key = f"k{source.randrange(200):04d}".encode()
        value = source.randbytes(8)
        survivor.put(key, value)
        acknowledged[key] = value
    second = crash(survivor)
    return all(second.get(key) == value for key, value in acknowledged.items())
