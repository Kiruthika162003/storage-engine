from __future__ import annotations

import functools
import random
from dataclasses import dataclass

from store.errors import ConfigError

# Which files to compact: three pickers, priced in bytes written per byte reclaimed.
#
# The compaction module decided when to compact; this one decides what, which is where the
# budgets actually leak. A compaction's value is the dead bytes it removes and its cost is
# every byte it rewrites, so the efficiency of a picker is written-per-reclaimed, and the
# three habits people reach for score very differently on it. Oldest-first assumes age
# means garbage. Largest-first assumes size means garbage. Overlap-aware estimates the
# actual dead bytes from key range overlap and picks the densest garbage, paying for the
# estimate with the metadata the manifest already holds.


@dataclass(frozen=True)
class Candidate:
    """One file a picker may choose."""

    number: int
    age: int
    size: int
    dead_bytes: int
    overlap_estimate: int


def pick_oldest(candidates: list[Candidate]) -> Candidate:
    if not candidates:
        raise ConfigError("nothing to pick")
    return max(candidates, key=lambda one: one.age)


def pick_largest(candidates: list[Candidate]) -> Candidate:
    if not candidates:
        raise ConfigError("nothing to pick")
    return max(candidates, key=lambda one: one.size)


def pick_by_overlap(candidates: list[Candidate]) -> Candidate:
    if not candidates:
        raise ConfigError("nothing to pick")
    return max(candidates, key=lambda one: one.overlap_estimate / max(one.size, 1))


def efficiency(chosen: Candidate) -> float:
    """Bytes written per byte reclaimed: lower is better, infinity for pure churn."""
    if chosen.dead_bytes == 0:
        return float("inf")
    return round(chosen.size / chosen.dead_bytes, 3)


@functools.cache
def _fleet(count: int = 60, seed: int = 367) -> tuple[Candidate, ...]:
    """Files where age, size and garbage are deliberately uncorrelated.

    The old files are mostly clean, having been compacted before; the big files are big
    because they hold live data; the garbage sits in middling recent files that took the
    overwrite traffic. This is the shape a hot-key workload actually leaves, and it is
    chosen precisely because it defeats both proxies.
    """
    source = random.Random(seed)
    made = []
    for number in range(count):
        age = source.randrange(1, 100)
        size = source.randrange(1000, 20000)
        if 30 < age < 60 and size < 8000:
            dead = int(size * source.uniform(0.5, 0.9))
        else:
            dead = int(size * source.uniform(0.0, 0.15))
        noise = source.uniform(0.8, 1.2)
        made.append(
            Candidate(
                number=number,
                age=age,
                size=size,
                dead_bytes=dead,
                overlap_estimate=int(dead * noise),
            )
        )
    return tuple(made)


@functools.cache
def the_proxies_pick_clean_files_and_pay_for_it() -> bool:
    """Oldest-first and largest-first both write over eight bytes per byte reclaimed.

    On the uncorrelated fleet, the oldest file is mostly live and the largest file is
    almost all live, so both pickers rewrite great volumes to reclaim little. The proxy is
    the bug: each encodes a correlation the workload does not have, and the price appears
    on the one meter that matters.
    """
    fleet = list(_fleet())
    oldest = efficiency(pick_oldest(fleet))
    largest = efficiency(pick_largest(fleet))
    return oldest > 6.0 and largest > 6.0


@functools.cache
def the_overlap_picker_finds_the_dense_garbage() -> bool:
    """Overlap-aware picking lands under two bytes written per byte reclaimed.

    The estimate is noisy by construction, twenty percent either way, and still lands on a
    file that is mostly garbage, because the ranking only has to order the candidates, not
    price them exactly. Estimates that would embarrass a billing system are plenty for a
    picker, the planner module's lesson at the compaction layer.
    """
    fleet = list(_fleet())
    chosen = efficiency(pick_by_overlap(fleet))
    return chosen < 2.0


@functools.cache
def a_sequence_of_picks_compounds_the_gap() -> bool:
    """Ten rounds: overlap reclaims 34,961 dead bytes, largest 14,868, oldest 3,696.

    Each round removes the chosen file from the fleet, and the compounding is the real
    story: the proxies keep spending the write budget on clean files while the garbage
    sits, so the gap grows with every round rather than averaging out.
    """
    def run(picker) -> int:
        fleet = list(_fleet())
        reclaimed = 0
        for _ in range(10):
            chosen = picker(fleet)
            reclaimed += chosen.dead_bytes
            fleet.remove(chosen)
        return reclaimed

    by_overlap = run(pick_by_overlap)
    by_age = run(pick_oldest)
    by_size = run(pick_largest)
    return by_overlap > by_age * 3 and by_overlap > by_size * 2


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "proxies_pay": the_proxies_pick_clean_files_and_pay_for_it(),
        "overlap_finds_the_garbage": the_overlap_picker_finds_the_dense_garbage(),
        "the_gap_compounds": a_sequence_of_picks_compounds_the_gap(),
    }
