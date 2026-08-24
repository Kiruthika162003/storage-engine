from __future__ import annotations

import functools
import random
from dataclasses import dataclass

from store.errors import ConfigError

# Tiered media: which bytes deserve the fast disk, decided by measurement.
#
# Storage comes in tiers, fast and small or slow and vast, priced accordingly. A store that
# fits on the fast tier has no decision to make; every real store does not fit, and the
# decision is which files go where. The right answer is not the newest files, it is the read
# files, and the two only coincide when the workload is recency shaped, which is an assumption
# worth measuring rather than baking in.
#
# The model prices a read at one for the fast tier and ten for the slow, charges each file's
# reads at its tier, and compares placement policies over workloads whose heat does and does
# not follow age.

FAST_COST = 1
SLOW_COST = 10


@dataclass(frozen=True)
class File:
    """One file: its age rank, its size, and how often it is read."""

    number: int
    age: int
    size: int
    reads_per_day: int


@dataclass
class Placement:
    """A division of files between the tiers."""

    fast: set[int]
    files: dict[int, File]
    budget: int

    def __post_init__(self) -> None:
        spent = sum(self.files[number].size for number in self.fast)
        if spent > self.budget:
            raise ConfigError(f"{spent} bytes placed on a fast tier of {self.budget}")

    def daily_cost(self) -> int:
        """What a day of reads costs under this placement."""
        total = 0
        for file in self.files.values():
            rate = FAST_COST if file.number in self.fast else SLOW_COST
            total += file.reads_per_day * rate
        return total

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "fast_files": len(self.fast),
            "fast_bytes": sum(self.files[number].size for number in self.fast),
            "budget": self.budget,
            "daily_cost": self.daily_cost(),
        }


def by_age(files: dict[int, File], budget: int) -> Placement:
    """Newest first, which is what everyone builds because age is free to know."""
    chosen: set[int] = set()
    spent = 0
    for file in sorted(files.values(), key=lambda one: one.age):
        if spent + file.size <= budget:
            chosen.add(file.number)
            spent += file.size
    return Placement(fast=chosen, files=files, budget=budget)


def by_heat(files: dict[int, File], budget: int) -> Placement:
    """Hottest per byte first, which needs read counters and is why they exist."""
    chosen: set[int] = set()
    spent = 0
    ranked = sorted(
        files.values(), key=lambda one: one.reads_per_day / max(one.size, 1), reverse=True
    )
    for file in ranked:
        if spent + file.size <= budget:
            chosen.add(file.number)
            spent += file.size
    return Placement(fast=chosen, files=files, budget=budget)


@functools.cache
def _fleet(kind: str, count: int = 200, seed: int = 87) -> dict[int, File]:
    """Files whose heat either follows age or does not."""
    source = random.Random(seed)
    made = {}
    for number in range(count):
        age = number
        size = source.randrange(50, 150)
        if kind == "recency":
            reads = max(1, int(1000 / (1 + age)))
        elif kind == "scattered":
            reads = source.choice((1, 1, 1, 5, 20, 400))
        elif kind == "archival":
            reads = 400 if age > count - 20 else 1
        else:
            raise ConfigError(f"{kind} is not a workload shape")
        made[number] = File(number=number, age=age, size=size, reads_per_day=reads)
    return made


def _budget(files: dict[int, File], share: float = 0.2) -> int:
    """A fast tier holding a fifth of the bytes."""
    return int(sum(file.size for file in files.values()) * share)


@functools.cache
def age_placement_wins_when_heat_follows_age() -> bool:
    """On the recency workload the two policies land within a few percent.

    When reads concentrate on recent files, newest-first and hottest-first choose nearly the
    same set, and the free policy is as good as the instrumented one. This is the workload
    everyone imagines, and on it the read counters buy nothing, which is worth knowing
    before paying for them.
    """
    files = _fleet("recency")
    budget = _budget(files)
    aged = by_age(files, budget).daily_cost()
    heated = by_heat(files, budget).daily_cost()
    return heated <= aged <= heated * 1.15


@functools.cache
def age_placement_loses_badly_when_heat_scatters() -> bool:
    """On scattered heat, newest-first costs 118,330 against the instrumented 21,850.

    A fifth of the files are hot and their age is uniform, so the age policy's fast tier is
    mostly cold recent files while hot old ones pay the slow rate. The heat policy fills the
    fast tier with exactly the earners. The gap is the value of the read counters, priced in
    slow reads.
    """
    files = _fleet("scattered")
    budget = _budget(files)
    aged = by_age(files, budget).daily_cost()
    heated = by_heat(files, budget).daily_cost()
    return aged > heated * 2


@functools.cache
def the_archival_shape_inverts_the_age_policy() -> bool:
    """When only the oldest files are read, newest-first is the worst possible choice.

    The archival workload reads the tail, compliance scans and yearly reports, and the age
    policy pins the never-read newcomers to the fast tier. Its cost approaches the everything
    slow ceiling while the heat policy approaches the everything fast floor for the same
    budget.
    """
    files = _fleet("archival")
    budget = _budget(files)
    aged = by_age(files, budget).daily_cost()
    heated = by_heat(files, budget).daily_cost()
    everything_slow = sum(file.reads_per_day * SLOW_COST for file in files.values())
    return heated < aged and aged > everything_slow * 0.8


@functools.cache
def the_budget_is_enforced() -> bool:
    """A placement over budget refuses to exist.

    The constraint is the whole problem; a policy that quietly exceeds it is solving an
    easier one and reporting the wrong answer.
    """
    files = _fleet("recency")
    try:
        Placement(fast=set(files), files=files, budget=10)
    except ConfigError:
        return True
    return False


def compare_the_shapes() -> list[dict]:
    """Both policies on all three workload shapes."""
    rows = []
    for kind in ("recency", "scattered", "archival"):
        files = _fleet(kind)
        budget = _budget(files)
        rows.append(
            {
                "workload": kind,
                "by_age": by_age(files, budget).daily_cost(),
                "by_heat": by_heat(files, budget).daily_cost(),
            }
        )
    return rows


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "age_wins_on_recency": age_placement_wins_when_heat_follows_age(),
        "age_loses_on_scatter": age_placement_loses_badly_when_heat_scatters(),
        "archival_inverts_age": the_archival_shape_inverts_the_age_policy(),
        "the_budget_binds": the_budget_is_enforced(),
    }
