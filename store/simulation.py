from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

from store.backup import library_of, restore, take
from store.engine import Store, crash
from store.verify.invariants import check as check_invariants

# The torture week: everything at once, which is what production is.
#
# Every module measured its subject alone. A store in production runs them together: a
# mixed workload, crashes at bad moments, checkpoints under load, restores that must not
# see the future, and the invariants held throughout. This module scripts seven days of
# that and asserts the whole way. It is the closest thing the package has to an
# integration bench, and its meters double as the demonstration that the parts compose:
# nothing here uses any hook the individual modules did not already have.

DAYS = 7
WRITES_PER_DAY = 2500


@dataclass
class Diary:
    """What one simulated week did and found."""

    writes: int = field(default=0)
    deletes: int = field(default=0)
    reads: int = field(default=0)
    read_mismatches: int = field(default=0)
    crashes: int = field(default=0)
    checkpoints: int = field(default=0)
    restores_checked: int = field(default=0)
    invariant_breaks: int = field(default=0)

    def clean(self) -> bool:
        return self.read_mismatches == 0 and self.invariant_breaks == 0

    def as_dict(self) -> dict:
        return {
            "writes": self.writes,
            "deletes": self.deletes,
            "reads": self.reads,
            "read_mismatches": self.read_mismatches,
            "crashes": self.crashes,
            "checkpoints": self.checkpoints,
            "restores_checked": self.restores_checked,
            "invariant_breaks": self.invariant_breaks,
            "clean": self.clean(),
        }


def a_week(seed: int = 0) -> Diary:
    """Seven days of everything."""
    source = random.Random(seed)
    store = Store(flush_at=400, fold_at=4)
    truth: dict[bytes, bytes] = {}
    diary = Diary()
    saved = None
    saved_truth: dict[bytes, bytes] = {}
    library: dict = {}
    for day in range(DAYS):
        for _ in range(WRITES_PER_DAY):
            key = f"k{source.randrange(1200):05d}".encode()
            roll = source.random()
            if roll < 0.6:
                value = source.randbytes(12)
                store.put(key, value)
                truth[key] = value
                diary.writes += 1
            elif roll < 0.72:
                store.delete(key)
                truth.pop(key, None)
                diary.deletes += 1
            else:
                diary.reads += 1
                if store.get(key) != truth.get(key):
                    diary.read_mismatches += 1
        if day == 2:
            store.flush()
            saved = take(store)
            saved_truth = dict(truth)
            library = library_of(store)
            diary.checkpoints += 1
        if day in (1, 4, 6):
            store = crash(store)
            diary.crashes += 1
        diary.invariant_breaks += len(check_invariants(store))
    if saved is not None:
        restored = restore(saved, library)
        diary.restores_checked += 1
        for key, value in saved_truth.items():
            if restored.get(key) != value:
                diary.read_mismatches += 1
    for key, value in truth.items():
        diary.reads += 1
        if store.get(key) != value:
            diary.read_mismatches += 1
    return diary


@functools.cache
def the_week_runs_clean_across_seeds() -> bool:
    """Three seeded weeks: zero mismatches, zero invariant breaks, all events fired.

    Seven days each of mixed writes, three crashes, a mid-week checkpoint restored and
    verified against the truth as it stood at checkpoint time, and the invariants swept
    after every day. The claim is composition: each behaviour was proven alone, and the
    week is the measurement that they stay proven together.
    """
    for seed in range(3):
        diary = a_week(seed)
        if not diary.clean():
            return False
        if diary.crashes != 3 or diary.checkpoints != 1 or diary.restores_checked != 1:
            return False
    return True


def summarise() -> dict:
    """Every claim in this module, run."""
    return {"the_week_is_clean": the_week_runs_clean_across_seeds()}
