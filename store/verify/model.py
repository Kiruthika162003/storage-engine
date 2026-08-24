from __future__ import annotations

import random
from dataclasses import dataclass, field

from store.engine import Store, crash
from store.verify.invariants import check as check_invariants

# The store against a dictionary, driven by the same random program.
#
# The model is a dict and a counter. It is the specification: no logs, no files, no compaction,
# and nothing to get wrong. The checker generates a random program of puts, deletes, reads,
# scans, flushes, folds and crashes, runs it against both, and any disagreement at any step is
# a bug in the store at that step, with the whole program as the reproduction.
#
# The point of driving the maintenance operations from the program rather than letting them
# happen naturally is coverage: a flush that only happens at a fixed threshold happens at one
# memtable shape, and a flush that can happen after any write happens at all of them.

OPERATIONS = (
    "put", "put", "put", "put", "delete", "get", "get", "scan", "flush", "fold", "crash"
)


@dataclass
class Step:
    """One operation as the program ran it, kept for the reproduction."""

    operation: str
    key: bytes = field(default=b"")
    value: bytes = field(default=b"")

    def as_dict(self) -> dict:
        """Flat mapping for reports."""
        return {"operation": self.operation, "key": repr(self.key), "value": repr(self.value)}


@dataclass
class Outcome:
    """What one program run found."""

    steps: int
    disagreement: str = field(default="")
    at_step: int = field(default=-1)
    program: list[Step] = field(default_factory=list)

    def __bool__(self) -> bool:
        """A run is clean if nothing disagreed."""
        return not self.disagreement

    def as_dict(self) -> dict:
        """Flat mapping for reports."""
        return {
            "steps": self.steps,
            "clean": bool(self),
            "disagreement": self.disagreement or "none",
            "at_step": self.at_step,
        }


def run(steps: int = 2000, seed: int = 0, keys: int = 200, check_every: int = 250) -> Outcome:
    """One random program against the store and the model, verifying as it goes."""
    source = random.Random(seed)
    store = Store(flush_at=10**9, fold_at=10**9)
    model: dict[bytes, bytes] = {}
    program: list[Step] = []
    for at in range(steps):
        operation = source.choice(OPERATIONS)
        key = f"k{source.randrange(keys):04d}".encode()
        step = Step(operation=operation, key=key)
        program.append(step)
        if operation == "put":
            value = source.randbytes(8)
            step.value = value
            store.put(key, value)
            model[key] = value
        elif operation == "delete":
            store.delete(key)
            model.pop(key, None)
        elif operation == "get":
            wanted = model.get(key)
            got = store.get(key)
            if got != wanted:
                return Outcome(
                    steps=at + 1,
                    disagreement=f"get {key!r} gave {got!r} wanted {wanted!r}",
                    at_step=at,
                    program=program,
                )
        elif operation == "scan":
            wanted_items = sorted(model.items())
            got_items = store.items()
            if got_items != wanted_items:
                return Outcome(
                    steps=at + 1,
                    disagreement=f"scan gave {len(got_items)} items wanted {len(wanted_items)}",
                    at_step=at,
                    program=program,
                )
        elif operation == "flush":
            store.flush()
        elif operation == "fold":
            store.fold()
        elif operation == "crash":
            store = crash(store)
        if at % check_every == check_every - 1:
            broken = check_invariants(store)
            if broken:
                return Outcome(
                    steps=at + 1,
                    disagreement=f"invariant: {broken[0].detail}",
                    at_step=at,
                    program=program,
                )
    for key, value in model.items():
        if store.get(key) != value:
            return Outcome(
                steps=steps,
                disagreement=f"final read {key!r} disagreed",
                at_step=steps,
                program=program,
            )
    return Outcome(steps=steps, program=program)


def sweep(runs: int = 20, steps: int = 1000) -> dict:
    """Many programs from different seeds, summarised."""
    outcomes = [run(steps=steps, seed=seed) for seed in range(runs)]
    failed = [one for one in outcomes if not one]
    return {
        "runs": runs,
        "steps_each": steps,
        "clean": len(outcomes) - len(failed),
        "failed": len(failed),
        "first_failure": failed[0].as_dict() if failed else None,
    }
