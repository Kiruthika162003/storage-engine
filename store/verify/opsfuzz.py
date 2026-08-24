from __future__ import annotations

import functools
import random

from store.engine import Store
from store.rangedel import Ranged
from store.ttl import Shelf
from store.writebatch import Batch, commit

# Model checking for the modules the first model checker never met.
#
# The verify.model module holds the engine to a dictionary; this one gives the same
# treatment to three later modules whose semantics are richer than put and get: range
# deletes, TTL expiry, and write batches. Each gets a reference model simple enough to be
# right, a random program, and a diff at every read. The programs leans deliberately on the
# semantic corners the modules' own tests name: sequence-versus-geometry for spans, the
# exclusive deadline for TTLs, insertion order for batches.


def fuzz_ranged(steps: int = 3000, seed: int = 0) -> str:
    """The range delete store against a dict, deletes applied as comprehensions."""
    source = random.Random(seed)
    store = Ranged()
    model: dict[bytes, bytes] = {}
    for at in range(steps):
        roll = source.random()
        key = f"k{source.randrange(300):04d}".encode()
        if roll < 0.55:
            value = source.randbytes(6)
            store.put(key, value)
            model[key] = value
        elif roll < 0.7:
            if key in model:
                store.delete(key)
                del model[key]
        elif roll < 0.85:
            start = f"k{source.randrange(300):04d}".encode()
            stop = f"k{source.randrange(300):04d}".encode()
            if start < stop:
                store.delete_range(start, stop)
                model = {
                    held: value
                    for held, value in model.items()
                    if not start <= held < stop
                }
        else:
            wanted = model.get(key)
            got = store.get(key)
            if got != wanted:
                return f"step {at}: get {key!r} gave {got!r} wanted {wanted!r}"
    if sorted(model) != store.keys():
        return "final keys diverged"
    return ""


def fuzz_shelf(steps: int = 3000, seed: int = 1) -> str:
    """The TTL shelf against a dict of deadlines."""
    source = random.Random(seed)
    shelf = Shelf()
    model: dict[bytes, tuple[bytes, int | None]] = {}
    now = 0
    for at in range(steps):
        roll = source.random()
        key = f"k{source.randrange(120):03d}".encode()
        if roll < 0.5:
            value = source.randbytes(5)
            ttl = source.choice((None, 1, 3, 10))
            shelf.put(key, value, ttl=ttl)
            model[key] = (value, now + ttl if ttl else None)
        elif roll < 0.65:
            ticks = source.randrange(1, 4)
            shelf.tick(ticks)
            now += ticks
        elif roll < 0.8:
            shelf.sweep()
        else:
            held = model.get(key)
            wanted = None
            if held is not None:
                value, deadline = held
                if deadline is None or now < deadline:
                    wanted = value
            got = shelf.get(key)
            if got != wanted:
                return f"step {at}: get {key!r} gave {got!r} wanted {wanted!r}"
    return ""


def fuzz_batches(steps: int = 400, seed: int = 2) -> str:
    """Batched commits against a dict applied batch-atomically."""
    source = random.Random(seed)
    store = Store(flush_at=10**9, fold_at=10**9)
    model: dict[bytes, bytes] = {}
    for at in range(steps):
        batch = Batch()
        staged: dict[bytes, bytes | None] = {}
        for _ in range(source.randrange(1, 6)):
            key = f"k{source.randrange(80):03d}".encode()
            if source.random() < 0.75:
                value = source.randbytes(5)
                batch.put(key, value)
                staged[key] = value
            else:
                batch.delete(key)
                staged[key] = None
        commit(store, batch)
        for key, value in staged.items():
            if value is None:
                model.pop(key, None)
            else:
                model[key] = value
        probe = f"k{source.randrange(80):03d}".encode()
        if store.get(probe) != model.get(probe):
            return f"step {at}: probe {probe!r} diverged"
    for key, value in model.items():
        if store.get(key) != value:
            return f"final: {key!r} diverged"
    return ""


@functools.cache
def every_fuzzer_runs_clean_across_seeds() -> bool:
    """Three modules, five seeds each, no divergence anywhere."""
    for seed in range(5):
        if fuzz_ranged(2000, seed):
            return False
        if fuzz_shelf(2000, seed):
            return False
        if fuzz_batches(200, seed):
            return False
    return True


def summarise() -> dict:
    """Every claim in this module, run."""
    return {"the_late_modules_hold": every_fuzzer_runs_clean_across_seeds()}
