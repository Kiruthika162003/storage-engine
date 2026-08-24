"""A store's whole life: writes, a crash, recovery, a checkpoint, a restore, verified.

Run with: python -m examples.lifecycle
"""

from __future__ import annotations

import random

from store.backup import library_of, restore, take
from store.engine import Store, crash
from store.verify.invariants import report

WRITES = 5000
KEYS = 1200


def a_working_day(store: Store, truth: dict, seed: int) -> None:
    source = random.Random(seed)
    for _ in range(WRITES):
        key = f"k{source.randrange(KEYS):05d}".encode()
        if source.random() < 0.1:
            store.delete(key)
            truth.pop(key, None)
        else:
            value = source.randbytes(12)
            store.put(key, value)
            truth[key] = value


def agree(store: Store, truth: dict) -> bool:
    return all(store.get(key) == value for key, value in truth.items())


def main() -> int:
    store = Store(flush_at=400, fold_at=4)
    truth: dict[bytes, bytes] = {}

    a_working_day(store, truth, seed=1)
    print(f"day one: {store.as_dict()}")
    print(f"agrees with the dictionary: {agree(store, truth)}")

    survivor = crash(store)
    print(f"after the crash: agrees={agree(survivor, truth)}")
    print(f"invariants: {report(survivor)['clean']}")

    saved = take(survivor)
    library = library_of(survivor)
    print(f"checkpoint cost {saved.cost} bytes for {survivor.as_dict()['tables']} tables")

    a_working_day(survivor, truth, seed=2)
    print(f"day two on the survivor: agrees={agree(survivor, truth)}")

    restored = restore(saved, library)
    stale_truth_hits = sum(1 for key in truth if restored.get(key) is not None)
    print(f"the restore is yesterday: {stale_truth_hits} of {len(truth)} current keys visible")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
