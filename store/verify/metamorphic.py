from __future__ import annotations

import functools
import random

from store.engine import Store, crash

# Properties that need no oracle, only a second run of the same store.
#
# The model checker needs a dictionary to compare against. A metamorphic property needs
# nothing but the store itself: run an operation twice, or run it and its inverse, or run the
# same reads through a different maintenance history, and the answers must relate in a way the
# contract fixes. These catch a class the model can miss, bugs in the comparison itself, and
# they cost nothing to generalise because there is no oracle to keep in sync.


def _drive(store: Store, writes: int, keys: int, seed: int) -> None:
    """A deterministic churn against any store."""
    source = random.Random(seed)
    for _ in range(writes):
        key = f"k{source.randrange(keys):05d}".encode()
        if source.random() < 0.12:
            store.delete(key)
        else:
            store.put(key, source.randbytes(10))


def _snapshot(store: Store) -> list[tuple[bytes, bytes]]:
    """The live contents, which every property compares."""
    return store.items()


@functools.cache
def maintenance_history_is_invisible() -> bool:
    """The same writes with wildly different flush and fold schedules read identically.

    One store never flushes, one flushes every 50 records, one folds constantly. The live
    contents are byte for byte identical, which is the statement that maintenance is an
    implementation detail. Any divergence here is a bug in flush or fold that the read path
    is faithfully reporting.
    """
    quiet = Store(flush_at=10**9, fold_at=10**9)
    eager = Store(flush_at=50, fold_at=2)
    middling = Store(flush_at=500, fold_at=4)
    for store in (quiet, eager, middling):
        _drive(store, 3000, 400, 71)
    return _snapshot(quiet) == _snapshot(eager) == _snapshot(middling)


@functools.cache
def a_crash_after_a_flush_is_invisible() -> bool:
    """Flush, crash, and the survivor's contents equal the uncrashed twin's.

    The flush put everything durable, so the crash had nothing to take. This pins the flush
    ordering from the outside: if any acknowledged record were still only in memory after the
    flush returned, this comparison would catch it.
    """
    steady = Store(flush_at=200, fold_at=4)
    crashed = Store(flush_at=200, fold_at=4)
    for store in (steady, crashed):
        _drive(store, 2000, 300, 72)
    crashed.flush()
    survivor = crash(crashed)
    return _snapshot(steady) == _snapshot(survivor)


@functools.cache
def rewriting_the_live_set_into_a_fresh_store_is_a_fixed_point() -> bool:
    """Copy a store's live contents into an empty store and the copy reads the same.

    Export import round trips are how stores migrate, and the property that makes them safe
    is that the live set fully determines the observable store. Nothing about the original's
    tombstones, stale versions or file layout is needed to reproduce its behaviour.
    """
    original = Store(flush_at=300, fold_at=3)
    _drive(original, 3000, 400, 73)
    copy = Store(flush_at=100, fold_at=2)
    for key, value in original.items():
        copy.put(key, value)
    return _snapshot(copy) == _snapshot(original)


@functools.cache
def deleting_everything_and_rewriting_restores_the_contents() -> bool:
    """Delete every live key, write the saved set back, and the store reads as before.

    The middle state is measured too: after the deletes the store is observably empty even
    though it holds more records than ever, which is the tombstone story told through the
    public interface alone.
    """
    store = Store(flush_at=300, fold_at=3)
    _drive(store, 2500, 300, 74)
    saved = store.items()
    for key, _ in saved:
        store.delete(key)
    emptied = store.items() == []
    for key, value in saved:
        store.put(key, value)
    return emptied and store.items() == saved


@functools.cache
def a_scan_is_its_gets() -> bool:
    """Every pair a scan returns is confirmed by a point get, and the counts agree.

    The scan and the get take different paths through the engine, and this property says the
    paths agree. A filter bug that loses a key for gets but not scans, or a merge bug that
    resurrects one for scans but not gets, lands exactly here.
    """
    store = Store(flush_at=250, fold_at=3)
    _drive(store, 2500, 300, 75)
    listed = store.items()
    if any(store.get(key) != value for key, value in listed):
        return False
    probes = [f"k{at:05d}".encode() for at in range(300)]
    gettable = sum(1 for key in probes if store.get(key) is not None)
    return gettable == len(listed)


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "maintenance_is_invisible": maintenance_history_is_invisible(),
        "a_flushed_crash_is_invisible": a_crash_after_a_flush_is_invisible(),
        "the_live_set_is_the_store": (
            rewriting_the_live_set_into_a_fresh_store_is_a_fixed_point()
        ),
        "delete_all_rewrite_restores": (
            deleting_everything_and_rewriting_restores_the_contents()
        ),
        "a_scan_is_its_gets": a_scan_is_its_gets(),
    }
