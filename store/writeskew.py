from __future__ import annotations

import functools

from store.errors import Conflict
from store.txn import Manager

# Write skew: the anomaly snapshot isolation permits, which this package's manager
# turned out to refuse already.
#
# The plan was the canonical demonstration: two doctors each check that the other is
# rostered, each signs only themselves off, snapshot isolation admits both commits, the
# ward is empty. Run against the transaction module's manager, the second commit
# conflicted, and the reason is worth the module: that manager validates every key a
# transaction read, and each doctor reads the other's key, which the other writes. Read
# set validation is strictly stronger than snapshot isolation, and the textbook anomaly
# needs the weaker rule to exist. So the weaker rule is built here explicitly, first
# committer wins on writes alone, the anomaly is produced under it, and the materialised
# guard repairs it. The package's own manager needs no guard, and knowing which rule a
# system runs is the difference between paying for one and not.


def roster(manager: Manager) -> None:
    """Two doctors on call."""
    setup = manager.begin()
    setup.put(b"oncall:alice", b"yes")
    setup.put(b"oncall:bob", b"yes")
    manager.commit(setup)


def on_call_count(txn) -> int:
    """How many doctors a transaction sees rostered."""
    return sum(
        1 for name in (b"oncall:alice", b"oncall:bob") if txn.get(name) == b"yes"
    )


def commit_si(manager: Manager, txn) -> None:
    """True snapshot isolation: validate writes only, first committer wins.

    The manager's own commit validates reads; this one ignores them and checks only that
    no key the transaction wrote has been written since its snapshot, which is the rule
    the write skew literature assumes.
    """
    if txn.state != "open":
        raise Conflict("the transaction is finished")
    for key in txn.writes:
        now = manager.history.get(key)
        then = manager.history.get(key, txn.snapshot)
        now_value = now.value if now else None
        then_value = then.value if then else None
        if now_value != then_value:
            manager.conflicts += 1
            manager._finish(txn, "aborted")
            raise Conflict(f"{key!r} written since the snapshot")
    for key, value in txn.writes.items():
        if value is None:
            manager.history.delete(key)
        else:
            manager.history.put(key, value)
    manager._finish(txn, "committed")


def sign_off_snapshot(manager: Manager, name: bytes) -> bool:
    """The buggy procedure under true SI: check the premise, write your own key only."""
    txn = manager.begin()
    if on_call_count(txn) < 2:
        manager.abort(txn)
        return False
    txn.put(b"oncall:" + name, b"no")
    try:
        commit_si(manager, txn)
        return True
    except Conflict:
        return False


def sign_off_guarded(manager: Manager, name: bytes) -> bool:
    """The serializable repair: write the premise as well as the conclusion.

    Materialising the constraint into a key both transactions write turns the disjoint
    write sets into overlapping reads-of-changed-data, which the existing first-committer
    validation already refuses. The general SSI machinery tracks read predicates instead;
    the write is the same idea spent at the schema level.
    """
    txn = manager.begin()
    if on_call_count(txn) < 2:
        manager.abort(txn)
        return False
    txn.get(b"oncall:guard")
    txn.put(b"oncall:guard", name)
    txn.put(b"oncall:" + name, b"no")
    try:
        manager.commit(txn)
        return True
    except Conflict:
        return False


def _interleave(manager: Manager, procedure) -> tuple[bool, bool]:
    """Both doctors run the procedure concurrently: read phase together, commits in turn."""
    alice = manager.begin()
    bob = manager.begin()
    alice_sees = on_call_count(alice)
    bob_sees = on_call_count(bob)
    outcomes = []
    for txn, name in ((alice, b"alice"), (bob, b"bob")):
        if (alice_sees if name == b"alice" else bob_sees) < 2:
            manager.abort(txn)
            outcomes.append(False)
            continue
        if procedure == "guarded":
            txn.get(b"oncall:guard")
            txn.put(b"oncall:guard", name)
        txn.put(b"oncall:" + name, b"no")
        try:
            if procedure == "occ":
                manager.commit(txn)
            else:
                commit_si(manager, txn)
            outcomes.append(True)
        except Conflict:
            outcomes.append(False)
    return outcomes[0], outcomes[1]


@functools.cache
def true_snapshot_isolation_empties_the_ward() -> bool:
    """Under write-only validation both commits land and nobody is on call.

    Each read both roster keys, each wrote only its own, the write sets are disjoint, and
    first-committer-wins finds nothing to refuse. The premise both relied on was destroyed
    by the pair while held by each: on_call_count lands at zero with zero conflicts, the
    anomaly on the record.
    """
    manager = Manager()
    roster(manager)
    alice_ok, bob_ok = _interleave(manager, "snapshot")
    audit = manager.begin()
    remaining = on_call_count(audit)
    manager.abort(audit)
    return alice_ok and bob_ok and remaining == 0 and manager.conflicts == 0


@functools.cache
def the_packages_own_manager_already_refuses_it() -> bool:
    """The same interleaving through the read-validating commit stops the second doctor.

    Bob read Alice's roster key and Alice wrote it, so Bob's premise is visibly stale to a
    validator that tracks reads, and the commit conflicts. This is the module's origin
    story: the demonstration failed against the house manager, because read set validation
    is strictly stronger than snapshot isolation, and the failure was the finding. A system
    on this rule needs no guard key; a system on SI does; and most production defaults are
    SI, which is why the guard idiom exists.
    """
    manager = Manager()
    roster(manager)
    alice_ok, bob_ok = _interleave(manager, "occ")
    audit = manager.begin()
    remaining = on_call_count(audit)
    manager.abort(audit)
    return alice_ok and not bob_ok and remaining == 1


@functools.cache
def the_materialised_guard_repairs_true_si() -> bool:
    """With the guard key, even write-only validation stops the second doctor.

    Both transactions write the guard, so the write sets overlap and first committer wins
    refuses the second, one doctor stays rostered, the anomaly gone for the price of one
    hot key. The guard is contention by design, which is the honest cost: serializability
    under SI is bought by making the invisible premise a visible, conflicted object.
    """
    manager = Manager()
    roster(manager)
    alice_ok, bob_ok = _interleave(manager, "guarded")
    audit = manager.begin()
    remaining = on_call_count(audit)
    manager.abort(audit)
    return alice_ok and not bob_ok and remaining == 1 and manager.conflicts == 1


@functools.cache
def sequential_sign_offs_never_needed_the_guard() -> bool:
    """Run one at a time, the snapshot procedure is already correct.

    Bob's read happens after Alice's commit, sees one doctor, and aborts himself, under
    plain SI with no guard. The anomaly needs the overlap, which is why it survives
    testing: serial tests cannot produce it, and the guard's cost is paid precisely for
    the schedules the test suite does not run.
    """
    manager = Manager()
    roster(manager)
    first = sign_off_snapshot(manager, b"alice")
    second = sign_off_snapshot(manager, b"bob")
    audit = manager.begin()
    remaining = on_call_count(audit)
    manager.abort(audit)
    return first and not second and remaining == 1


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "the_ward_empties_under_si": true_snapshot_isolation_empties_the_ward(),
        "read_validation_refuses_it": the_packages_own_manager_already_refuses_it(),
        "the_guard_repairs_si": the_materialised_guard_repairs_true_si(),
        "serial_runs_hide_the_anomaly": sequential_sign_offs_never_needed_the_guard(),
    }
