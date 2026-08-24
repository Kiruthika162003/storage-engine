from __future__ import annotations

import contextlib
import functools
import random
from dataclasses import dataclass, field

from store.errors import Closed, ConfigError, Conflict
from store.mvcc import History, Snapshot

# Transactions over the version store, and the two moments a conflict can be caught.
#
# A transaction reads from a snapshot and buffers its writes, so it sees a frozen world plus its
# own changes and nothing else sees anything until commit. The question a design has to answer
# is when to discover that two transactions collided.
#
# Optimistic concurrency answers: at commit. Track what each transaction read, and at commit
# check whether any of it changed since the snapshot. Nothing blocks, deadlock is impossible,
# and the cost is that a transaction can do all of its work and then be told to throw it away.
#
# The alternative, locking at first touch, pays the opposite way: conflicts surface early, no
# work is wasted, and in exchange writers block readers, readers block writers, and two
# transactions can each hold what the other wants. This module builds the optimistic kind and
# measures the retry cost that is its price.

FIRST_WRITE_WINS = "first write wins"


@dataclass
class Txn:
    """One transaction: a snapshot to read from, buffers to write into."""

    history: History
    snapshot: Snapshot
    writes: dict[bytes, bytes | None] = field(default_factory=dict)
    read_keys: set[bytes] = field(default_factory=set)
    state: str = field(default="open")

    def get(self, key: bytes) -> bytes | None:
        """Read a key: own writes first, then the snapshot."""
        self._check()
        self.read_keys.add(key)
        if key in self.writes:
            return self.writes[key]
        found = self.history.get(key, self.snapshot)
        return found.value if found else None

    def put(self, key: bytes, value: bytes) -> None:
        """Buffer a write, invisible to everyone until commit."""
        self._check()
        if not key:
            raise ConfigError("a key needs at least one byte")
        self.writes[key] = value

    def delete(self, key: bytes) -> None:
        """Buffer a delete."""
        self._check()
        self.writes[key] = None

    def _check(self) -> None:
        if self.state != "open":
            raise Closed(f"the transaction is {self.state}")

    def as_dict(self) -> dict:
        """Flat mapping for logs."""
        return {
            "snapshot": self.snapshot.sequence,
            "writes": len(self.writes),
            "reads": len(self.read_keys),
            "state": self.state,
        }


@dataclass
class Manager:
    """Begins, commits and aborts transactions, and counts what conflicts cost."""

    history: History = field(default_factory=History)
    begun: int = field(default=0)
    committed: int = field(default=0)
    aborted: int = field(default=0)
    conflicts: int = field(default=0)

    def begin(self) -> Txn:
        """A new transaction against the current moment."""
        self.begun += 1
        return Txn(history=self.history, snapshot=self.history.snapshot())

    def commit(self, txn: Txn) -> int:
        """Validate and apply, or refuse and leave no trace.

        Validation is the optimistic bet settled: every key the transaction read must be
        unchanged since its snapshot, because the transaction's writes were computed from those
        reads. A write-write collision without a read is allowed to land; last write wins is
        the semantics of a plain put and two blind puts are no different.
        """
        if txn.state != "open":
            raise Closed(f"the transaction is {txn.state}")
        for key in txn.read_keys:
            now = self.history.get(key)
            then = self.history.get(key, txn.snapshot)
            now_value = now.value if now else None
            then_value = then.value if then else None
            if now_value != then_value:
                self.conflicts += 1
                self._finish(txn, "aborted")
                raise Conflict(f"{key!r} changed after the snapshot")
        for key, value in txn.writes.items():
            if value is None:
                self.history.delete(key)
            else:
                self.history.put(key, value)
        self._finish(txn, "committed")
        return self.history.sequence

    def abort(self, txn: Txn) -> None:
        """Throw the buffers away."""
        if txn.state != "open":
            raise Closed(f"the transaction is {txn.state}")
        self._finish(txn, "aborted")

    def _finish(self, txn: Txn, state: str) -> None:
        txn.state = state
        self.history.release(txn.snapshot)
        if state == "committed":
            self.committed += 1
        else:
            self.aborted += 1

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "begun": self.begun,
            "committed": self.committed,
            "aborted": self.aborted,
            "conflicts": self.conflicts,
        }


def transfer(manager: Manager, source: bytes, target: bytes, amount: int) -> bool:
    """Move an amount between two counters, retrying is the caller's business."""
    txn = manager.begin()
    held = int.from_bytes(txn.get(source) or b"\x00", "big")
    if held < amount:
        manager.abort(txn)
        return False
    other = int.from_bytes(txn.get(target) or b"\x00", "big")
    txn.put(source, (held - amount).to_bytes(8, "big"))
    txn.put(target, (other + amount).to_bytes(8, "big"))
    try:
        manager.commit(txn)
    except Conflict:
        return False
    return True


def balance(manager: Manager, key: bytes) -> int:
    """One counter's current value."""
    found = manager.history.value(key)
    return int.from_bytes(found or b"\x00", "big")


@functools.cache
def _contended(accounts: int = 10, attempts: int = 2000, seed: int = 31) -> Manager:
    """A manager after a storm of interleaved transfers, many of them colliding."""
    source = random.Random(seed)
    manager = Manager()
    setup = manager.begin()
    for at in range(accounts):
        setup.put(f"acct:{at:02d}".encode(), (1000).to_bytes(8, "big"))
    manager.commit(setup)
    pending = []
    for _ in range(attempts):
        if pending and source.random() < 0.5:
            txn, giving, taking = pending.pop(source.randrange(len(pending)))
            held = int.from_bytes(txn.get(giving) or b"\x00", "big")
            other = int.from_bytes(txn.get(taking) or b"\x00", "big")
            if held >= 1:
                txn.put(giving, (held - 1).to_bytes(8, "big"))
                txn.put(taking, (other + 1).to_bytes(8, "big"))
                with contextlib.suppress(Conflict):
                    manager.commit(txn)
            else:
                manager.abort(txn)
        else:
            giving = f"acct:{source.randrange(accounts):02d}".encode()
            taking = f"acct:{source.randrange(accounts):02d}".encode()
            if giving == taking:
                continue
            pending.append((manager.begin(), giving, taking))
    for txn, _, _ in pending:
        manager.abort(txn)
    return manager


@functools.cache
def money_is_conserved_under_interleaving() -> bool:
    """Two thousand interleaved transfers over ten accounts, and the total never moves.

    Transactions are begun, held open while others commit, and finished in random order, which
    is the schedule that breaks read-modify-write without isolation. Every commit either sees a
    consistent pair of balances or is refused, so the sum across accounts after the storm is
    exactly what the setup deposited.

    This is the test that a lost update fails: two transfers read the same balance, both
    subtract, and one subtraction vanishes. The conflict check turns the second one into a
    refusal instead.

    The first run of this check failed, and the fault was not in the transaction machinery. A
    transfer where the two accounts came up equal read the balance twice, buffered a debit and
    then buffered a credit over it, and the credit overwrote the debit: one unit minted, every
    isolation level satisfied. The transaction did exactly what it was told. Conservation is a
    property of the client program too, and a sum check is the only thing here that could have
    caught it.
    """
    manager = _contended()
    total = sum(balance(manager, f"acct:{at:02d}".encode()) for at in range(10))
    return total == 10 * 1000 and manager.conflicts > 0


@functools.cache
def a_lost_update_is_impossible_by_construction() -> bool:
    """The textbook interleaving, run literally.

    Two transactions read the same counter at 100, both add 50, both try to commit. Without
    validation the counter ends at 150 and one update is silently gone. Here the first commit
    lands, the second is refused with a conflict, and a retry of the loser lands at 200.
    """
    manager = Manager()
    setup = manager.begin()
    setup.put(b"counter", (100).to_bytes(8, "big"))
    manager.commit(setup)
    first, second = manager.begin(), manager.begin()
    a = int.from_bytes(first.get(b"counter"), "big")
    b = int.from_bytes(second.get(b"counter"), "big")
    first.put(b"counter", (a + 50).to_bytes(8, "big"))
    second.put(b"counter", (b + 50).to_bytes(8, "big"))
    manager.commit(first)
    try:
        manager.commit(second)
        return False
    except Conflict:
        pass
    retry = manager.begin()
    c = int.from_bytes(retry.get(b"counter"), "big")
    retry.put(b"counter", (c + 50).to_bytes(8, "big"))
    manager.commit(retry)
    return balance(manager, b"counter") == 200


@functools.cache
def blind_writes_do_not_conflict() -> bool:
    """Two transactions that write the same key without reading it both commit.

    The validation protects reads, not keys. A write computed from nothing the transaction saw
    cannot be invalidated by anything that happened, so refusing it would be refusing a plain
    put for arriving second. Last write wins, exactly as two puts would.
    """
    manager = Manager()
    first, second = manager.begin(), manager.begin()
    first.put(b"k", b"first")
    second.put(b"k", b"second")
    manager.commit(first)
    manager.commit(second)
    return manager.history.value(b"k") == b"second"


@functools.cache
def a_transaction_reads_its_own_writes_and_nobody_elses() -> bool:
    """The buffer is visible to its owner and to no one else until commit.

    A transaction that writes a key and reads it back gets its own value. A second transaction
    open at the same time reads through to the snapshot and sees nothing, and only after the
    first commits does a third transaction, begun later, see the write.
    """
    manager = Manager()
    writer = manager.begin()
    writer.put(b"k", b"mine")
    other = manager.begin()
    unseen = other.get(b"k")
    own = writer.get(b"k")
    manager.commit(writer)
    manager.abort(other)
    later = manager.begin()
    seen = later.get(b"k")
    manager.abort(later)
    return own == b"mine" and unseen is None and seen == b"mine"


@functools.cache
def a_finished_transaction_refuses_everything() -> bool:
    """Committed and aborted transactions are closed, not reusable.

    Reusing a transaction after commit silently reads from a stale snapshot, which is the same
    class of bug as the double release in the version store, and it is refused the same way.
    """
    manager = Manager()
    txn = manager.begin()
    txn.put(b"k", b"v")
    manager.commit(txn)
    calls = (lambda: txn.get(b"k"), lambda: txn.put(b"k", b"x"), lambda: manager.commit(txn))
    for call in calls:
        try:
            call()
            return False
        except Closed:
            continue
    return True


@functools.cache
def the_conflict_rate_is_the_contention_not_the_load() -> bool:
    """The same transaction count over ten accounts and a thousand collides far more often.

    Spread the storm over a thousand accounts and conflicts nearly vanish, because a conflict
    needs two transactions on the same key with overlapping lifetimes. The retry cost of
    optimistic concurrency is therefore not a property of throughput. It is a property of how
    hot the hottest keys are, which is why the design suits stores whose load is spread and
    punishes stores with one famous counter.
    """
    hot = _contended(10, 2000, 31)
    cool = _contended(1000, 2000, 31)
    return hot.conflicts > cool.conflicts * 3


def compare_the_contention(attempts: int = 2000) -> list[dict]:
    """A row per account count, conflicts against spread.

    The rate is not monotonic: two accounts shows a lower rate than ten, because half the
    drawn pairs at two accounts are self transfers the storm discards, which thins the set of
    overlapping transactions. The contention effect is real from ten accounts outward and the
    two account row is a measurement artefact worth leaving visible.
    """
    rows = []
    for accounts in (2, 10, 100, 1000):
        manager = _contended(accounts, attempts, 31)
        rows.append(
            {
                "accounts": accounts,
                "committed": manager.committed,
                "conflicts": manager.conflicts,
                "conflict_rate": round(manager.conflicts / max(manager.begun, 1), 4),
            }
        )
    return rows


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "money_is_conserved": money_is_conserved_under_interleaving(),
        "no_lost_updates": a_lost_update_is_impossible_by_construction(),
        "blind_writes_land": blind_writes_do_not_conflict(),
        "own_writes_are_visible": a_transaction_reads_its_own_writes_and_nobody_elses(),
        "finished_means_finished": a_finished_transaction_refuses_everything(),
        "conflicts_track_contention": the_conflict_rate_is_the_contention_not_the_load(),
    }
