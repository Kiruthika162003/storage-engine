from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

from store.errors import Closed, Conflict

# Pessimistic locking, built to lose the comparison it sometimes wins.
#
# The transaction module went optimistic: nothing blocks, conflicts surface at commit, wasted
# work is the price. Locking is the other answer. A transaction takes a lock before touching a
# key, conflicts surface at the touch, no work is ever wasted, and the price is the deadlock:
# two transactions each holding what the other wants, forever, unless something notices.
#
# The something is a wait-for graph. Each blocked transaction points at the holder it waits
# on, a cycle in that graph is a deadlock by definition, and the resolution is to abort
# somebody, here the youngest, on the theory that it has done the least work worth saving.
# The measurements run the same contended transfers as the optimistic module and compare the
# two prices: retries there, deadlock aborts here.


@dataclass
class Locker:
    """One client of the lock table."""

    name: int
    holding: set[bytes] = field(default_factory=set)
    waiting_for: bytes | None = field(default=None)
    state: str = field(default="open")


@dataclass
class Table:
    """The lock table and its wait-for graph."""

    lockers: dict[int, Locker] = field(default_factory=dict)
    owners: dict[bytes, int] = field(default_factory=dict)
    next_name: int = field(default=0)
    granted: int = field(default=0)
    blocked: int = field(default=0)
    deadlocks: int = field(default=0)

    def begin(self) -> Locker:
        """A new client."""
        self.next_name += 1
        made = Locker(name=self.next_name)
        self.lockers[made.name] = made
        return made

    def acquire(self, locker: Locker, key: bytes) -> bool:
        """Take a lock, or join the queue; a deadlock aborts the youngest in the cycle.

        The model is cooperative rather than threaded: True means granted, False means the
        caller is now waiting and should call again once the holder releases, and a Conflict
        means the caller was chosen as the deadlock victim. The first draft of this method
        returned False and quietly granted the lock anyway, which is not waiting, it is
        stealing, and the textbook deadlock sailed through it undetected because nobody was
        ever actually blocked.
        """
        if locker.state != "open":
            raise Closed(f"locker {locker.name} is {locker.state}")
        holder = self.owners.get(key)
        if holder is None or holder == locker.name:
            self.owners[key] = locker.name
            locker.holding.add(key)
            locker.waiting_for = None
            self.granted += 1
            return True
        locker.waiting_for = key
        self.blocked += 1
        victim = self._cycle_victim(locker)
        if victim is not None:
            self.deadlocks += 1
            self._abort(victim)
            if victim is locker:
                raise Conflict(f"locker {locker.name} chosen as the deadlock victim")
            if self.owners.get(key) is None:
                self.owners[key] = locker.name
                locker.holding.add(key)
                locker.waiting_for = None
                self.granted += 1
                return True
        return False

    def _cycle_victim(self, start: Locker) -> Locker | None:
        """Walk the wait-for graph from a blocked locker, hunting a cycle."""
        seen = [start]
        at = start
        while at.waiting_for is not None:
            holder_name = self.owners.get(at.waiting_for)
            if holder_name is None:
                return None
            holder = self.lockers[holder_name]
            if holder in seen:
                return max(seen[seen.index(holder) :], key=lambda one: one.name)
            seen.append(holder)
            at = holder
        return None

    def _abort(self, locker: Locker) -> None:
        """Take everything back."""
        for key in locker.holding:
            if self.owners.get(key) == locker.name:
                del self.owners[key]
        locker.holding.clear()
        locker.waiting_for = None
        locker.state = "aborted"

    def release(self, locker: Locker) -> None:
        """Commit's tail: give every lock back."""
        if locker.state != "open":
            raise Closed(f"locker {locker.name} is {locker.state}")
        self._abort(locker)
        locker.state = "committed"

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "granted": self.granted,
            "blocked": self.blocked,
            "deadlocks": self.deadlocks,
            "held": len(self.owners),
        }


@functools.cache
def a_free_lock_grants_immediately() -> bool:
    """Uncontended locking is bookkeeping and nothing else.

    One client takes three locks, releases them, and another takes the same three. Every
    grant is immediate. The uncontended path is what nearly every real acquisition takes,
    and its cost being a dictionary write is why locking survives despite its worst case.
    """
    table = Table()
    first = table.begin()
    grants = [table.acquire(first, key) for key in (b"a", b"b", b"c")]
    table.release(first)
    second = table.begin()
    grants += [table.acquire(second, key) for key in (b"a", b"b", b"c")]
    table.release(second)
    return all(grants) and table.deadlocks == 0


@functools.cache
def the_textbook_deadlock_is_detected_and_the_youngest_dies() -> bool:
    """A holds x wants y, B holds y wants x: the cycle is found, the younger aborts.

    The wait-for graph makes the deadlock a structural fact rather than a timeout guess. B
    is younger, B dies, B's locks return to the table, and A can finish. Timeout based
    detection cannot tell a deadlock from a slow holder, which is why it always waits too
    long or kills too eagerly.
    """
    table = Table()
    older = table.begin()
    younger = table.begin()
    table.acquire(older, b"x")
    table.acquire(younger, b"y")
    still_waiting = table.acquire(older, b"y")
    try:
        table.acquire(younger, b"x")
        return False
    except Conflict:
        pass
    finished = table.acquire(older, b"y")
    return (
        not still_waiting
        and younger.state == "aborted"
        and finished
        and table.deadlocks == 1
    )


@functools.cache
def an_aborted_locker_frees_everything_it_held() -> bool:
    """The victim's locks are back in the table the moment it dies.

    A victim that kept its locks would deadlock everyone behind it, turning the resolution
    into a bigger version of the problem, so the abort and the release are one motion.
    """
    table = Table()
    victim = table.begin()
    table.acquire(victim, b"a")
    table.acquire(victim, b"b")
    table._abort(victim)
    bystander = table.begin()
    return table.acquire(bystander, b"a") and table.acquire(bystander, b"b")


@functools.cache
def deadlocks_track_contention_like_conflicts_did() -> bool:
    """The two lock transfer storm: hot keys deadlock, spread keys barely do.

    Two harness bugs preceded this measurement and both are worth their sentence. The first
    storm aborted any locker the moment a wait began, so nobody stood in the wait-for graph
    and no cycle could close. The second took both locks inside one iteration, so nothing
    was ever held when anyone else arrived and the blocked count was zero at every
    contention. A deadlock needs transactions that hold across time, which means the storm
    has to be a state machine: take the first lock, come back later for the second.

    Run that way, the hot keyspace of three keys deadlocks ten times while five hundred
    keys produce none, which is the optimistic module's contention curve with
    the other policy plugged in. The price moved from commit time retries to acquisition
    time aborts; its driver, how hot the hottest keys are, did not move at all.
    """
    def storm(keys: int, seed: int) -> Table:
        source = random.Random(seed)
        table = Table()
        active: list[list] = []
        for _ in range(500):
            advance = active and source.random() < 0.6
            if advance:
                entry = active.pop(source.randrange(len(active)))
                locker, second = entry
                if locker.state != "open":
                    continue
                try:
                    if table.acquire(locker, second):
                        table.release(locker)
                    else:
                        active.append(entry)
                except Conflict:
                    pass
                continue
            locker = table.begin()
            first = f"k{source.randrange(keys):04d}".encode()
            second = f"k{source.randrange(keys):04d}".encode()
            try:
                if table.acquire(locker, first):
                    active.append([locker, second])
            except Conflict:
                pass
        return table

    hot = storm(3, 91)
    cool = storm(500, 91)
    return hot.deadlocks >= 10 and cool.deadlocks <= hot.deadlocks / 5


@functools.cache
def a_finished_locker_refuses_more_work() -> bool:
    """Committed and aborted lockers are closed, the same rule as everywhere else."""
    table = Table()
    done = table.begin()
    table.acquire(done, b"a")
    table.release(done)
    try:
        table.acquire(done, b"b")
        return False
    except Closed:
        pass
    try:
        table.release(done)
        return False
    except Closed:
        return True


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "free_locks_grant_immediately": a_free_lock_grants_immediately(),
        "the_youngest_dies": the_textbook_deadlock_is_detected_and_the_youngest_dies(),
        "aborts_free_the_locks": an_aborted_locker_frees_everything_it_held(),
        "deadlocks_track_contention": deadlocks_track_contention_like_conflicts_did(),
        "finished_means_finished": a_finished_locker_refuses_more_work(),
    }
