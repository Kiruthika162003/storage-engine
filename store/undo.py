from __future__ import annotations

import functools
from dataclasses import dataclass, field

from store.errors import ConfigError

# Undo against redo: the two logs, and the buffering policies that force the choice.
#
# The package's wal is a redo log: it records what committed writes will redo after a
# crash, and it works because the engine never writes uncommitted data to files, no steal,
# and never requires committed data to be in files, no force, on its own schedule. Page
# systems with big transactions cannot hold both: a transaction larger than the buffer
# pool must spill uncommitted pages, steal, and then a crash exposes uncommitted data on
# disk, which redo cannot fix because redo only reapplies. The undo log is the other half:
# it records how to reverse what should not have happened. The model here runs the policy
# matrix over crashes and shows which log each cell needs, which is the ARIES lecture as
# a measurement.


@dataclass
class System:
    """Pages on disk, a buffer, two logs, and the policy switches."""

    steal: bool
    force: bool
    disk: dict[str, int] = field(default_factory=dict)
    buffer: dict[str, int] = field(default_factory=dict)
    redo_log: list[tuple[str, str, int]] = field(default_factory=list)
    undo_log: list[tuple[str, str, int]] = field(default_factory=list)
    active: str | None = field(default=None)
    committed: list[str] = field(default_factory=list)

    def begin(self, name: str) -> None:
        if self.active is not None:
            raise ConfigError("one transaction at a time in this model")
        self.active = name

    def write(self, key: str, value: int) -> None:
        """A transactional write: buffered, undo noted before any steal could expose it."""
        if self.active is None:
            raise ConfigError("no transaction is active")
        before = self.buffer.get(key, self.disk.get(key, 0))
        self.undo_log.append((self.active, key, before))
        self.buffer[key] = value

    def maybe_steal(self, key: str) -> None:
        """Memory pressure: evict a dirty page mid transaction if the policy allows."""
        if not self.steal or key not in self.buffer:
            return
        self.disk[key] = self.buffer.pop(key)

    def commit(self) -> None:
        """Commit under the force policy."""
        if self.active is None:
            raise ConfigError("no transaction is active")
        for key, value in list(self.buffer.items()):
            self.redo_log.append((self.active, key, value))
        if self.force:
            for key, value in list(self.buffer.items()):
                self.disk[key] = value
            self.buffer.clear()
        self.committed.append(self.active)
        self.active = None

    def crash_and_recover(self) -> dict[str, int]:
        """Lose the buffer, then undo the losers and redo the winners."""
        self.buffer.clear()
        state = dict(self.disk)
        for name, key, before in reversed(self.undo_log):
            if name not in self.committed:
                state[key] = before
        for name, key, value in self.redo_log:
            if name in self.committed:
                state[key] = value
        return state


def _run(steal: bool, force: bool, crash_mid: bool) -> dict[str, int]:
    """One committed transaction, one mid-flight loser, a crash, recovery."""
    system = System(steal=steal, force=force)
    system.disk["a"] = 10
    system.disk["b"] = 20
    system.begin("winner")
    system.write("a", 11)
    system.commit()
    system.begin("loser")
    system.write("b", 99)
    system.maybe_steal("b")
    if crash_mid:
        return system.crash_and_recover()
    return dict(system.disk) | dict(system.buffer)


@functools.cache
def no_steal_no_force_needs_only_redo() -> bool:
    """The engine's cell: uncommitted data never reaches disk, so undo has nothing to do.

    A crash mid loser recovers a=11 and b=20 from redo alone: the loser's write existed
    only in the lost buffer. This is why the package's wal never needed an undo half, and
    the sentence is worth having: no-steal is what redo-only quietly assumes.
    """
    state = _run(steal=False, force=False, crash_mid=True)
    return state["a"] == 11 and state["b"] == 20


@functools.cache
def steal_without_undo_persists_the_losers_write() -> bool:
    """Turn on steal and drop the undo pass: the crashed loser's 99 survives on disk.

    The eviction wrote uncommitted data home, the crash killed the transaction, and
    nothing reverses the write, because redo can only reapply winners. The corruption is
    silent: b reads 99, a value no committed transaction ever wrote.
    """
    system = System(steal=True, force=False)
    system.disk["b"] = 20
    system.begin("loser")
    system.write("b", 99)
    system.maybe_steal("b")
    system.buffer.clear()
    state = dict(system.disk)
    for name, key, value in system.redo_log:
        if name in system.committed:
            state[key] = value
    return state["b"] == 99


@functools.cache
def steal_with_undo_reverses_the_losers_write() -> bool:
    """The same eviction with the undo pass: b comes back to 20.

    The undo record was written before the page could be stolen, write-ahead in its
    original sense, so recovery always holds the reversal for anything the steal exposed.
    Steal is safe exactly when undo is present and ordered before the exposure.
    """
    state = _run(steal=True, force=False, crash_mid=True)
    return state["b"] == 20 and state["a"] == 11


@functools.cache
def force_makes_redo_redundant_and_slow() -> bool:
    """Under force, the winner's data is on disk at commit, and redo finds nothing to add.

    Recovery without replaying the redo log at all still shows a=11, because force flushed
    it at commit. The price is the flush inside every commit, the sync-per-record lesson
    from the wal module wearing a buffering policy, and it is why no-force plus redo won:
    the log sync is sequential and the page flushes are not.
    """
    system = System(steal=False, force=True)
    system.disk["a"] = 10
    system.begin("winner")
    system.write("a", 11)
    system.commit()
    system.buffer.clear()
    return system.disk["a"] == 11


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "the_engines_cell_is_redo_only": no_steal_no_force_needs_only_redo(),
        "steal_without_undo_corrupts": steal_without_undo_persists_the_losers_write(),
        "steal_with_undo_recovers": steal_with_undo_reverses_the_losers_write(),
        "force_prepays_redo": force_makes_redo_redundant_and_slow(),
    }
