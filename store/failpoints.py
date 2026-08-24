from __future__ import annotations

import contextlib
import functools
from dataclasses import dataclass, field

from store.disk import Disk
from store.engine import Store, build_table, crash
from store.manifest import Edit, add, sequence
from store.memtable import Memtable
from store.wal import Log

# Failpoints: crashing on purpose at the exact line that matters.
#
# The crash fuzzer kills the store at random write counts, which finds ordering bugs
# eventually. A failpoint kills it at a named step of a specific operation, which finds them
# on the first run, and documents in the test's name exactly which window is being claimed
# safe. The operation instrumented here is the flush, whose ordering the engine module calls
# its entire contribution: file first, manifest second, log and memtable last. Each of the
# four windows gets a scripted crash and an assertion about what the survivor must know.


class Fall(Exception):
    """The scripted failure."""


@dataclass
class FlushWithFailpoints:
    """The engine's flush, unrolled so a failure can be injected between the steps."""

    store: Store
    fail_at: str = field(default="")
    steps: list[str] = field(default_factory=list)

    def run(self) -> None:
        """The flush, step by named step."""
        records = self.store.memtable.records()
        if not records:
            return
        self._step("before_anything")
        number = self.store.next_file
        table = build_table(number, records)
        self._step("file_written")
        self.store.next_file += 1
        self.store.tables.insert(0, table)
        self._step("table_installed")
        self.store.manifest.install(
            Edit(changes=(add(number, 0, len(records)), sequence(self.store.sequence)))
        )
        self._step("manifest_synced")
        self.store.wal = Log(disk=Disk(name=f"WAL-{number}"), policy=self.store.wal.policy)
        self.store.memtable = Memtable()
        self._step("log_dropped")
        self.store.flushes += 1

    def _step(self, name: str) -> None:
        self.steps.append(name)
        if name == self.fail_at:
            raise Fall(name)


def survivor_after(fail_at: str, writes: int = 300) -> tuple[Store, dict[bytes, bytes]]:
    """A store crashed at the named flush step, recovered, with its expected contents."""
    store = Store(flush_at=10**9, fold_at=10**9)
    truth: dict[bytes, bytes] = {}
    for at in range(writes):
        key = f"k{at:04d}".encode()
        value = at.to_bytes(4, "big")
        store.put(key, value)
        truth[key] = value
    flusher = FlushWithFailpoints(store=store, fail_at=fail_at)
    with contextlib.suppress(Fall):
        flusher.run()
    return crash(store), truth


@functools.cache
def a_crash_before_anything_replays_the_log() -> bool:
    """Nothing happened yet, so the survivor rebuilds every write from the log."""
    survivor, truth = survivor_after("before_anything")
    return all(survivor.get(key) == value for key, value in truth.items())


@functools.cache
def a_crash_after_the_file_is_written_is_harmless_duplication() -> bool:
    """The file exists, the manifest never heard of it, and recovery ignores it.

    This window is the one the manifest module's whole argument covers: an orphan file is
    garbage a sweep collects, not a correctness problem, because the manifest decides what
    counts. The survivor still answers everything, from the log.
    """
    survivor, truth = survivor_after("file_written")
    correct = all(survivor.get(key) == value for key, value in truth.items())
    orphan_invisible = len(survivor.tables) == 0
    return correct and orphan_invisible


@functools.cache
def a_crash_after_the_manifest_syncs_needs_no_log() -> bool:
    """The edit landed, so the survivor holds the file, and the log is now redundant.

    The window between the manifest sync and the log drop is the one where a write exists
    in two durable places at once, which wastes space and threatens nothing: replaying the
    log over the file writes the same records with the same sequences, and the newest-wins
    rule makes that a no-op. Idempotent replay is what makes this window boring, and boring
    is the design goal.
    """
    survivor, truth = survivor_after("manifest_synced")
    correct = all(survivor.get(key) == value for key, value in truth.items())
    return correct and len(survivor.tables) == 1


@functools.cache
def a_crash_after_the_log_drops_still_answers() -> bool:
    """The file carries everything now, and the empty log replays to nothing."""
    survivor, truth = survivor_after("log_dropped")
    correct = all(survivor.get(key) == value for key, value in truth.items())
    return correct and len(survivor.memtable.records()) == 0


@functools.cache
def every_window_answers_identically() -> bool:
    """The survivor's contents are the same at all four failpoints, which is the theorem.

    The flush ordering exists so that no window loses or duplicates a visible write, and
    saying it per window is weaker than saying it across them: five crashes at five lines,
    one observable store.
    """
    windows = (
        "before_anything",
        "file_written",
        "table_installed",
        "manifest_synced",
        "log_dropped",
    )
    contents = []
    for fail_at in windows:
        survivor, _ = survivor_after(fail_at)
        contents.append(survivor.items())
    return all(one == contents[0] for one in contents)


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "before_anything_replays": a_crash_before_anything_replays_the_log(),
        "orphan_files_are_harmless": (
            a_crash_after_the_file_is_written_is_harmless_duplication()
        ),
        "double_durability_is_boring": a_crash_after_the_manifest_syncs_needs_no_log(),
        "after_the_drop_the_file_carries": a_crash_after_the_log_drops_still_answers(),
        "all_windows_agree": every_window_answers_identically(),
    }
