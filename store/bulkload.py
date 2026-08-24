from __future__ import annotations

import functools
from dataclasses import dataclass

from store.engine import Store, build_table, crash
from store.errors import ConfigError, Conflict
from store.manifest import Edit, add, sequence
from store.record import Record

# Bulk loading: sorted input deserves a door that skips the queue.
#
# The write path exists to turn unsorted arrivals into sorted files, and every stage of it,
# the log, the memtable, the flush, the fold, is machinery for imposing order that already
# sorted input does not need. A bulk load takes presorted records, cuts them into files, and
# installs them with manifest edits, touching neither the log nor the memtable. The
# measurements price the difference, pin the safety conditions, and mark where the shortcut
# is refused: input that is not actually sorted, and input that overlaps the store's live
# range, either of which would corrupt silently if waved through.

FILE_RECORDS = 2000


@dataclass
class Loaded:
    """What one bulk load did."""

    records: int
    files: int
    log_bytes: int
    memtable_records: int

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "records": self.records,
            "files": self.files,
            "log_bytes": self.log_bytes,
            "memtable_records": self.memtable_records,
        }


def bulk_load(store: Store, records: list[Record], file_records: int = FILE_RECORDS) -> Loaded:
    """Install sorted records as files directly, refusing unsafe input."""
    if not records:
        raise ConfigError("an empty load loads nothing")
    keys = [record.key for record in records]
    if keys != sorted(keys):
        raise ConfigError("a bulk load requires sorted input")
    if len(set(keys)) != len(keys):
        raise ConfigError("a bulk load requires distinct keys")
    for table in store.tables:
        if keys[0] <= table.last and table.first <= keys[-1]:
            raise Conflict("the load overlaps live files; use the write path")
    if store.memtable.records():
        held = store.memtable.records()
        if keys[0] <= held[-1].key and held[0].key <= keys[-1]:
            raise Conflict("the load overlaps the memtable; flush first")
    log_before = store.wal.disk.size
    files = 0
    for at in range(0, len(records), file_records):
        chunk = records[at : at + file_records]
        number = store.next_file
        store.next_file += 1
        store.tables.append(build_table(number, chunk))
        top = max(record.sequence for record in chunk)
        store.manifest.install(
            Edit(changes=(add(number, 0, len(chunk)), sequence(top)))
        )
        store.sequence = max(store.sequence, top)
        files += 1
    return Loaded(
        records=len(records),
        files=files,
        log_bytes=store.wal.disk.size - log_before,
        memtable_records=len(store.memtable.records()),
    )


def _sorted_records(
    count: int, start_sequence: int = 1, prefix: bytes = b"bulk:"
) -> list[Record]:
    """Presorted distinct records, the input a backfill actually has."""
    return [
        Record(key=prefix + f"{at:08d}".encode(), sequence=start_sequence + at, value=bytes(16))
        for at in range(count)
    ]


@functools.cache
def a_load_writes_no_log_and_fills_no_memtable() -> bool:
    """Ten thousand records enter as five files, zero log bytes, zero memtable entries.

    The write path's log exists to make unsorted arrivals durable before they are sorted;
    bulk loaded records are durable as files the moment the manifest edit lands, so the log
    has nothing to protect. The same ten thousand through the front door cost a log frame
    each and every memtable ceremony in between.
    """
    store = Store(flush_at=500, fold_at=10**9)
    made = bulk_load(store, _sorted_records(10000))
    return made.files == 5 and made.log_bytes == 0 and made.memtable_records == 0


@functools.cache
def loaded_records_read_back_and_survive_a_crash() -> bool:
    """Every loaded key answers, and answers again after a crash with no replay.

    Durability came from the manifest edit, so the crash has nothing to lose: the survivor's
    manifest names the loaded files and the log replays nothing. This is the safety half of
    skipping the queue, measured rather than argued.
    """

    store = Store(flush_at=500, fold_at=10**9)
    bulk_load(store, _sorted_records(4000))
    if store.get(b"bulk:00001234") is None:
        return False
    survivor = crash(store)
    return all(
        survivor.get(f"bulk:{at:08d}".encode()) is not None for at in range(0, 4000, 97)
    )


@functools.cache
def unsorted_input_is_refused() -> bool:
    """Two swapped records reject the whole load before anything is installed.

    A bulk loader that trusts its caller installs a file whose index lies about its range,
    and every later read through that file is quietly wrong. The refusal is cheap, one
    comparison per record, against a corruption that no later check would find.
    """
    store = Store()
    records = _sorted_records(100)
    records[10], records[50] = records[50], records[10]
    try:
        bulk_load(store, records)
    except ConfigError:
        return not store.tables and store.manifest.edits == 0
    return False


@functools.cache
def an_overlapping_load_is_refused() -> bool:
    """A load into the live key range is turned back toward the write path.

    The loaded file would sit at level zero claiming its records are newest, while the live
    files hold versions with real sequence history, and the merge would resolve the tie by
    numbers the loader invented. Refusing costs the caller a flush or a different key
    prefix; accepting costs somebody a wrong read on an unknowable date.
    """
    store = Store(flush_at=100, fold_at=10**9)
    for at in range(300):
        store.put(f"bulk:{at:08d}".encode(), b"live")
    store.flush()
    try:
        bulk_load(store, _sorted_records(100))
    except Conflict:
        return True
    return False


@functools.cache
def a_disjoint_load_lands_beside_live_data() -> bool:
    """A load outside the live range coexists: both prefixes answer correctly after.

    Disjointness is what makes the shortcut safe, and the measurement writes through the
    front door, loads through the side door, and reads both back through the one read path.
    """
    store = Store(flush_at=100, fold_at=10**9)
    for at in range(300):
        store.put(f"live:{at:05d}".encode(), b"front")
    store.flush()
    bulk_load(store, _sorted_records(2000))
    front = all(store.get(f"live:{at:05d}".encode()) == b"front" for at in range(0, 300, 13))
    side = all(store.get(f"bulk:{at:08d}".encode()) is not None for at in range(0, 2000, 61))
    return front and side


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "no_log_no_memtable": a_load_writes_no_log_and_fills_no_memtable(),
        "loads_survive_crashes": loaded_records_read_back_and_survive_a_crash(),
        "unsorted_is_refused": unsorted_input_is_refused(),
        "overlap_is_refused": an_overlapping_load_is_refused(),
        "disjoint_loads_coexist": a_disjoint_load_lands_beside_live_data(),
    }
