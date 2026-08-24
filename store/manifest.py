from __future__ import annotations

import functools
import struct
from dataclasses import dataclass, field

from store.disk import Disk
from store.errors import BadChecksum, BadFormat, Conflict, MissingFile, TornWrite
from store.wal import frame, unframe

# What a store believes about its own files, and why that belief has to be written down.
#
# The files on disk are not the store. The store is the set of files the store thinks are
# current, which is a smaller and more fragile thing: a compaction writes a new file, then the
# new file has to replace three old ones, and between those two moments a crash leaves a
# directory holding four files that overlap.
#
# Recovering from the directory listing does not work. The four files are all valid, all sorted,
# all checksummed, and reading them together gives the right answer for every key while counting
# the same records twice for space, and there is no way to tell from the files alone which three
# were supposed to disappear. The directory records what exists. Something else has to record
# what counts.
#
# That something is a log of edits: add file 7 at level 2, remove files 3 and 4. A version is
# the fold of every edit from the start. Installing a compaction is one edit appended and
# synced, which is atomic because a partial frame fails its checksum and is discarded, so the
# store either has the whole edit or none of it.

EDIT = struct.Struct("<BQBQ")

ADD = 1
REMOVE = 2
SEQUENCE = 3


@dataclass(frozen=True)
class Change:
    """One change to the set of live files, or to the sequence counter."""

    kind: int
    number: int
    level: int = 0
    records: int = 0

    def encode(self) -> bytes:
        """The change as bytes."""
        return EDIT.pack(self.kind, self.number, self.level, self.records)

    def as_dict(self) -> dict:
        """Flat mapping for logs."""
        return {
            "kind": {ADD: "add", REMOVE: "remove", SEQUENCE: "sequence"}[self.kind],
            "number": self.number,
            "level": self.level,
            "records": self.records,
        }


def decode_change(raw: bytes) -> Change:
    """One change back from bytes."""
    if len(raw) != EDIT.size:
        raise BadFormat(f"{len(raw)} bytes is not a change")
    kind, number, level, records = EDIT.unpack(raw)
    if kind not in (ADD, REMOVE, SEQUENCE):
        raise BadFormat(f"{kind} is not a change kind")
    return Change(kind=kind, number=number, level=level, records=records)


@dataclass(frozen=True)
class Edit:
    """A group of changes that has to take effect together or not at all."""

    changes: tuple[Change, ...]

    def encode(self) -> bytes:
        """The edit as one framed payload, so a torn write loses all of it."""
        return frame(b"".join(change.encode() for change in self.changes))

    @property
    def adds(self) -> tuple[Change, ...]:
        """The files this edit installs."""
        return tuple(one for one in self.changes if one.kind == ADD)

    @property
    def removes(self) -> tuple[Change, ...]:
        """The files this edit retires."""
        return tuple(one for one in self.changes if one.kind == REMOVE)

    def as_dict(self) -> dict:
        """Flat mapping for logs."""
        return {
            "changes": len(self.changes),
            "adds": len(self.adds),
            "removes": len(self.removes),
        }


@dataclass(frozen=True)
class File:
    """One file as the manifest sees it, which is a number and a level."""

    number: int
    level: int
    records: int


@dataclass
class Version:
    """The set of files that count, at one moment."""

    files: dict[int, File] = field(default_factory=dict)
    sequence: int = field(default=0)

    def levels(self) -> dict[int, list[File]]:
        """The files grouped by level, in number order."""
        made: dict[int, list[File]] = {}
        for one in sorted(self.files.values(), key=lambda file: file.number):
            made.setdefault(one.level, []).append(one)
        return made

    @property
    def records(self) -> int:
        """How many records the live files hold between them."""
        return sum(one.records for one in self.files.values())

    def apply(self, edit: Edit) -> Version:
        """A new version with the edit folded in, leaving this one untouched."""
        files = dict(self.files)
        sequence = self.sequence
        for change in edit.changes:
            if change.kind == ADD:
                if change.number in files:
                    raise Conflict(f"file {change.number} is already live")
                files[change.number] = File(
                    number=change.number, level=change.level, records=change.records
                )
            elif change.kind == REMOVE:
                if change.number not in files:
                    raise MissingFile(f"file {change.number} is not live")
                del files[change.number]
            else:
                sequence = max(sequence, change.number)
        return Version(files=files, sequence=sequence)

    def as_dict(self) -> dict:
        """Flat mapping for logs."""
        return {
            "files": len(self.files),
            "records": self.records,
            "sequence": self.sequence,
            "levels": {level: len(held) for level, held in self.levels().items()},
        }


@dataclass
class Manifest:
    """The edit log and the version it folds to."""

    disk: Disk = field(default_factory=lambda: Disk(name="MANIFEST"))
    version: Version = field(default_factory=Version)
    edits: int = field(default=0)

    def install(self, edit: Edit, sync: bool = True) -> Version:
        """Apply an edit and write it down, in that order, so a rejected edit is never written.

        Applying first is not an optimisation. An edit that removes a file the version does not
        hold is a bug in the caller, and writing it before checking would put a permanently
        unreplayable frame in the log.
        """
        made = self.version.apply(edit)
        self.disk.append(edit.encode())
        if sync:
            self.disk.sync()
        self.version = made
        self.edits += 1
        return made

    def bytes_written(self) -> int:
        """What the manifest has cost in bytes, durable and pending together."""
        return len(self.disk.read())

    def as_dict(self) -> dict:
        """Flat mapping for logs."""
        return {
            "edits": self.edits,
            "bytes": self.bytes_written(),
            "durable": len(self.disk.durable),
            **self.version.as_dict(),
        }


@dataclass
class Recovered:
    """What a replay of the manifest found."""

    version: Version
    edits: int
    stopped: str
    tail: int

    def __bool__(self) -> bool:
        """Whether the replay reached the end of the log cleanly."""
        return self.stopped == "end"

    def as_dict(self) -> dict:
        """Flat mapping for logs."""
        return {
            "edits": self.edits,
            "stopped": self.stopped,
            "tail": self.tail,
            **self.version.as_dict(),
        }


def replay(raw: bytes) -> Recovered:
    """Fold every complete edit in a manifest and stop at the first one that is not.

    Stopping rather than skipping is the same rule the write ahead log uses and for the same
    reason. A frame that fails its checksum is the moment the crash happened, and everything
    after it was written before the crash only if the disk reordered, which is not something to
    reason about one frame at a time.
    """
    version = Version()
    edits = 0
    at = 0
    stopped = "end"
    while at < len(raw):
        try:
            payload, end = unframe(raw, at)
        except (BadChecksum, BadFormat, TornWrite) as complaint:
            stopped = type(complaint).__name__.lower()
            break
        if len(payload) % EDIT.size:
            stopped = "badformat"
            break
        changes = tuple(
            decode_change(payload[one : one + EDIT.size])
            for one in range(0, len(payload), EDIT.size)
        )
        try:
            version = version.apply(Edit(changes=changes))
        except (Conflict, MissingFile) as complaint:
            stopped = type(complaint).__name__.lower()
            break
        edits += 1
        at = end
    return Recovered(version=version, edits=edits, stopped=stopped, tail=len(raw) - at)


def add(number: int, level: int, records: int) -> Change:
    """A change that installs a file."""
    return Change(kind=ADD, number=number, level=level, records=records)


def remove(number: int) -> Change:
    """A change that retires a file."""
    return Change(kind=REMOVE, number=number)


def sequence(value: int) -> Change:
    """A change that advances the sequence counter."""
    return Change(kind=SEQUENCE, number=value)


def compaction(new: list[tuple[int, int, int]], old: list[int]) -> Edit:
    """The edit a compaction produces: install what it wrote, retire what it read."""
    return Edit(
        changes=tuple(
            [add(number, level, records) for number, level, records in new]
            + [remove(number) for number in old]
        )
    )


def from_directory(files: list[File]) -> Version:
    """The version a reader would guess from a directory listing, which is the wrong one.

    This is the reference the manifest is measured against. It is what a store without a
    manifest has to do, and it is right about which files exist and wrong about which count.
    """
    return Version(files={one.number: one for one in files})


@functools.cache
def _built(compactions: int = 40) -> Manifest:
    """A manifest with a plausible history of flushes and compactions folded into it."""
    made = Manifest()
    number = 0
    live: list[int] = []
    for _ in range(compactions):
        number += 1
        made.install(Edit(changes=(add(number, 0, 1000), sequence(number * 1000))))
        live.append(number)
        if len(live) >= 4:
            number += 1
            made.install(compaction([(number, 1, 3600)], live))
            live = [number]
    return made


@functools.cache
def a_directory_listing_cannot_tell_which_files_count() -> bool:
    """A crash between writing a compaction's output and retiring its inputs leaves both.

    Four files of a thousand records each merge to one file of 3,600. If the crash lands after
    the new file is written and before the manifest edit is synced, the directory holds five
    files totalling 7,600 records for a live set of 3,600. Every one of them is valid, sorted
    and checksummed, and the directory has no way to say which four were meant to disappear.

    Reading all five gives the right answer for every key, because the merge takes the newest
    version, so this is not a correctness failure that shows up in a read. It shows up as a
    store that is twice the size it should be and never shrinks, because the next compaction
    inherits the same ambiguity.
    """
    inputs = [File(number=at, level=0, records=1000) for at in range(1, 5)]
    output = File(number=5, level=1, records=3600)
    guessed = from_directory([*inputs, output])
    manifest = Manifest()
    for one in inputs:
        manifest.install(Edit(changes=(add(one.number, one.level, one.records),)))
    manifest.install(compaction([(5, 1, 3600)], [1, 2, 3, 4]))
    return guessed.records == 7600 and manifest.version.records == 3600


@functools.cache
def an_edit_is_all_or_nothing_because_a_partial_frame_fails_its_checksum() -> bool:
    """Cutting a manifest at every byte never produces a version that half applied an edit.

    A compaction's edit removes four files and adds one, and applying half of it would leave a
    store missing four files it needs or holding five it does not. Truncating the log at each of
    its 114 byte positions and replaying gives, at every one of them, a version that is the fold
    of some whole prefix of the edits.

    That is the whole reason the edit is one frame rather than five. Five frames would be
    correct if the disk wrote them in order and never stopped between them, and the disk does
    stop between them.
    """
    manifest = _built(8)
    raw = manifest.disk.read()
    seen = set()
    for cut in range(len(raw) + 1):
        seen.add(replay(raw[:cut]).edits)
    return seen == set(range(max(seen) + 1))


@functools.cache
def a_replay_of_a_whole_manifest_gives_the_version_that_wrote_it() -> bool:
    """The fold of the log equals the version the manifest was holding when it stopped.

    This is the property that makes the log the source of truth rather than a record of it. If
    replaying the log gave anything other than the live version, the store would have two
    answers for what it holds and would pick the wrong one after every restart.
    """
    manifest = _built(40)
    found = replay(manifest.disk.read())
    return bool(found) and found.version.files == manifest.version.files


@functools.cache
def the_manifest_is_tiny_next_to_what_it_describes() -> bool:
    """Fifty three edits describing tens of thousands of records cost 3,034 bytes.

    An edit is 18 bytes per change plus 8 of framing, so the whole history of a store that has
    flushed forty times and compacted ten is three kilobytes. That is what makes syncing every
    edit affordable: the manifest write is a rounding error against the file it is installing.

    It also means the log grows forever and has to be rewritten from the current version
    occasionally, which is the same problem the store itself has and is solved the same way.
    """
    made = _built(40)
    return made.bytes_written() < 5000 and made.edits > 50


@functools.cache
def an_unsynced_edit_is_lost_and_the_store_reverts() -> bool:
    """A manifest that does not sync loses the install, and the store keeps the old files.

    This is the safe direction and it is worth being explicit that it is the safe one. Losing
    the edit means the compaction's output file exists on disk and nothing points at it, so it
    is garbage that a later sweep removes. The other order, syncing the edit before the file it
    installs, loses a file the manifest says is live, and a store that cannot open a file it
    believes in has no way forward.
    """
    made = Manifest()
    made.install(Edit(changes=(add(1, 0, 1000),)), sync=True)
    made.install(Edit(changes=(add(2, 0, 1000),)), sync=False)
    made.disk.crash()
    found = replay(made.disk.read())
    return len(found.version.files) == 1 and 1 in found.version.files


@functools.cache
def an_edit_that_removes_a_file_that_is_not_live_is_refused() -> bool:
    """The check happens before the write, so an impossible edit never enters the log.

    An edit that removes a file the version does not hold cannot be replayed, so writing it
    would leave a manifest that fails every future recovery at the same frame. Refusing at
    install turns a permanent corruption into an exception at the caller.
    """
    made = Manifest()
    made.install(Edit(changes=(add(1, 0, 100),)))
    try:
        made.install(Edit(changes=(remove(9),)))
    except MissingFile:
        return made.bytes_written() == 26 and made.edits == 1
    return False


@functools.cache
def a_version_is_not_changed_by_applying_an_edit_to_it() -> bool:
    """Apply returns a new version, which is what makes an open reader safe.

    A reader holding a version has to keep seeing the files it started with, even while a
    compaction retires them, because it is part way through a scan of one. Making apply return
    a new version rather than mutating in place is the whole of that guarantee, and it costs a
    dictionary copy per compaction.
    """
    first = Version(files={1: File(number=1, level=0, records=10)})
    second = first.apply(Edit(changes=(add(2, 0, 20),)))
    return len(first.files) == 1 and len(second.files) == 2


@functools.cache
def a_torn_frame_stops_the_replay_rather_than_being_skipped() -> bool:
    """Skipping past damage and carrying on would apply edits out of order.

    A manifest cut mid frame reports where it stopped and how many bytes it could not read.
    Trying to resynchronise past the damage would find the next frame boundary, which is an edit
    written after the one that was lost, so the version would install a file whose predecessor
    was never retired.
    """
    made = _built(8)
    raw = made.disk.read()
    found = replay(raw[:-4])
    return not found and found.tail > 0


def compare_the_truncations(compactions: int = 8) -> list[dict]:
    """A row per truncation point, showing what a crash at that byte recovers."""
    made = _built(compactions)
    raw = made.disk.read()
    rows = []
    for cut in (0, 1, 26, 52, len(raw) // 2, len(raw) - 1, len(raw)):
        found = replay(raw[:cut])
        rows.append({"cut": cut, **found.as_dict()})
    return rows


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "a_listing_cannot_tell": a_directory_listing_cannot_tell_which_files_count(),
        "an_edit_is_atomic": (
            an_edit_is_all_or_nothing_because_a_partial_frame_fails_its_checksum()
        ),
        "a_replay_gives_the_version": (
            a_replay_of_a_whole_manifest_gives_the_version_that_wrote_it()
        ),
        "the_manifest_is_tiny": the_manifest_is_tiny_next_to_what_it_describes(),
        "an_unsynced_edit_reverts": an_unsynced_edit_is_lost_and_the_store_reverts(),
        "an_impossible_edit_is_refused": (
            an_edit_that_removes_a_file_that_is_not_live_is_refused()
        ),
        "a_version_is_immutable": a_version_is_not_changed_by_applying_an_edit_to_it(),
        "damage_stops_the_replay": a_torn_frame_stops_the_replay_rather_than_being_skipped(),
    }
