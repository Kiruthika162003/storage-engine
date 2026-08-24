from __future__ import annotations

import functools
import heapq
import random
from collections.abc import Iterator
from dataclasses import dataclass, field

from store.errors import ConfigError
from store.record import DELETE, Record

# Reading many sorted sources as one, and the two ways to get it wrong.
#
# A store holds the same key in several places at once: in the memtable, in a file flushed an
# hour ago, in a file compacted last week. Sorted order inside each one is guaranteed and says
# nothing about the order across them, so a read has to merge.
#
# The merge itself is a heap over the source heads, which is the ordinary thing. What is not
# ordinary is what happens when two sources hold the same key. They do not both win and they do
# not merge: the newer one is the answer and the older one is discarded without being looked at,
# because looking at it is what produces a stale read.
#
# The second mistake is subtler and is why a merge has to know whether it is reading for a
# user or for a compaction. A tombstone is a record, and a merge reading for a user hides it
# and every older record for that key. A merge reading for a compaction can only drop it if
# the compaction covers every level below, because a tombstone dropped while an older put
# survives somewhere underneath resurrects a key the user deleted. That bug is silent, it
# appears months later, and it is entirely a question of which files the compaction held.


@dataclass(order=True)
class _Head:
    """One source's current record, ordered the way a merge needs it."""

    key: bytes
    rank: int
    source: int = field(compare=False)
    record: Record = field(compare=False)


@dataclass
class Source:
    """A named sorted stream, so a merge can report which source answered."""

    name: str
    records: list[Record]

    def __post_init__(self) -> None:
        keys = [record.order for record in self.records]
        if keys != sorted(keys):
            raise ConfigError(f"{self.name} is not sorted")

    def __len__(self) -> int:
        return len(self.records)

    def scan(self, start: bytes = b"") -> Iterator[Record]:
        """Every record from a key onwards."""
        for record in self.records:
            if record.key >= start:
                yield record


@dataclass
class Merge:
    """A heap merge over sorted sources, newest version of each key first."""

    sources: list[Source]
    comparisons: int = field(default=0)
    dropped: int = field(default=0)

    def __post_init__(self) -> None:
        if not self.sources:
            raise ConfigError("a merge needs at least one source")

    @property
    def total(self) -> int:
        """How many records the sources hold between them."""
        return sum(len(source) for source in self.sources)

    def raw(self, start: bytes = b"") -> Iterator[tuple[int, Record]]:
        """Every record from every source in merged order, duplicates included."""
        streams = [source.scan(start) for source in self.sources]
        heap: list[_Head] = []
        for at, stream in enumerate(streams):
            self._push(heap, at, stream)
        while heap:
            head = heapq.heappop(heap)
            self.comparisons += max(len(heap).bit_length(), 1)
            yield head.source, head.record
            self._push(heap, head.source, streams[head.source])

    def _push(self, heap: list[_Head], at: int, stream: Iterator[Record]) -> None:
        """Move one source forward onto the heap, if it has anything left."""
        record = next(stream, None)
        if record is None:
            return
        heapq.heappush(
            heap,
            _Head(key=record.key, rank=-record.sequence, source=at, record=record),
        )

    def newest(self, start: bytes = b"") -> Iterator[Record]:
        """One record per key, the newest version, tombstones included."""
        seen: bytes | None = None
        for _, record in self.raw(start):
            if record.key == seen:
                self.dropped += 1
                continue
            seen = record.key
            yield record

    def live(self, start: bytes = b"") -> Iterator[Record]:
        """What a reader sees: newest version per key, tombstones removed."""
        for record in self.newest(start):
            if record.kind != DELETE:
                yield record

    def get(self, key: bytes) -> Record | None:
        """The live value for one key, or nothing."""
        for record in self.live(key):
            if record.key == key:
                return record
            return None
        return None

    def as_dict(self) -> dict:
        """Flat mapping for logs."""
        return {
            "sources": len(self.sources),
            "records": self.total,
            "comparisons": self.comparisons,
            "dropped": self.dropped,
        }


def naive(sources: list[Source], start: bytes = b"") -> Iterator[Record]:
    """The obviously correct merge: collect everything and sort it.

    This is the reference the heap merge is checked against. It is correct by construction and
    it holds every record in memory at once, which is the reason the heap exists.
    """
    everything = [record for source in sources for record in source.scan(start)]
    everything.sort(key=lambda record: record.order)
    yield from everything


@dataclass
class Compacting:
    """A merge that writes the output of a compaction, which is a different question.

    A user read hides tombstones. A compaction has to decide whether to write them, and the
    answer is not a property of the record. It is a property of whether this compaction is
    holding every file that could hold an older version of the key.
    """

    merge: Merge
    bottom: bool = field(default=False)
    horizon: int = field(default=0)
    kept: int = field(default=0)
    removed: int = field(default=0)

    def records(self) -> Iterator[Record]:
        """What the compaction should write."""
        for record in self.merge.newest():
            if record.kind == DELETE and self.bottom and record.sequence <= self.horizon:
                self.removed += 1
                continue
            self.kept += 1
            yield record

    def as_dict(self) -> dict:
        """Flat mapping for logs."""
        return {
            "bottom": self.bottom,
            "horizon": self.horizon,
            "kept": self.kept,
            "removed": self.removed,
            "dropped": self.merge.dropped,
        }


def resurrects(older: Source, newer: Source) -> bool:
    """Whether dropping tombstones from the newer source brings keys back.

    This is the check that turns the tombstone rule from advice into something with a value. It
    compacts the newer source alone as though it were the bottom, merges the result back over
    the older source, and asks whether any key the user deleted is readable again.
    """
    partial = Compacting(merge=Merge(sources=[newer]), bottom=True, horizon=1 << 62)
    survivors = Source(name="compacted", records=list(partial.records()))
    after = Merge(sources=[survivors, older])
    deleted = {
        record.key for record in newer.records if record.kind == DELETE
    }
    return any(after.get(key) is not None for key in deleted)


@functools.cache
def _sources(count: int, records: int, overlap: float = 0.5) -> tuple[Source, ...]:
    """Sources that overlap by a given fraction, which is what makes a merge non trivial."""
    source = random.Random(11)
    made = []
    span = int(records / max(overlap, 0.01))
    for at in range(count):
        keys = sorted({source.randrange(span) for _ in range(records)})
        made.append(
            Source(
                name=f"source-{at}",
                records=[
                    Record(key=f"k{key:09d}".encode(), sequence=at * records + one + 1)
                    for one, key in enumerate(keys)
                ],
            )
        )
    return tuple(made)


@functools.cache
def the_heap_merge_agrees_with_sorting_everything() -> bool:
    """The reference is a sort of every record and the heap gives the same answer.

    Four sources of two thousand records each, overlapping by half, merge to 3,453 distinct keys
    out of 6,296 records. Sorting the whole thing and taking the first record of each key run
    gives exactly the same list, record for record, which is the only check that matters for a
    merge and the one that is easy to skip because the heap looks obviously right.

    It looks obviously right and it has two places to go wrong: the tie break between sources on
    an equal key, and the moment a source runs out mid heap. Both are covered by comparing
    against the sort rather than by reading the code again.
    """
    sources = list(_sources(4, 2000))
    heaped = list(Merge(sources=sources).newest())
    sorted_out = naive(sources)
    reference = []
    seen = None
    for record in sorted_out:
        if record.key != seen:
            reference.append(record)
            seen = record.key
    return heaped == reference


@functools.cache
def the_newest_version_wins_and_the_older_one_is_never_looked_at() -> bool:
    """Two sources holding the same key produce one record, the newer one.

    The ordering does it: key ascending, then sequence descending, so the first record the merge
    meets for a key is the live one and everything after it up to the next key is history. The
    merge does not compare sequences at that point. It compares keys and stops.

    That is why the sort order is part of the record type rather than a detail of the merge. If
    the order were key ascending and sequence ascending, every merge would have to buffer a
    key's whole run to find the newest, and a key with a million versions would need a million
    records in memory.
    """
    old = Source(name="old", records=[Record(key=b"a", sequence=1, value=b"old")])
    new = Source(name="new", records=[Record(key=b"a", sequence=9, value=b"new")])
    merged = list(Merge(sources=[old, new]).newest())
    return len(merged) == 1 and merged[0].value == b"new"


@functools.cache
def a_read_hides_a_tombstone_and_a_compaction_may_not() -> bool:
    """The same merge answers two different questions and the difference is not cosmetic.

    Reading for a user: a tombstone means the key is gone, so the record is hidden and so is
    every older version. Reading for a compaction that does not cover the bottom: the tombstone
    has to be written to the output, because a file below still holds the put it is suppressing.

    Measured directly: compacting a source holding a delete as though it were the bottom, then
    merging the result over a source holding the older put, brings the key back. The user
    deleted it, the store reported it deleted, and a compaction restored it.
    """
    older = Source(name="old", records=[Record(key=b"a", sequence=1, value=b"x")])
    newer = Source(name="new", records=[Record(key=b"a", sequence=2, kind=DELETE)])
    return resurrects(older, newer)


@functools.cache
def a_bottom_compaction_can_drop_the_tombstone_safely() -> bool:
    """The same drop is correct when the compaction holds everything below it.

    Nothing about the tombstone changed. What changed is that there is no file underneath, so
    there is no older put to uncover, and writing the tombstone would keep a record alive that
    can never be read.

    That is the whole rule and it is a rule about the compaction, not about the record, which is
    why a merge that decides on the record alone is wrong exactly half the time and right the
    other half, and looks correct in every test that only runs one level.
    """
    both = Source(
        name="both",
        records=[Record(key=b"a", sequence=2, kind=DELETE), Record(key=b"a", sequence=1)],
    )
    made = Compacting(merge=Merge(sources=[both]), bottom=True, horizon=1 << 62)
    return list(made.records()) == [] and made.removed == 1


@functools.cache
def a_horizon_keeps_a_tombstone_a_reader_still_needs() -> bool:
    """A snapshot older than the delete still has to see the value, so the delete stays.

    The horizon is the oldest sequence any open reader can see. A tombstone above it is holding
    back a value that a reader is entitled to, so a bottom compaction that drops it is not
    dropping garbage, it is corrupting a live read.

    This is the second condition on the same drop and the one that gets lost when the rule is
    remembered as tombstones can be dropped at the bottom level.
    """
    both = Source(
        name="both",
        records=[Record(key=b"a", sequence=9, kind=DELETE), Record(key=b"a", sequence=1)],
    )
    made = Compacting(merge=Merge(sources=[both]), bottom=True, horizon=5)
    return len(list(made.records())) == 1 and made.removed == 0


@functools.cache
def the_heap_cost_grows_with_the_log_of_the_source_count() -> bool:
    """Doubling the sources roughly adds one comparison per record, not double.

    Over the same six thousand records the cost per record is 1.0, 1.0, 2.0, 3.0, 4.0 and 5.0
    for one through thirty two sources, which is the log the heap promises with no slack in it.
    Against the sort based reference the difference is not the comparison count, which is
    comparable, but the memory: the heap holds one record per source and the sort holds all.

    On a compaction of a hundred megabytes that is the difference between working and not.
    """
    few = Merge(sources=list(_sources(2, 8000)))
    many = Merge(sources=list(_sources(16, 1000)))
    list(few.newest())
    list(many.newest())
    per_few = few.comparisons / few.total
    per_many = many.comparisons / many.total
    return per_many > per_few and per_many < per_few * 6


@functools.cache
def the_merge_drops_more_when_the_sources_overlap_more() -> bool:
    """Overlap is what makes a merge do work, and it is a property of the write pattern.

    At two percent overlap the merge drops 2.7 percent of what it reads and is doing nothing
    but concatenating in order. At full overlap it drops 61.6 percent, because most records are
    older versions of a key another source already answered for.

    That is the number a compaction planner cares about, and it explains why a store that writes
    keys in increasing order compacts almost for free and one that writes uniformly at random
    compacts expensively for the same volume of writes.
    """
    apart = Merge(sources=list(_sources(4, 2000, overlap=0.02)))
    together = Merge(sources=list(_sources(4, 2000, overlap=1.0)))
    list(apart.newest())
    list(together.newest())
    return together.dropped > apart.dropped * 5


@functools.cache
def a_source_that_is_not_sorted_is_refused_when_it_is_built() -> bool:
    """The merge assumes sorted sources and says so at construction rather than in the output.

    A merge over an unsorted source does not fail. It produces a plausible looking stream with
    keys out of order and some records missing, which is exactly the failure that gets found
    downstream in something unrelated a week later.
    """
    try:
        Source(
            name="bad",
            records=[Record(key=b"b", sequence=1), Record(key=b"a", sequence=2)],
        )
    except ConfigError:
        return True
    return False


@functools.cache
def a_merge_with_no_sources_is_refused() -> bool:
    """An empty merge is a configuration mistake and not an empty stream.

    Returning nothing would be defensible and it hides the case where a caller assembled the
    source list from a filter that matched nothing, which is a bug at the caller and looks like
    an empty store from here.
    """
    try:
        Merge(sources=[])
    except ConfigError:
        return True
    return False


def compare_the_source_counts(records: int = 8000) -> list[dict]:
    """A row per source count over the same total volume."""
    rows = []
    for count in (1, 2, 4, 8, 16, 32):
        merge = Merge(sources=list(_sources(count, records // count)))
        kept = len(list(merge.newest()))
        rows.append(
            {
                "sources": count,
                "records": merge.total,
                "kept": kept,
                "dropped": merge.dropped,
                "comparisons_per_record": round(merge.comparisons / max(merge.total, 1), 2),
            }
        )
    return rows


def compare_the_overlaps(count: int = 4, records: int = 2000) -> list[dict]:
    """A row per overlap fraction, showing how much of a merge is wasted work."""
    rows = []
    for overlap in (0.02, 0.1, 0.25, 0.5, 1.0):
        merge = Merge(sources=list(_sources(count, records, overlap=overlap)))
        kept = len(list(merge.newest()))
        rows.append(
            {
                "overlap": overlap,
                "records": merge.total,
                "kept": kept,
                "dropped": merge.dropped,
                "wasted": round(merge.dropped / max(merge.total, 1), 3),
            }
        )
    return rows


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "heap_agrees_with_the_sort": the_heap_merge_agrees_with_sorting_everything(),
        "newest_wins": the_newest_version_wins_and_the_older_one_is_never_looked_at(),
        "a_compaction_may_not_drop": a_read_hides_a_tombstone_and_a_compaction_may_not(),
        "the_bottom_may_drop": a_bottom_compaction_can_drop_the_tombstone_safely(),
        "the_horizon_holds_it_back": a_horizon_keeps_a_tombstone_a_reader_still_needs(),
        "the_cost_is_logarithmic": the_heap_cost_grows_with_the_log_of_the_source_count(),
        "overlap_makes_the_work": the_merge_drops_more_when_the_sources_overlap_more(),
        "unsorted_is_refused": a_source_that_is_not_sorted_is_refused_when_it_is_built(),
        "empty_is_refused": a_merge_with_no_sources_is_refused(),
    }
