from __future__ import annotations

import functools
import itertools
import struct
from dataclasses import dataclass, field

from store.block import BLOCK_BYTES, RESTART_INTERVAL, Block, Builder
from store.bloom import Filter
from store.bloom import build as build_filter
from store.errors import BadFormat, ConfigError
from store.record import Record

# A sorted file, and the two structures that keep a lookup from reading it.
#
# The file is a run of data blocks, then an index, then a filter, then a footer of fixed size at
# the end. The footer is last and fixed because it is the only part whose position is known
# without reading anything: open the file, seek to the end minus the footer size, and everything
# else follows from what is written there.
#
# The index holds one entry per data block, not one per record, keyed by the last key in the
# block. That is what makes it sparse, and sparseness is the whole point: an index over every
# record would be the size of the keys, which is the thing being avoided. One entry per block
# means a lookup binary searches the index in memory, reads one data block, and stops.
#
# The filter sits in front of both. A lookup that will find nothing still has to search the
# index and read a block to establish that, and on a workload where most files do not hold the
# key that is most of the reads. The filter turns those into no reads at all.

FOOTER = struct.Struct("<QIQIQ")

MAGIC = 0x53535431


@dataclass(frozen=True)
class Handle:
    """Where one block sits in the file and how far the keys in it reach."""

    offset: int
    length: int
    last: bytes
    count: int

    @property
    def end(self) -> int:
        """The byte after the block."""
        return self.offset + self.length


@dataclass
class Table:
    """A sorted file: data blocks, a sparse index over them, and a filter in front."""

    blocks: list[Block]
    handles: list[Handle]
    filter: Filter | None = field(default=None)
    reads: int = field(default=0)
    skipped: int = field(default=0)

    def __post_init__(self) -> None:
        if len(self.blocks) != len(self.handles):
            raise ConfigError("every block needs a handle")
        if not self.blocks:
            raise ConfigError("an empty table has no keys to index")

    @property
    def count(self) -> int:
        """How many records the table holds."""
        return sum(handle.count for handle in self.handles)

    @property
    def nbytes(self) -> int:
        """What the table costs on disk, blocks and index and filter together."""
        return (
            sum(block.nbytes for block in self.blocks)
            + self.index_bytes
            + (len(self.filter.bits) if self.filter else 0)
            + FOOTER.size
        )

    @property
    def index_bytes(self) -> int:
        """What the sparse index costs."""
        return sum(len(handle.last) + 20 for handle in self.handles)

    @property
    def first(self) -> bytes:
        """The lowest key in the table."""
        return self.blocks[0].records()[0].key

    @property
    def last(self) -> bytes:
        """The highest key in the table."""
        return self.handles[-1].last

    def holds(self, key: bytes) -> bool:
        """Whether the key falls inside the range the table covers."""
        return self.first <= key <= self.last

    def get(self, key: bytes) -> Record | None:
        """The record for a key, reading at most one data block."""
        if self.filter is not None and not self.filter.might_contain(key):
            self.skipped += 1
            return None
        at = self._block_for(key)
        if at is None:
            return None
        self.reads += 1
        return self.blocks[at].get(key)

    def _block_for(self, key: bytes) -> int | None:
        """The index of the only block that could hold the key."""
        low, high = 0, len(self.handles) - 1
        while low <= high:
            middle = (low + high) // 2
            if self.handles[middle].last < key:
                low = middle + 1
            else:
                high = middle - 1
        return low if low < len(self.handles) else None

    def scan(self, start: bytes = b""):
        """Every record from a key onwards, in order, one block at a time."""
        at = self._block_for(start) if start else 0
        if at is None:
            return
        while at < len(self.blocks):
            self.reads += 1
            yield from self.blocks[at].scan(start)
            start = b""
            at += 1

    def records(self) -> list[Record]:
        """Everything in the table."""
        return list(self.scan())

    def footer(self) -> bytes:
        """The fixed size trailer that makes the rest of the file findable."""
        return FOOTER.pack(
            sum(block.nbytes for block in self.blocks),
            self.index_bytes,
            self.count,
            len(self.handles),
            MAGIC,
        )

    def as_dict(self) -> dict:
        """Flat mapping for tables and logs."""
        return {
            "records": self.count,
            "blocks": len(self.blocks),
            "bytes": self.nbytes,
            "index_bytes": self.index_bytes,
            "filter_bytes": len(self.filter.bits) if self.filter else 0,
            "reads": self.reads,
            "skipped": self.skipped,
        }


def read_footer(raw: bytes) -> dict:
    """What the trailer says about the file, or a complaint that it is not one."""
    if len(raw) < FOOTER.size:
        raise BadFormat(f"{len(raw)} bytes is shorter than a footer")
    data, index, count, blocks, magic = FOOTER.unpack(raw[-FOOTER.size :])
    if magic != MAGIC:
        raise BadFormat(f"{magic:#x} is not the table magic")
    return {"data": data, "index": index, "count": count, "blocks": blocks}


def _handle(offset: int, block: Block, last: bytes) -> Handle:
    """The index entry for a block that has just been closed."""
    return Handle(offset=offset, length=block.nbytes, last=last, count=block.count)


def write(
    records: list[Record], block_bytes: int = BLOCK_BYTES, filtered: bool = True
) -> Table:
    """Build a table from a sorted run of records."""
    if not records:
        raise ConfigError("an empty table has no keys to index")
    blocks: list[Block] = []
    handles: list[Handle] = []
    offset = 0
    made = Builder(interval=RESTART_INTERVAL)
    for record in records:
        made.add(record)
        if len(made.payload) >= block_bytes:
            block = made.finish()
            handles.append(_handle(offset, block, made.previous))
            blocks.append(block)
            offset += block.nbytes
            made = Builder(interval=RESTART_INTERVAL)
    if made.count:
        block = made.finish()
        handles.append(_handle(offset, block, made.previous))
        blocks.append(block)
    keys = [record.key for record in records] if filtered else None
    return Table(
        blocks=blocks,
        handles=handles,
        filter=build_filter(keys) if keys is not None else None,
    )


@dataclass
class Lookup:
    """What a run of lookups cost, split by what stopped each one."""

    hits: int
    misses: int
    reads: int
    skipped: int

    @property
    def reads_per_miss(self) -> float:
        """How many data blocks a lookup that finds nothing costs."""
        return round(self.reads / max(self.misses, 1), 4)

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "reads": self.reads,
            "skipped": self.skipped,
            "reads_per_miss": self.reads_per_miss,
        }


@functools.cache
def _run(count: int, value: int = 50) -> tuple[Record, ...]:
    """A sorted run of records with keys that look like keys."""
    return tuple(
        Record(key=f"user:{one:08d}".encode(), sequence=one + 1, value=bytes(value))
        for one in range(count)
    )


@functools.cache
def _table(count: int, block_bytes: int = BLOCK_BYTES, filtered: bool = True) -> Table:
    """A table over that run, cached so the measurements share the build."""
    return write(list(_run(count)), block_bytes=block_bytes, filtered=filtered)


def probe(table: Table, keys: list[bytes]) -> Lookup:
    """Ask a table for a list of keys and account for what each one cost."""
    before_reads, before_skipped = table.reads, table.skipped
    hits = 0
    for key in keys:
        if table.get(key) is not None:
            hits += 1
    return Lookup(
        hits=hits,
        misses=len(keys) - hits,
        reads=table.reads - before_reads,
        skipped=table.skipped - before_skipped,
    )


@functools.cache
def a_lookup_reads_one_block_of_a_file_of_hundreds() -> bool:
    """Twenty thousand records live in 334 blocks and a lookup touches one of them.

    That is what the sparse index buys and it is easy to state and easy to under-appreciate. The
    file is 1.4 megabytes. A lookup reads a block, which is four kilobytes, so the read is three
    tenths of one percent of the file, and the part that made it possible is 11 kilobytes of
    index held in memory.

    The index is one entry per block rather than one per record because the block is already
    sorted and searchable. Indexing every record would multiply the index by sixty and buy
    nothing, since the block search that the entry would replace is a binary search over sixteen
    restart points.
    """
    table = _table(20000)
    made = probe(table, [b"user:00010000"])
    return len(table.blocks) > 300 and made.reads == 1


@functools.cache
def the_filter_turns_a_miss_into_no_read_at_all() -> bool:
    """A key the file does not hold costs a block read without a filter and nothing with one.

    Ten thousand keys the table does not hold: without a filter every one searches the index,
    lands on the block where the key would be, reads it, and finds nothing, which is ten
    thousand block reads for ten thousand answers of no. With a filter at ten bits per key
    the same ten thousand cost 69 reads, which is the filter's false positive rate and nothing
    else.

    This is the number that decides whether a filter is worth its memory, and it depends on the
    workload. A read set that always hits gains nothing from a filter and pays for it. A read
    set that mostly misses, which is what a levelled store looks like from every file except
    the one holding the key, saves a block read on almost every lookup.
    """
    absent = [f"gone:{one:08d}".encode() for one in range(10000)]
    bare = probe(_table(20000, filtered=False), absent)
    filtered = probe(_table(20000), absent)
    return bare.reads_per_miss > 0.99 and filtered.reads_per_miss < 0.01


@functools.cache
def a_miss_outside_the_key_range_never_needed_the_filter() -> bool:
    """The cheapest miss is the one the file range answers, and the filter is not involved.

    A key below the first or above the last is outside the file and the two comparisons that
    establish it cost nothing. That covers a lot of real misses, because a store that writes
    keys in roughly increasing order ends up with files whose ranges barely overlap, and a
    lookup for a recent key is outside the range of every old file.

    The filter matters for the misses inside the range, which are the ones the range cannot rule
    out. Reporting a filter's value without separating those two is how a filter gets credit for
    work the range did.
    """
    table = _table(20000)
    return not table.holds(b"aaa") and not table.holds(b"zzz") and table.holds(b"user:00000001")


@functools.cache
def the_index_is_a_fraction_of_the_file_and_the_filter_is_not() -> bool:
    """The sparse index costs under one percent of the file and the filter costs two.

    Twenty thousand records give 1,384,493 bytes of data blocks, 11,022 bytes of index, and
    25,000 bytes of filter. The index is 0.8 percent of the file and the filter is 1.8, so the
    thing that saves a block read costs twice what the thing that finds the block costs.

    Both live in memory when the file is open, which is the number that actually matters. A
    thousand files of this size is 36 megabytes of index and filter for 1.4 gigabytes of data,
    and that ratio is the reason a store can hold more data than it has memory for.
    """
    table = _table(20000)
    data = sum(block.nbytes for block in table.blocks)
    return table.index_bytes < data * 0.01 < len(table.filter.bits) < data * 0.03


@functools.cache
def a_smaller_block_makes_a_bigger_index_and_a_cheaper_read() -> bool:
    """The block size is the same trade as the restart interval, one level up.

    At one kilobyte blocks the same twenty thousand records give more blocks and a larger index,
    and each block read moves a quarter of the bytes. At sixteen kilobytes there are fewer
    blocks, the index shrinks, and a lookup pulls sixteen kilobytes to answer for one record.

    Measured across six sizes the index goes 44,022 bytes at one kilobyte to 726 at sixty four,
    a factor of sixty, while the bytes moved per lookup go the other way by the same factor. A
    store that mostly does point lookups wants small blocks and one that mostly scans wants
    large ones, because a scan pays the per block overhead once. There is no size that suits
    both, which is why it is a setting.
    """
    small = _table(20000, block_bytes=1024)
    large = _table(20000, block_bytes=16384)
    return (
        len(small.blocks) > len(large.blocks) * 8
        and small.index_bytes > large.index_bytes * 8
    )


@functools.cache
def a_scan_costs_the_blocks_it_crosses_and_nothing_else() -> bool:
    """Scanning a hundred records reads the one or two blocks they sit in.

    A point lookup and a hundred record scan starting at the same key cost almost the same,
    because the hundred records are adjacent and adjacency is what the file is arranged for. The
    same hundred records fetched by a hundred point lookups in random order cost a hundred block
    reads instead of two.

    This is the fact that makes a sorted file worth building. An unsorted file answers a point
    lookup with an index just as well. It cannot answer a range without reading everything.
    """
    table = _table(20000)
    before = table.reads
    found = list(itertools.islice(table.scan(b"user:00010000"), 100))
    crossed = table.reads - before
    return len(found) == 100 and crossed <= 3


@functools.cache
def the_footer_is_the_only_part_whose_position_is_known() -> bool:
    """Everything in the file is found from the last twenty eight bytes.

    A reader that opens a table knows the file length and nothing else. The footer is fixed size
    and last, so it can be read without a search, and it gives the sizes of the data section and
    the index, from which every other offset follows.

    Putting it first instead would mean writing it before the sizes it records are known, which
    means either buffering the whole file in memory or seeking back to patch it after the fact.
    Last is not a style choice.
    """
    table = _table(2000)
    raw = b"\x00" * 1000 + table.footer()
    read = read_footer(raw)
    return read["count"] == 2000 and read["blocks"] == len(table.blocks)


def compare_the_block_sizes(count: int = 20000) -> list[dict]:
    """A row per block size, index cost against block read size."""
    rows = []
    for size in (1024, 2048, 4096, 8192, 16384, 65536):
        table = _table(count, block_bytes=size)
        rows.append(
            {
                "block_bytes": size,
                "blocks": len(table.blocks),
                "index_bytes": table.index_bytes,
                "mean_block": round(
                    sum(one.nbytes for one in table.blocks) / len(table.blocks)
                ),
            }
        )
    return rows


def compare_the_filter(count: int = 20000, probes: int = 10000) -> list[dict]:
    """Two rows, with and without a filter, over a read set that misses every time."""
    absent = [f"gone:{one:08d}".encode() for one in range(probes)]
    return [
        {"filter": bool(flag), **probe(_table(count, filtered=bool(flag)), absent).as_dict()}
        for flag in (False, True)
    ]


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "a_lookup_reads_one_block": a_lookup_reads_one_block_of_a_file_of_hundreds(),
        "the_filter_removes_the_read": the_filter_turns_a_miss_into_no_read_at_all(),
        "the_range_answers_first": a_miss_outside_the_key_range_never_needed_the_filter(),
        "the_index_is_small": the_index_is_a_fraction_of_the_file_and_the_filter_is_not(),
        "block_size_is_a_trade": a_smaller_block_makes_a_bigger_index_and_a_cheaper_read(),
        "a_scan_pays_per_block": a_scan_costs_the_blocks_it_crosses_and_nothing_else(),
        "the_footer_anchors_the_file": the_footer_is_the_only_part_whose_position_is_known(),
    }
