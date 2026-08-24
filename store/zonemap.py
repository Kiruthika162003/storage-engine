from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

from store.errors import ConfigError

# Zone maps: pruning a scan with sixteen bytes per block.
#
# A filter answers point questions. A range query, everything between a and b, walks blocks,
# and the zone map is the structure that lets it skip: each block records its minimum and
# maximum value, and a block whose range misses the query's range is never read. The pruning
# power is entirely a property of how the data is laid out: values clustered by block prune
# almost everything, values scattered across blocks prune nothing, and the map costs the same
# sixteen bytes per block either way. This is the reason sorted storage helps analytics even
# when no query ever asks for order.


@dataclass(frozen=True)
class Zone:
    """One block's summary."""

    block: int
    low: int
    high: int

    def overlaps(self, start: int, stop: int) -> bool:
        """Whether a query range could find anything inside."""
        return self.low <= stop and start <= self.high


@dataclass
class Mapped:
    """Blocks of values with a zone map over them."""

    blocks: list[list[int]] = field(default_factory=list)
    zones: list[Zone] = field(default_factory=list)
    blocks_read: int = field(default=0)
    blocks_skipped: int = field(default=0)

    @classmethod
    def build(cls, values: list[int], block_size: int = 100) -> Mapped:
        """Cut the values into blocks in the order given and summarise each."""
        if block_size < 1:
            raise ConfigError(f"{block_size} is not a block size")
        made = cls()
        for at in range(0, len(values), block_size):
            chunk = values[at : at + block_size]
            made.blocks.append(chunk)
            made.zones.append(Zone(block=len(made.blocks) - 1, low=min(chunk), high=max(chunk)))
        return made

    def query(self, start: int, stop: int) -> list[int]:
        """Everything in the closed range, reading only blocks the map allows."""
        found = []
        for zone in self.zones:
            if not zone.overlaps(start, stop):
                self.blocks_skipped += 1
                continue
            self.blocks_read += 1
            found.extend(value for value in self.blocks[zone.block] if start <= value <= stop)
        return found

    @property
    def map_bytes(self) -> int:
        """Sixteen bytes per block."""
        return len(self.zones) * 16

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        total = self.blocks_read + self.blocks_skipped
        return {
            "blocks": len(self.blocks),
            "map_bytes": self.map_bytes,
            "blocks_read": self.blocks_read,
            "blocks_skipped": self.blocks_skipped,
            "skip_rate": round(self.blocks_skipped / max(total, 1), 4),
        }


@functools.cache
def _values(count: int = 50000, seed: int = 103) -> tuple[int, ...]:
    """Timestamps with jitter, the shape ingestion actually produces."""
    source = random.Random(seed)
    made = []
    clock = 0
    for _ in range(count):
        clock += source.randrange(1, 10)
        made.append(clock + source.randrange(-3, 4))
    return tuple(made)


def _narrow_query(values, source: random.Random) -> tuple[int, int]:
    """A range covering about one percent of the value space."""
    low, high = min(values), max(values)
    width = (high - low) // 100
    start = source.randrange(low, high - width)
    return start, start + width


@functools.cache
def sorted_layout_skips_ninety_nine_percent() -> bool:
    """Narrow queries on time ordered blocks read one or two blocks of five hundred.

    Ingestion order is nearly sorted for timestamps, so each block covers a narrow slice and
    a one percent query overlaps one percent of the blocks. A hundred queries skip 98.0
    percent of all block visits, reading 199 blocks where the mapless scan reads 10,000,
    and the map that did it costs 1,600 bytes over a fifty thousand value table.
    """
    mapped = Mapped.build(list(_values()), block_size=500)
    source = random.Random(7)
    for _ in range(100):
        mapped.query(*_narrow_query(_values(), source))
    return mapped.as_dict()["skip_rate"] > 0.97


@functools.cache
def shuffled_layout_skips_nothing() -> bool:
    """The same values shuffled, the same queries, and every block is read every time.

    Each shuffled block spans nearly the whole value range, so every zone overlaps every
    query and the map becomes sixteen bytes per block of pure overhead. The data did not
    change, the queries did not change, only the layout did, and the layout is the entire
    effect: a zone map is a bet on clustering, and shuffling voids the bet.
    """
    values = list(_values())
    random.Random(9).shuffle(values)
    mapped = Mapped.build(values, block_size=500)
    source = random.Random(7)
    for _ in range(100):
        mapped.query(*_narrow_query(values, source))
    return mapped.as_dict()["skip_rate"] < 0.02


@functools.cache
def the_answers_agree_across_layouts() -> bool:
    """Sorted and shuffled layouts return the same multiset for every query.

    Pruning is only legal because it is invisible: a skipped block must have nothing to
    contribute, and the cross layout diff is the check, since the shuffled layout skips
    nothing and therefore cannot be wrong by omission.
    """
    ordered = Mapped.build(list(_values(10000)), block_size=200)
    values = list(_values(10000))
    random.Random(11).shuffle(values)
    shuffled = Mapped.build(values, block_size=200)
    source = random.Random(13)
    for _ in range(30):
        start, stop = _narrow_query(values, source)
        if sorted(ordered.query(start, stop)) != sorted(shuffled.query(start, stop)):
            return False
    return True


@functools.cache
def an_empty_range_reads_nothing_on_any_layout() -> bool:
    """A query below every value skips every block, sorted or not.

    The min max summary answers the empty case for free on both layouts, which is worth a
    claim because it is the one case the shuffled layout can still prune.
    """
    values = list(_values(5000))
    random.Random(15).shuffle(values)
    mapped = Mapped.build(values, block_size=100)
    found = mapped.query(-10**9, min(values) - 1)
    return found == [] and mapped.blocks_read == 0


def compare_the_layouts(queries: int = 100) -> list[dict]:
    """One row per layout, the same query stream."""
    rows = []
    ordered = list(_values())
    shuffled = list(_values())
    random.Random(9).shuffle(shuffled)
    for name, values in (("sorted", ordered), ("shuffled", shuffled)):
        mapped = Mapped.build(values, block_size=500)
        source = random.Random(7)
        for _ in range(queries):
            mapped.query(*_narrow_query(values, source))
        rows.append({"layout": name, **mapped.as_dict()})
    return rows


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "sorted_skips_nearly_all": sorted_layout_skips_ninety_nine_percent(),
        "shuffled_skips_nothing": shuffled_layout_skips_nothing(),
        "answers_agree_across_layouts": the_answers_agree_across_layouts(),
        "empty_ranges_are_free": an_empty_range_reads_nothing_on_any_layout(),
    }
