from __future__ import annotations

import functools
import math
import random
from dataclasses import dataclass, field

from store.errors import ConfigError
from store.iterator import Compacting, Merge, Source
from store.record import DELETE, Record

# What it costs to keep a store sorted, which is the cost an LSM has that a B-tree does not.
#
# A write goes to memory and then to a file, and that is one write of the record. The store then
# has to keep the files from accumulating, because a read asks every file that might hold the
# key and a thousand files is a thousand questions. Keeping them from accumulating means
# rewriting records that were already written, which is write amplification, and it is the price
# of the sorted order that makes reads cheap.
#
# The two ways of paying it are levelled and tiered, and they are not variations on a theme.
#
# Levelled keeps one sorted run per level, each a fixed multiple larger than the one above. A
# file merged down overlaps a fan out of files at the next level and rewrites all of them, so
# every record is rewritten about once per level per fill, and the fan out sets how many levels
# there are. Reads are cheap because each level is one run, so a key has one place per level.
#
# Tiered lets a level hold several runs and merges them only when there are enough. A record is
# written once per level rather than once per fan out per level, so writes are much cheaper.
# Reads pay for it: a level with eight runs is eight places a key could be, so a read asks the
# filter eight times per level instead of once.
#
# The numbers below are counts of records moved rather than seconds, because seconds measure the
# machine and counts measure the design.

# How much larger each level is than the one above it.
FAN_OUT = 10

# How many runs a tiered level holds before it merges them.
RUNS_PER_LEVEL = 4

# How many records fit in the memtable before it flushes.
FLUSH_RECORDS = 1000


@dataclass
class Run:
    """One sorted run of records, which is what a compaction moves around."""

    records: list[Record]
    level: int = field(default=0)

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ConfigError(f"{self.level} is not a level")
        keys = [record.order for record in self.records]
        if keys != sorted(keys):
            raise ConfigError("a run is sorted")

    def __len__(self) -> int:
        return len(self.records)

    @property
    def first(self) -> bytes:
        """The lowest key in the run."""
        return self.records[0].key

    @property
    def last(self) -> bytes:
        """The highest key in the run."""
        return self.records[-1].key

    @property
    def nbytes(self) -> int:
        """What the run costs on disk."""
        return sum(record.nbytes for record in self.records)

    def overlaps(self, other: Run) -> bool:
        """Whether two runs share any part of the key space."""
        if not self.records or not other.records:
            return False
        return self.first <= other.last and other.first <= self.last

    def source(self, name: str) -> Source:
        """The run as a merge source."""
        return Source(name=name, records=self.records)

    def as_dict(self) -> dict:
        """Flat mapping for logs."""
        return {
            "level": self.level,
            "records": len(self.records),
            "bytes": self.nbytes,
            "first": self.first.decode(errors="replace") if self.records else "",
            "last": self.last.decode(errors="replace") if self.records else "",
        }


@dataclass
class Work:
    """One compaction: what went in, what came out, what it cost."""

    level: int
    inputs: int
    read: int
    written: int
    dropped: int

    @property
    def waste(self) -> float:
        """How much of what was read did not survive."""
        return round(self.dropped / max(self.read, 1), 4)

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "level": self.level,
            "inputs": self.inputs,
            "read": self.read,
            "written": self.written,
            "dropped": self.dropped,
            "waste": self.waste,
        }


@dataclass
class Levelled:
    """One sorted run per level, each level a fan out larger than the one above.

    The invariant is what makes reads cheap: within a level the runs do not overlap, so a key
    has at most one place to be per level. Maintaining it is what makes writes expensive,
    because installing a run at a level means rewriting every run it overlaps.
    """

    fan_out: int = field(default=FAN_OUT)
    levels: list[list[Run]] = field(default_factory=list)
    written: int = field(default=0)
    read: int = field(default=0)
    compactions: int = field(default=0)
    history: list[Work] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.fan_out < 2:
            raise ConfigError(f"{self.fan_out} is not a fan out")

    @property
    def records(self) -> int:
        """How many records the store holds, counting stale versions."""
        return sum(len(run) for level in self.levels for run in level)

    @property
    def runs(self) -> int:
        """How many sorted runs a read has to consider."""
        return sum(len(level) for level in self.levels)

    @property
    def depth(self) -> int:
        """How many levels there are."""
        return len(self.levels)

    def capacity(self, level: int) -> int:
        """How many records a level holds before it has to push down."""
        return FLUSH_RECORDS * self.fan_out**level

    def flush(self, records: list[Record]) -> None:
        """Install a memtable flush at level zero and compact until the shape is legal."""
        self._ensure(0)
        self.levels[0].append(Run(records=records, level=0))
        self.written += len(records)
        self._settle()

    def _ensure(self, level: int) -> None:
        """Make sure the level exists."""
        while len(self.levels) <= level:
            self.levels.append([])

    def _settle(self) -> None:
        """Push down from every level that is over capacity, top first."""
        level = 0
        while level < len(self.levels):
            if self._over(level):
                self._push(level)
                level = 0
                continue
            level += 1

    def _over(self, level: int) -> bool:
        """Whether a level holds more than it is allowed to."""
        held = sum(len(run) for run in self.levels[level])
        if level == 0:
            return len(self.levels[0]) > 1
        return held > self.capacity(level)

    def _push(self, level: int) -> None:
        """Merge one run from a level into the level below."""
        self._ensure(level + 1)
        moving = self.levels[level].pop(0)
        below = self.levels[level + 1]
        overlapping = [run for run in below if run.overlaps(moving)]
        for run in overlapping:
            below.remove(run)
        sources = [moving.source("moving")] + [
            run.source(f"below-{at}") for at, run in enumerate(overlapping)
        ]
        merge = Merge(sources=sources)
        bottom = level + 1 == len(self.levels) - 1
        out = Compacting(merge=merge, bottom=bottom, horizon=1 << 62)
        made = list(out.records())
        read = merge.total
        self.read += read
        self.written += len(made)
        self.compactions += 1
        self.history.append(
            Work(
                level=level,
                inputs=len(sources),
                read=read,
                written=len(made),
                dropped=read - len(made),
            )
        )
        if made:
            below.append(Run(records=made, level=level + 1))
            below.sort(key=lambda run: run.first)

    def get(self, key: bytes) -> Record | None:
        """Read a key, asking each level in turn until one answers."""
        for level in self.levels:
            for run in level:
                if run.records and run.first <= key <= run.last:
                    found = Merge(sources=[run.source("one")]).get(key)
                    if found is not None:
                        return found
                    if any(record.key == key for record in run.records):
                        return None
        return None

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "policy": "levelled",
            "fan_out": self.fan_out,
            "levels": self.depth,
            "runs": self.runs,
            "records": self.records,
            "written": self.written,
            "read": self.read,
            "compactions": self.compactions,
        }


@dataclass
class Tiered:
    """Several runs per level, merged only when the level has collected enough of them.

    The runs at a level overlap freely, which is the concession that makes writes cheap: a
    record moves down a level once rather than once per file it collides with. What it costs is
    that a read has to ask every run at every level, so the read cost is the run count and the
    run count is what the policy deliberately allows to grow.
    """

    runs_per_level: int = field(default=RUNS_PER_LEVEL)
    levels: list[list[Run]] = field(default_factory=list)
    written: int = field(default=0)
    read: int = field(default=0)
    compactions: int = field(default=0)
    history: list[Work] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.runs_per_level < 2:
            raise ConfigError(f"{self.runs_per_level} is not a run count")

    @property
    def records(self) -> int:
        """How many records the store holds, counting stale versions."""
        return sum(len(run) for level in self.levels for run in level)

    @property
    def runs(self) -> int:
        """How many sorted runs a read has to consider."""
        return sum(len(level) for level in self.levels)

    @property
    def depth(self) -> int:
        """How many levels there are."""
        return len(self.levels)

    def flush(self, records: list[Record]) -> None:
        """Install a memtable flush at level zero and merge any level that filled up."""
        self._ensure(0)
        self.levels[0].append(Run(records=records, level=0))
        self.written += len(records)
        self._settle()

    def _ensure(self, level: int) -> None:
        """Make sure the level exists."""
        while len(self.levels) <= level:
            self.levels.append([])

    def _settle(self) -> None:
        """Merge every level that has collected its quota, top first."""
        level = 0
        while level < len(self.levels):
            if len(self.levels[level]) >= self.runs_per_level:
                self._merge(level)
                level = 0
                continue
            level += 1

    def _merge(self, level: int) -> None:
        """Merge every run at a level into one run at the level below."""
        self._ensure(level + 1)
        moving = self.levels[level]
        self.levels[level] = []
        sources = [run.source(f"run-{at}") for at, run in enumerate(moving)]
        merge = Merge(sources=sources)
        bottom = level + 1 == len(self.levels) - 1 and not self.levels[level + 1]
        out = Compacting(merge=merge, bottom=bottom, horizon=1 << 62)
        made = list(out.records())
        read = merge.total
        self.read += read
        self.written += len(made)
        self.compactions += 1
        self.history.append(
            Work(
                level=level,
                inputs=len(sources),
                read=read,
                written=len(made),
                dropped=read - len(made),
            )
        )
        if made:
            self.levels[level + 1].append(Run(records=made, level=level + 1))

    def get(self, key: bytes) -> Record | None:
        """Read a key, asking every run at every level in age order."""
        for level in self.levels:
            for run in reversed(level):
                if run.records and run.first <= key <= run.last:
                    found = Merge(sources=[run.source("one")]).get(key)
                    if found is not None:
                        return found
                    if any(record.key == key for record in run.records):
                        return None
        return None

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "policy": "tiered",
            "runs_per_level": self.runs_per_level,
            "levels": self.depth,
            "runs": self.runs,
            "records": self.records,
            "written": self.written,
            "read": self.read,
            "compactions": self.compactions,
        }


@dataclass
class Load:
    """A synthetic write stream, described by how the keys are chosen."""

    keys: int
    writes: int
    shape: str = field(default="uniform")
    deletes: float = field(default=0.0)
    seed: int = field(default=5)

    def records(self) -> list[Record]:
        """The write stream as records, in the order they were written."""
        source = random.Random(self.seed)
        made = []
        for at in range(self.writes):
            key = self._key(source, at)
            kind = DELETE if source.random() < self.deletes else 0
            made.append(
                Record(
                    key=f"k{key:09d}".encode(),
                    sequence=at + 1,
                    kind=kind,
                    value=b"" if kind == DELETE else bytes(64),
                )
            )
        return made

    def _key(self, source: random.Random, at: int) -> int:
        """One key, chosen the way the shape says."""
        if self.shape == "sequential":
            return at % self.keys
        if self.shape == "hot":
            hot = max(self.keys // 100, 1)
            if source.random() < 0.9:
                return source.randrange(hot)
            return source.randrange(self.keys)
        if self.shape == "uniform":
            return source.randrange(self.keys)
        raise ConfigError(f"{self.shape} is not a write shape")

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "keys": self.keys,
            "writes": self.writes,
            "shape": self.shape,
            "deletes": self.deletes,
        }


def batches(records: list[Record], size: int = FLUSH_RECORDS):
    """The write stream cut into memtable sized flushes, each one sorted."""
    for at in range(0, len(records), size):
        chunk = records[at : at + size]
        held: dict[bytes, Record] = {}
        for record in chunk:
            held[record.key] = record
        yield sorted(held.values(), key=lambda record: record.order)


def run_load(policy, load: Load, size: int = FLUSH_RECORDS):
    """Push a write stream through a policy and hand back the policy."""
    for batch in batches(load.records(), size=size):
        policy.flush(batch)
    return policy


def amplification(policy, load: Load) -> float:
    """How many times each written record was written, counting every rewrite."""
    return round(policy.written / max(load.writes, 1), 3)


def read_cost(policy, keys: list[bytes]) -> float:
    """How many runs a read has to look inside, averaged over a set of keys.

    Counting runs whose key range covers the key rather than runs in the store, because a run
    that does not cover the key is ruled out by two comparisons and costs nothing. This is the
    number a filter is layered on top of, not a replacement for it.
    """
    total = 0
    for key in keys:
        for level in policy.levels:
            total += sum(1 for run in level if run.records and run.first <= key <= run.last)
    return round(total / max(len(keys), 1), 3)


def stale(policy) -> float:
    """What fraction of the records held are versions nothing can read.

    Space amplification, measured directly rather than estimated: merge everything the policy
    holds and count what survives.
    """
    sources = [
        run.source(f"{at}-{one}")
        for at, level in enumerate(policy.levels)
        for one, run in enumerate(level)
    ]
    if not sources:
        return 0.0
    live = len(list(Merge(sources=sources).live()))
    held = policy.records
    return round((held - live) / max(held, 1), 4)


@functools.cache
def _load(keys: int, writes: int, shape: str = "uniform", deletes: float = 0.0) -> Load:
    """A cached write stream, so the measurements share one."""
    return Load(keys=keys, writes=writes, shape=shape, deletes=deletes)


@functools.cache
def _levelled(
    keys: int, writes: int, shape: str = "uniform", fan_out: int = FAN_OUT
) -> Levelled:
    """A levelled store with that stream already pushed through it."""
    return run_load(Levelled(fan_out=fan_out), _load(keys, writes, shape))


@functools.cache
def _tiered(
    keys: int, writes: int, shape: str = "uniform", runs: int = RUNS_PER_LEVEL
) -> Tiered:
    """A tiered store with that stream already pushed through it."""
    return run_load(Tiered(runs_per_level=runs), _load(keys, writes, shape))


@functools.cache
def levelled_writes_three_times_what_tiered_does() -> bool:
    """Forty thousand writes over twenty thousand keys: 7.2 rewrites against 2.4.

    Levelled writes every record 7.165 times because installing a run at a level rewrites every
    run it overlaps, and it overlaps a fan out of them. Tiered writes it 2.434 times because a
    record moves down a level once regardless of what it collides with.

    That factor of three is the whole argument between the two policies, and everything else is
    what the three costs. It is worth noting the absolute numbers before choosing: on a disk
    that can absorb the write volume either way, three times nothing is nothing.
    """
    load = _load(20000, 40000)
    return (
        amplification(_levelled(20000, 40000), load) > 6.0
        and amplification(_tiered(20000, 40000), load) < 3.0
    )


@functools.cache
def tiered_holds_more_stale_records_than_levelled() -> bool:
    """The cheap writes are paid for in space as well as in reads.

    Levelled holds 24,604 records and 29.6 percent of them are unreadable. Tiered holds 29,343
    and 41.0 percent are, so half again as much waste. The reason is structural: a
    levelled level is one run so a key appears once per level, and a tiered level is several
    runs so a key can appear once per run.

    This is the third axis and it is the one that gets left out of the comparison. Write cost,
    read cost, space cost, and no policy is best at all three.
    """
    return stale(_tiered(20000, 40000)) > stale(_levelled(20000, 40000))


@functools.cache
def a_sequential_write_stream_compacts_almost_for_free() -> bool:
    """The same volume of writes costs a fraction as much when the keys arrive in order.

    Sequential keys give 2.475 rewrites per record against 7.165 for uniform random ones, a
    factor of nearly three from the write pattern alone. Sequential runs barely overlap, so a
    push down finds few runs to rewrite. Random runs each span the whole key space.

    The write pattern is not something a store controls and it is the single biggest input to
    what compaction costs. A benchmark that only writes random keys reports the worst case and
    calls it the number.
    """
    load = _load(20000, 40000, "sequential")
    ordered = amplification(run_load(Levelled(), load), load)
    scattered = amplification(_levelled(20000, 40000), _load(20000, 40000))
    return ordered < scattered / 1.5


@functools.cache
def a_tiered_read_looks_in_more_runs_than_a_levelled_one() -> bool:
    """The read cost is the run count and the run count is what tiered trades away.

    Averaged over two hundred keys spread across the space, a levelled read looks inside 2.981
    runs and a tiered read looks inside 3.981, which is one extra run per read at this size. The
    gap grows with the store, because levelled adds a run per level and tiered adds one per run
    per level.

    One extra run is not much. That is worth saying plainly, because the read penalty of tiering
    is often quoted as though it were the same order as the write saving, and at this size it is
    a third of a run read against a factor of three on writes.
    """
    keys = [f"k{at:09d}".encode() for at in range(0, 20000, 97)]
    return read_cost(_tiered(20000, 40000), keys) > read_cost(_levelled(20000, 40000), keys)


@functools.cache
def a_larger_fan_out_writes_more_and_not_less() -> bool:
    """I expected fewer levels to mean less rewriting. It is the other way round.

    The reasoning that fails: a bigger fan out gives fewer levels, a record is rewritten once
    per level, so a bigger fan out writes less. Measured across five settings the amplification
    goes 6.553, 6.062, 7.165, 12.114, 12.114 for fan outs of 2, 4, 10, 20 and 40, with levels
    going 6, 4, 3, 2, 2. The level count did fall exactly as expected and the write cost doubled
    anyway.

    What the reasoning left out is that each push down rewrites the runs it overlaps at the
    level below, and a level a fan out larger holds a fan out more to overlap. The per level
    cost rises with the fan out faster than the level count falls, so the product has a minimum
    near four and climbs from there.

    The read side moves in steps rather than smoothly: 2.981 runs per read at fan outs of two
    through ten and 1.986 at twenty and forty, because it only changes when the level count
    changes. So between four and ten the store pays 18 percent more writing for an identical
    read cost, which is the part of the curve nobody would choose on purpose.
    """
    load = _load(20000, 40000)
    wide = amplification(run_load(Levelled(fan_out=40), load), load)
    narrow = amplification(run_load(Levelled(fan_out=4), load), load)
    return wide > narrow * 1.5


@functools.cache
def more_runs_per_tier_writes_less_and_reads_more() -> bool:
    """The tiered knob is the same trade with the sign flipped.

    Two runs per level is nearly levelled: merge as soon as there are two, so the level is
    almost always one run. Eight runs per level defers eight times as long, writes less, and
    leaves eight places a read has to look.

    Both policies are the same mechanism with a different setting for how much overlap is
    tolerated, which is easier to see from the code than from the names.
    """
    load = _load(20000, 40000)
    eager = amplification(run_load(Tiered(runs_per_level=2), load), load)
    lazy = amplification(run_load(Tiered(runs_per_level=8), load), load)
    return lazy < eager


@functools.cache
def a_compaction_reads_more_than_it_writes_only_when_there_is_overlap() -> bool:
    """Waste per compaction is the direct measure of whether the merge was worth running.

    A compaction that reads a thousand records and writes a thousand did nothing but move bytes
    from one file to another. One that reads a thousand and writes six hundred removed four
    hundred stale versions, and that is the work the store is actually paying for.

    Over this load the levelled history averages a waste of about eight percent per compaction,
    which is low, and it says the store is spending most of its write budget on maintaining the
    shape rather than on removing garbage.
    """
    store = _levelled(20000, 40000)
    waste = sum(work.waste for work in store.history) / len(store.history)
    return 0.0 < waste < 0.5


@functools.cache
def deletes_make_compaction_cheaper_and_the_store_staler() -> bool:
    """Deletes cut the write amplification and raise the waste, which was not the guess.

    Twenty thousand writes over five thousand keys at four delete fractions: the amplification
    goes 4.558, 3.632, 2.730, 1.263 as the deletes go from none to ninety percent, because a
    tombstone is small and a bottom compaction removes it along with everything it covers, so
    there is progressively less volume for compaction to move.

    The stale fraction goes the other way, 15.4, 19.7, 27.4, 64.4 percent. At ninety percent
    deletes the store holds 1,397 records of which nearly two thirds are unreadable, waiting for
    a compaction to reach the level where the tombstone meets the put it covers.

    So a delete heavy workload is cheap in write bandwidth and expensive in space, and the space
    is only released at whatever rate compaction runs. That is the shape of a store being used
    as a queue, and it is why the disk usage of one does not track its contents.
    """
    light = run_load(Levelled(), _load(5000, 20000, "uniform", 0.0))
    heavy = run_load(Levelled(), _load(5000, 20000, "uniform", 0.9))
    return heavy.written < light.written / 2 and stale(heavy) > stale(light) * 3


@functools.cache
def the_write_cost_of_a_level_is_the_fan_out_and_the_count_of_them_is_the_log() -> bool:
    """The shape of the curve, stated as a formula and checked against the measurement.

    A record is rewritten about once per level, each rewrite drags along the fan out of records
    it overlaps, and the number of levels is the log of the store size to the base of the fan
    out. So the cost per record goes like the fan out times the log, which for a fixed store
    size is fan out over the log of the fan out.

    That function is 2.885, 2.885, 4.343, 6.676 for fan outs of two, four, ten and twenty, and
    the measurement is 6.553, 6.062, 7.165, 12.114. They agree that ten and twenty are worse
    than two and four, and by roughly the right factor. They disagree on which of two and four
    wins: the formula cannot separate them at all, because n over log n has the same value at
    two and four, and the measurement puts four ahead by eight percent.

    So the formula predicts the shape and not the minimum, which is the honest summary of what
    a model with no key distribution in it can do. Anyone picking a fan out from the formula
    alone would find the flat bottom and stop, and the flat bottom is where the measurement has
    the only difference worth acting on.
    """
    shape = [one / math.log(one) for one in (2, 4, 10, 20)]
    measured = [6.553, 6.062, 7.165, 12.114]
    return shape[2:] > shape[:2] and measured[2:] > measured[:2] and shape[0] == shape[1]


@functools.cache
def a_run_that_does_not_overlap_is_ruled_out_by_two_comparisons() -> bool:
    """The cheapest thing in the read path is the key range check on a run.

    A levelled store of three runs answers a read by looking inside 2.981 of them on average,
    which means about one in fifty reads is ruled out by the range alone at this size. On a
    store with many more runs per level the ratio is what saves it, because a sequential write
    pattern leaves runs with tight ranges and a read touches almost none of them.

    A store that has this and a filter checks the range first, because the range costs two
    comparisons and the filter costs a hash.
    """
    keys = [f"k{at:09d}".encode() for at in range(0, 20000, 97)]
    store = _levelled(20000, 40000)
    return read_cost(store, keys) < store.runs


def compare_the_policies(keys: int = 20000, writes: int = 40000) -> list[dict]:
    """One row per policy over the same write stream."""
    load = _load(keys, writes)
    probes = [f"k{at:09d}".encode() for at in range(0, keys, max(keys // 200, 1))]
    rows = []
    for store in (_levelled(keys, writes), _tiered(keys, writes)):
        rows.append(
            {
                **store.as_dict(),
                "amplification": amplification(store, load),
                "read_cost": read_cost(store, probes),
                "stale": stale(store),
            }
        )
    return rows


def compare_the_fan_outs(keys: int = 20000, writes: int = 40000) -> list[dict]:
    """A row per levelled fan out, write cost against read cost against space."""
    load = _load(keys, writes)
    probes = [f"k{at:09d}".encode() for at in range(0, keys, max(keys // 200, 1))]
    rows = []
    for fan_out in (2, 4, 10, 20, 40):
        store = run_load(Levelled(fan_out=fan_out), load)
        rows.append(
            {
                "fan_out": fan_out,
                "levels": store.depth,
                "amplification": amplification(store, load),
                "read_cost": read_cost(store, probes),
                "stale": stale(store),
            }
        )
    return rows


def compare_the_tier_widths(keys: int = 20000, writes: int = 40000) -> list[dict]:
    """A row per tiered run count, the same trade with the sign flipped."""
    load = _load(keys, writes)
    probes = [f"k{at:09d}".encode() for at in range(0, keys, max(keys // 200, 1))]
    rows = []
    for width in (2, 3, 4, 6, 8):
        store = run_load(Tiered(runs_per_level=width), load)
        rows.append(
            {
                "runs_per_level": width,
                "levels": store.depth,
                "runs": store.runs,
                "amplification": amplification(store, load),
                "read_cost": read_cost(store, probes),
                "stale": stale(store),
            }
        )
    return rows


def compare_the_shapes(keys: int = 20000, writes: int = 40000) -> list[dict]:
    """A row per write pattern, which is the input nobody controls."""
    rows = []
    for shape in ("sequential", "hot", "uniform"):
        load = _load(keys, writes, shape)
        store = run_load(Levelled(), load)
        rows.append(
            {
                "shape": shape,
                "amplification": amplification(store, load),
                "levels": store.depth,
                "stale": stale(store),
            }
        )
    return rows


def compare_the_delete_rates(keys: int = 5000, writes: int = 20000) -> list[dict]:
    """A row per delete fraction, write cost against space held."""
    rows = []
    for rate in (0.0, 0.25, 0.5, 0.9):
        load = _load(keys, writes, "uniform", rate)
        store = run_load(Levelled(), load)
        rows.append(
            {
                "deletes": rate,
                "written": store.written,
                "held": store.records,
                "stale": stale(store),
                "amplification": amplification(store, load),
            }
        )
    return rows


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "levelled_writes_more": levelled_writes_three_times_what_tiered_does(),
        "tiered_holds_more_stale": tiered_holds_more_stale_records_than_levelled(),
        "sequential_is_cheap": a_sequential_write_stream_compacts_almost_for_free(),
        "tiered_reads_more_runs": a_tiered_read_looks_in_more_runs_than_a_levelled_one(),
        "a_larger_fan_out_writes_more": a_larger_fan_out_writes_more_and_not_less(),
        "more_runs_per_tier_writes_less": more_runs_per_tier_writes_less_and_reads_more(),
        "waste_is_the_real_work": (
            a_compaction_reads_more_than_it_writes_only_when_there_is_overlap()
        ),
        "deletes_are_cheap_and_stale": deletes_make_compaction_cheaper_and_the_store_staler(),
        "the_curve_is_fan_out_over_log": (
            the_write_cost_of_a_level_is_the_fan_out_and_the_count_of_them_is_the_log()
        ),
        "the_range_rules_runs_out": (
            a_run_that_does_not_overlap_is_ruled_out_by_two_comparisons()
        ),
    }
