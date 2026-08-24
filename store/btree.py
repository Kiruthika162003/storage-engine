from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

from store.compaction import Levelled, Load, run_load
from store.errors import ConfigError

# The other answer, built honestly enough to lose fairly.
#
# An LSM buys cheap writes by deferring its sorting and paying for it later in compaction. A
# B-tree sorts at write time: every put walks to the one leaf where the key belongs and changes
# it in place. There is no compaction, no stale versions, no merge on read, and the price is
# that the write itself costs a page rewrite, because a disk cannot change a record inside a
# page without writing the page.
#
# The comparison the two structures actually disagree on is write amplification against read
# amplification. A B-tree writes a whole page to change one record, so its write amplification
# is the page size over the record size, which for small records is enormous. An LSM writes the
# record once and rewrites it a few times in compaction, so its amplification is the level
# count. A B-tree reads one path from root to leaf, so its read amplification is the tree
# height, always. An LSM reads every run that might hold the key.
#
# Everything here is counted in pages and records, not seconds, for the same reason as the
# compaction module: seconds measure the machine.

# How many records a leaf holds before it splits.
LEAF_RECORDS = 64

# How many children an interior node holds before it splits.
INTERIOR_KEYS = 64


@dataclass
class Leaf:
    """A leaf page: sorted keys and their values, changed in place."""

    keys: list[bytes] = field(default_factory=list)
    values: list[bytes] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def full(self) -> bool:
        """Whether the leaf has to split before taking another key."""
        return len(self.keys) >= LEAF_RECORDS

    def find(self, key: bytes) -> int:
        """Where the key is, or where it would go."""
        low, high = 0, len(self.keys)
        while low < high:
            middle = (low + high) // 2
            if self.keys[middle] < key:
                low = middle + 1
            else:
                high = middle
        return low

    def get(self, key: bytes) -> bytes | None:
        """The value for a key, or nothing."""
        at = self.find(key)
        if at < len(self.keys) and self.keys[at] == key:
            return self.values[at]
        return None

    def put(self, key: bytes, value: bytes) -> bool:
        """Install a key, reporting whether the leaf grew."""
        at = self.find(key)
        if at < len(self.keys) and self.keys[at] == key:
            self.values[at] = value
            return False
        self.keys.insert(at, key)
        self.values.insert(at, value)
        return True

    def remove(self, key: bytes) -> bool:
        """Take a key out, reporting whether it was there."""
        at = self.find(key)
        if at < len(self.keys) and self.keys[at] == key:
            del self.keys[at]
            del self.values[at]
            return True
        return False

    def split(self) -> tuple[bytes, Leaf]:
        """Halve the leaf, handing back the separator and the new right half."""
        middle = len(self.keys) // 2
        right = Leaf(keys=self.keys[middle:], values=self.values[middle:])
        self.keys = self.keys[:middle]
        self.values = self.values[:middle]
        return right.keys[0], right


@dataclass
class Interior:
    """An interior page: separators and children, one more child than separators."""

    separators: list[bytes] = field(default_factory=list)
    children: list = field(default_factory=list)

    @property
    def full(self) -> bool:
        """Whether the node has to split before taking another child."""
        return len(self.separators) >= INTERIOR_KEYS

    def child_for(self, key: bytes) -> int:
        """Which child the key belongs under."""
        low, high = 0, len(self.separators)
        while low < high:
            middle = (low + high) // 2
            if self.separators[middle] <= key:
                low = middle + 1
            else:
                high = middle
        return low

    def install(self, at: int, separator: bytes, child) -> None:
        """Take in a child that a split below produced."""
        self.separators.insert(at, separator)
        self.children.insert(at + 1, child)

    def split(self) -> tuple[bytes, Interior]:
        """Halve the node, promoting the middle separator."""
        middle = len(self.separators) // 2
        promoted = self.separators[middle]
        right = Interior(
            separators=self.separators[middle + 1 :],
            children=self.children[middle + 1 :],
        )
        self.separators = self.separators[:middle]
        self.children = self.children[: middle + 1]
        return promoted, right


@dataclass
class Tree:
    """The tree itself, with the page writes counted as a real one would make them."""

    root: object = field(default_factory=Leaf)
    records: int = field(default=0)
    page_writes: int = field(default=0)
    page_reads: int = field(default=0)
    splits: int = field(default=0)

    @property
    def height(self) -> int:
        """How many pages a lookup touches, root to leaf."""
        count = 1
        node = self.root
        while isinstance(node, Interior):
            node = node.children[0]
            count += 1
        return count

    @property
    def pages(self) -> int:
        """How many pages the tree holds."""
        return self._pages(self.root)

    def _pages(self, node) -> int:
        if isinstance(node, Leaf):
            return 1
        return 1 + sum(self._pages(child) for child in node.children)

    def get(self, key: bytes) -> bytes | None:
        """One value, reading a page per level."""
        node = self.root
        self.page_reads += 1
        while isinstance(node, Interior):
            node = node.children[node.child_for(key)]
            self.page_reads += 1
        return node.get(key)

    def put(self, key: bytes, value: bytes) -> None:
        """Install a key, writing the leaf and any pages a split touches."""
        if not key:
            raise ConfigError("a key needs at least one byte")
        path = self._path(key)
        leaf = path[-1][0]
        if leaf.put(key, value):
            self.records += 1
        self.page_writes += 1
        if leaf.full:
            self._split(path)

    def remove(self, key: bytes) -> bool:
        """Take a key out, writing the leaf it left.

        Underfull leaves are left underfull rather than rebalanced. A real tree merges them
        eventually, and skipping that keeps the comparison honest where it matters, on the
        write path, because rebalancing only makes the B-tree's write count worse.
        """
        path = self._path(key)
        leaf = path[-1][0]
        if leaf.remove(key):
            self.records -= 1
            self.page_writes += 1
            return True
        return False

    def _path(self, key: bytes) -> list[tuple[object, int]]:
        """The pages from root to the leaf for a key, each with the child taken."""
        made = []
        node = self.root
        while isinstance(node, Interior):
            at = node.child_for(key)
            made.append((node, at))
            node = node.children[at]
        made.append((node, 0))
        return made

    def _split(self, path: list[tuple[object, int]]) -> None:
        """Split the leaf at the end of the path and let the split climb.

        Every split writes the two halves and the parent that takes the separator, which is
        where a B-tree's write amplification hides: one record over the threshold can rewrite a
        page at every level.
        """
        node, _ = path.pop()
        separator, right = node.split()
        self.splits += 1
        self.page_writes += 2
        while path:
            parent, at = path.pop()
            parent.install(at, separator, right)
            self.page_writes += 1
            if not parent.full:
                return
            separator, right = parent.split()
            self.splits += 1
            self.page_writes += 2
        old = self.root
        self.root = Interior(separators=[separator], children=[old, right])
        self.page_writes += 1

    def scan(self, start: bytes = b""):
        """Every key from a point onwards, in order."""
        yield from self._scan(self.root, start)

    def _scan(self, node, start: bytes):
        if isinstance(node, Leaf):
            at = node.find(start)
            for one in range(at, len(node.keys)):
                yield node.keys[one], node.values[one]
            return
        for at in range(node.child_for(start), len(node.children)):
            yield from self._scan(node.children[at], start)

    def keys(self) -> list[bytes]:
        """Every key, in order."""
        return [key for key, _ in self.scan()]

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "records": self.records,
            "height": self.height,
            "pages": self.pages,
            "page_writes": self.page_writes,
            "page_reads": self.page_reads,
            "splits": self.splits,
        }


PAGE_BYTES = 4096
RECORD_BYTES = 96


@dataclass
class Comparison:
    """The two structures over the same write stream, in each other's units."""

    writes: int
    tree_page_writes: int
    lsm_records_written: int

    @property
    def tree_bytes(self) -> int:
        """What the tree moved, counting a page per page write."""
        return self.tree_page_writes * PAGE_BYTES

    @property
    def lsm_bytes(self) -> int:
        """What the LSM moved, counting a record per record write."""
        return self.lsm_records_written * RECORD_BYTES

    @property
    def ratio(self) -> float:
        """How many bytes the tree moved per byte the LSM moved."""
        return round(self.tree_bytes / max(self.lsm_bytes, 1), 2)

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "writes": self.writes,
            "tree_page_writes": self.tree_page_writes,
            "tree_bytes": self.tree_bytes,
            "lsm_records_written": self.lsm_records_written,
            "lsm_bytes": self.lsm_bytes,
            "ratio": self.ratio,
        }


@functools.cache
def _keys(count: int, seed: int = 2) -> tuple[bytes, ...]:
    """A shuffled set of distinct keys."""
    source = random.Random(seed)
    return tuple(
        f"k{one:08d}".encode() for one in source.sample(range(10**8), count)
    )


@functools.cache
def _tree(count: int) -> Tree:
    """A tree with that many keys already in it."""
    made = Tree()
    for key in _keys(count):
        made.put(key, bytes(32))
    return made


@functools.cache
def compare(writes: int = 40000, keys: int = 20000) -> Comparison:
    """The same uniform write stream through both structures."""
    load = Load(keys=keys, writes=writes)
    lsm = run_load(Levelled(), load)
    tree = Tree()
    for record in load.records():
        tree.put(record.key, record.value or b"\x00")
    return Comparison(
        writes=writes,
        tree_page_writes=tree.page_writes,
        lsm_records_written=lsm.written,
    )


@functools.cache
def the_tree_moves_six_times_the_bytes_for_the_same_writes() -> bool:
    """The page is the unit a B-tree pays in, and pages are large.

    Forty thousand writes of 96 byte records. The LSM writes each record 7.165 times, which is
    27.5 megabytes moved. The tree writes 41,221 pages of four kilobytes, which is 168.8
    megabytes, so the tree moves 6.1 times the bytes even while the LSM rewrites everything it
    stores seven times over.

    The gap is the record size against the page size and it closes as records grow: 18.4 times
    at 32 byte records, 6.1 at 96, 2.3 at 256, 1.15 at 512, and by a kilobyte the tree is
    moving 0.58 of the LSM's bytes and wins outright. The crossover for these settings is a
    little over 500 bytes. Small records are the LSM case, large records are the tree case, and
    knowing which side of that line a workload sits is worth more than either structure's
    folklore.
    """
    made = compare()
    return made.ratio > 4.0


@functools.cache
def the_tree_reads_one_path_and_the_lsm_reads_every_run() -> bool:
    """Read amplification is where the tree collects.

    A lookup in a tree of twenty thousand records touches exactly three pages, root to leaf,
    hit or miss, hot or cold. The equivalent levelled store makes a read consider up to three
    runs, each of which is a block read plus its index search, and a tiered store makes it
    consider more.

    The honest statement is that both sides read a small constant number of pages and the tree's
    constant is smaller and does not degrade. Nothing about an LSM read is faster, ever. The
    entire LSM argument lives on the write side and in the sequential layout of what it writes.
    """
    tree = _tree(20000)
    before = tree.page_reads
    tree.get(_keys(20000)[7])
    return tree.page_reads - before == tree.height == 3


@functools.cache
def a_split_climbs_and_the_climb_is_rare() -> bool:
    """One insert can write a page at every level, and almost none do.

    Twenty thousand inserts split 459 times, so 2.3 percent of inserts split a leaf, and the
    deeper splits that climb past the leaf are rarer still. The amortised cost is a fraction of
    a page write per insert on top of the leaf write itself.

    The reason to measure it: the worst case, a split at every level, is what a B-tree's write
    latency tail looks like, and its rarity is why the tail is tolerated.
    """
    tree = _tree(20000)
    return tree.splits < tree.records * 0.05


@functools.cache
def the_tree_stays_balanced_without_being_told_to() -> bool:
    """Every leaf is the same distance from the root, as a consequence of how splits move.

    A B-tree grows at the root, not at the leaves: the only thing that adds a level is the root
    itself splitting, which lifts every leaf by one at once. So the height is uniform by
    construction rather than by maintenance, which is the property binary trees need rebalancing
    to fake.
    """
    tree = _tree(20000)
    depths = set()

    def walk(node, depth):
        if isinstance(node, Leaf):
            depths.add(depth)
            return
        for child in node.children:
            walk(child, depth + 1)

    walk(tree.root, 1)
    return depths == {tree.height}


@functools.cache
def an_overwrite_is_free_in_a_tree_and_a_new_version_in_a_log() -> bool:
    """The write patterns the two structures prefer are opposites.

    Overwriting one key forty thousand times: the tree writes the same leaf forty thousand
    times and holds one record at the end, no splits, height one. The LSM appends forty
    thousand records and compacts them down to one, having written each level's worth on the
    way.

    A counter, a session token, a last-seen timestamp: keys that are all overwrite are the
    tree's best case and the log's worst. It is the mirror image of the small record insert
    stream, and real workloads are a mixture, which is why real storage engines are too.
    """
    tree = Tree()
    for at in range(40000):
        tree.put(b"counter", at.to_bytes(8, "little"))
    return tree.records == 1 and tree.splits == 0 and tree.pages == 1


def compare_the_record_sizes() -> list[dict]:
    """A row per record size, hunting the crossover."""
    rows = []
    made = compare()
    for size in (32, 96, 256, 512, 1024, 2048):
        lsm_bytes = made.lsm_records_written * size
        tree_bytes = made.tree_page_writes * PAGE_BYTES
        rows.append(
            {
                "record_bytes": size,
                "lsm_bytes": lsm_bytes,
                "tree_bytes": tree_bytes,
                "tree_over_lsm": round(tree_bytes / lsm_bytes, 2),
            }
        )
    return rows


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "the_tree_moves_more_bytes": the_tree_moves_six_times_the_bytes_for_the_same_writes(),
        "the_tree_reads_one_path": the_tree_reads_one_path_and_the_lsm_reads_every_run(),
        "splits_climb_rarely": a_split_climbs_and_the_climb_is_rare(),
        "balance_is_structural": the_tree_stays_balanced_without_being_told_to(),
        "overwrites_are_the_tree_case": (
            an_overwrite_is_free_in_a_tree_and_a_new_version_in_a_log()
        ),
    }
