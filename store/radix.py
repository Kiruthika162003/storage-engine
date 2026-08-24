from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

from store.errors import ConfigError
from store.memtable import Memtable
from store.record import Record

# A radix tree memtable, measured against the skiplist it would replace.
#
# The skiplist compares whole keys at every level. A radix tree walks the key's bytes once,
# branching on each byte, so a lookup costs the key's length regardless of how many keys are
# stored, and keys that share prefixes share their path. The trade is memory shape: a radix
# node with two children still carries a branch table, and a store of long random keys builds
# long chains of single-child nodes that a skiplist never pays for. Path compression fixes the
# chains and is exactly the same idea as the block module's prefix sharing, applied to a tree.


@dataclass
class Node:
    """One radix node: a compressed prefix, children by byte, and maybe a record."""

    prefix: bytes = field(default=b"")
    children: dict[int, Node] = field(default_factory=dict)
    record: Record | None = field(default=None)

    def count(self) -> int:
        """Nodes below and including this one."""
        return 1 + sum(child.count() for child in self.children.values())


@dataclass
class Radix:
    """A path compressed radix tree over byte keys."""

    root: Node = field(default_factory=Node)
    records: int = field(default=0)
    steps: int = field(default=0)

    def put(self, record: Record) -> None:
        """Install a record, splitting compressed prefixes where paths diverge."""
        if not record.key:
            raise ConfigError("a key needs at least one byte")
        node = self.root
        rest = record.key
        while True:
            shared = _shared(node.prefix, rest)
            if shared < len(node.prefix):
                split = Node(
                    prefix=node.prefix[shared + 1 :],
                    children=node.children,
                    record=node.record,
                )
                branch = node.prefix[shared]
                node.prefix = node.prefix[:shared]
                node.children = {branch: split}
                node.record = None
            rest = rest[shared:]
            if not rest:
                if node.record is None or record.sequence > node.record.sequence:
                    if node.record is None:
                        self.records += 1
                    node.record = record
                return
            head = rest[0]
            if head not in node.children:
                node.children[head] = Node(prefix=rest[1:], record=record)
                self.records += 1
                return
            node = node.children[head]
            rest = rest[1:]

    def get(self, key: bytes) -> Record | None:
        """One lookup, walking the key's bytes."""
        node = self.root
        rest = key
        while True:
            self.steps += 1
            if not rest.startswith(node.prefix):
                return None
            rest = rest[len(node.prefix) :]
            if not rest:
                return node.record
            head = rest[0]
            if head not in node.children:
                return None
            node = node.children[head]
            rest = rest[1:]

    def scan(self):
        """Every record in key order, which a radix tree gives without sorting."""
        yield from self._walk(self.root)

    def _walk(self, node: Node):
        if node.record is not None:
            yield node.record
        for byte in sorted(node.children):
            yield from self._walk(node.children[byte])

    def nodes(self) -> int:
        """The node count, which is the memory shape."""
        return self.root.count()


def _shared(left: bytes, right: bytes) -> int:
    """Leading bytes in common."""
    limit = min(len(left), len(right))
    at = 0
    while at < limit and left[at] == right[at]:
        at += 1
    return at


@functools.cache
def _prefixed_keys(count: int = 8000, seed: int = 47) -> tuple[bytes, ...]:
    """Keys that share deep prefixes, the radix tree's best case."""
    source = random.Random(seed)
    made = {
        f"tenant:{source.randrange(20):03d}:table:{source.randrange(5)}:row:{source.randrange(10**6):07d}".encode()
        for _ in range(count * 2)
    }
    return tuple(sorted(made)[:count])


@functools.cache
def _random_keys(count: int = 8000, seed: int = 48) -> tuple[bytes, ...]:
    """Keys with no shared structure, the radix tree's worst case."""
    source = random.Random(seed)
    made = set()
    while len(made) < count:
        made.add(source.randbytes(24))
    return tuple(sorted(made))


def _filled_radix(keys) -> Radix:
    """A radix tree over the keys."""
    made = Radix()
    for at, key in enumerate(keys):
        made.put(Record(key=key, sequence=at + 1, value=b"v"))
    return made


def _filled_skiplist(keys) -> Memtable:
    """The incumbent over the same keys."""
    made = Memtable()
    for at, key in enumerate(keys):
        made.put(Record(key=key, sequence=at + 1, value=b"v"))
    return made


@functools.cache
def both_structures_agree_on_contents_and_order() -> bool:
    """The radix tree and the skiplist hold the same records in the same order.

    Same puts, same scan output, record for record, on both key shapes. The agreement is what
    licenses every comparison after it, and it is the differential discipline again: two
    implementations of one contract, diffed.
    """
    for keys in (_prefixed_keys(3000), _random_keys(3000)):
        radix = _filled_radix(keys)
        skiplist = _filled_skiplist(keys)
        if list(radix.scan()) != skiplist.records():
            return False
    return True


def stored_bytes(tree: Radix) -> int:
    """Key bytes the tree actually stores, prefixes once plus a branch byte per node."""
    total = 0
    stack = [tree.root]
    while stack:
        node = stack.pop()
        total += len(node.prefix) + 1
        stack.extend(node.children.values())
    return total


@functools.cache
def prefixes_save_bytes_and_cost_nodes_which_was_half_expected() -> bool:
    """The structured keys store 15 percent of their bytes; the random keys store 96.

    Half the expectation held: sharing pays enormously in bytes, because the tenant prefix
    is stored once per path instead of once per key, 36,285 bytes held for 240,000 written.
    Random keys share nothing and store 95.7 percent of themselves.

    The half that inverted: I claimed random keys would also build more nodes, and they
    build fewer, 1.09 per key against 1.42. Path compression already collapses every
    unshared run, so a random key is one leaf hanging off a shallow branch, nearly optimal,
    while the structured keys pay an interior split node at every position where their
    digits diverge. The folklore that tries win on shared prefixes is a claim about bytes,
    not nodes, and uncompressed tries, where chains of single children make random keys
    catastrophic. With compression the node count inverts and the byte count is the story.
    """
    tidy = _filled_radix(_prefixed_keys())
    messy = _filled_radix(_random_keys())
    tidy_ratio = stored_bytes(tidy) / sum(len(key) for key in _prefixed_keys())
    messy_ratio = stored_bytes(messy) / sum(len(key) for key in _random_keys())
    inverted = messy.nodes() / messy.records < tidy.nodes() / tidy.records
    return tidy_ratio < 0.2 and messy_ratio > 0.9 and inverted


@functools.cache
def lookup_cost_is_the_key_not_the_population() -> bool:
    """Ten times the keys leaves the walk within a step of the same length.

    The radix walk visits one node per consumed chunk of key, so its cost is bounded by the
    key's length however many keys are stored. The skiplist's comparisons grow with the log
    of the population. Both are small numbers; the shapes are what differ, and the flat one
    is the reason tries back real memtables for long structured keys.
    """
    small = _filled_radix(_prefixed_keys(800))
    large = _filled_radix(_prefixed_keys(8000))
    small_before = small.steps
    large_before = large.steps
    for key in _prefixed_keys(800)[:200]:
        small.get(key)
    for key in _prefixed_keys(8000)[:200]:
        large.get(key)
    small_cost = (small.steps - small_before) / 200
    large_cost = (large.steps - large_before) / 200
    return abs(small_cost - large_cost) < 2.0


@functools.cache
def an_overwrite_keeps_the_newest_sequence() -> bool:
    """Writing a key twice keeps one record, the newer one, matching the memtable's rule.

    The radix tree stores one record per key like the memtable, so the overwrite rule has to
    match or the differential test above would be comparing different contracts.
    """
    made = Radix()
    made.put(Record(key=b"k", sequence=1, value=b"old"))
    made.put(Record(key=b"k", sequence=2, value=b"new"))
    stale = Radix()
    stale.put(Record(key=b"k", sequence=2, value=b"new"))
    stale.put(Record(key=b"k", sequence=1, value=b"old"))
    return (
        made.records == 1
        and made.get(b"k").value == b"new"
        and stale.get(b"k").value == b"new"
    )


def compare_the_shapes() -> list[dict]:
    """One row per key shape, nodes per key against steps per lookup."""
    rows = []
    for name, keys in (("prefixed", _prefixed_keys()), ("random", _random_keys())):
        made = _filled_radix(keys)
        before = made.steps
        for key in keys[:500]:
            made.get(key)
        rows.append(
            {
                "keys": name,
                "records": made.records,
                "nodes": made.nodes(),
                "nodes_per_key": round(made.nodes() / made.records, 2),
                "steps_per_lookup": round((made.steps - before) / 500, 2),
            }
        )
    return rows


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "agrees_with_the_skiplist": both_structures_agree_on_contents_and_order(),
        "prefixes_save_bytes_not_nodes": (
            prefixes_save_bytes_and_cost_nodes_which_was_half_expected()
        ),
        "lookups_cost_the_key": lookup_cost_is_the_key_not_the_population(),
        "overwrites_keep_the_newest": an_overwrite_keeps_the_newest_sequence(),
    }
