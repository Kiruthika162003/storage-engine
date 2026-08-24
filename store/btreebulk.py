from __future__ import annotations

import functools
import statistics
from dataclasses import dataclass

from store.btree import INTERIOR_KEYS, LEAF_RECORDS, Interior, Leaf, Tree
from store.errors import ConfigError

# Building a B-tree from sorted input bottom-up, against inserting it.
#
# The bulkload module gave the LSM its side door; the B-tree has one too, and its shape is
# different. Inserting sorted keys one by one splits every leaf at the moment it fills, and
# the split leaves both halves half full forever, because sorted input never comes back to
# refill them. The bottom-up build packs leaves to a chosen fill, stacks interior levels
# over them, and hands back a tree that is denser, shallower at the margin, and built
# without a single split. The fill knob is the interesting part: pack to one hundred
# percent and the first later insert splits; the customary ninety percent is priced here.

LEAF_FILL = 0.9
INTERIOR_FILL = 0.9


@dataclass(frozen=True)
class Built:
    """What the build produced, next to what inserting produced."""

    method: str
    pages: int
    height: int
    page_writes: int
    splits: int
    mean_leaf_fill: float

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "method": self.method,
            "pages": self.pages,
            "height": self.height,
            "page_writes": self.page_writes,
            "splits": self.splits,
            "mean_leaf_fill": self.mean_leaf_fill,
        }


def bulk_build(pairs: list[tuple[bytes, bytes]], fill: float = LEAF_FILL) -> Tree:
    """A tree from sorted pairs, packed bottom-up, no splits ever."""
    if not pairs:
        raise ConfigError("an empty build builds nothing")
    keys = [key for key, _ in pairs]
    if keys != sorted(keys) or len(set(keys)) != len(keys):
        raise ConfigError("a bulk build requires sorted distinct keys")
    if not 0.1 <= fill <= 1.0:
        raise ConfigError(f"{fill} is not a fill fraction")

    per_leaf = max(int(LEAF_RECORDS * fill), 1)
    leaves = []
    for at in range(0, len(pairs), per_leaf):
        chunk = pairs[at : at + per_leaf]
        leaves.append(
            Leaf(keys=[key for key, _ in chunk], values=[value for _, value in chunk])
        )
    tree = Tree()
    tree.page_writes = len(leaves)
    tree.records = len(pairs)
    level: list = leaves
    firsts = [leaf.keys[0] for leaf in leaves]

    per_node = max(int(INTERIOR_KEYS * INTERIOR_FILL), 2)
    while len(level) > 1:
        parents = []
        parent_firsts = []
        for at in range(0, len(level), per_node):
            children = level[at : at + per_node]
            child_firsts = firsts[at : at + per_node]
            node = Interior(separators=child_firsts[1:], children=children)
            parents.append(node)
            parent_firsts.append(child_firsts[0])
            tree.page_writes += 1
        level = parents
        firsts = parent_firsts
    tree.root = level[0]
    return tree


def leaf_fills(tree: Tree) -> list[float]:
    """Every leaf's occupancy."""

    fills = []

    def walk(node) -> None:
        if isinstance(node, Leaf):
            fills.append(len(node.keys) / LEAF_RECORDS)
            return
        for child in node.children:
            walk(child)

    walk(tree.root)
    return fills


@functools.cache
def _pairs(count: int = 20000) -> tuple[tuple[bytes, bytes], ...]:
    """Sorted distinct pairs."""
    return tuple(
        (f"k{at:07d}".encode(), at.to_bytes(4, "big")) for at in range(count)
    )


def _inserted(count: int = 20000) -> Tree:
    """The same pairs through the ordinary insert path, in sorted order."""
    tree = Tree()
    for key, value in _pairs(count):
        tree.put(key, value)
    return tree


def measure(count: int = 20000) -> list[Built]:
    """Both construction methods, one row each."""
    made = []
    built = bulk_build(list(_pairs(count)))
    for method, tree in (("insert", _inserted(count)), ("bulk", built)):
        fills = leaf_fills(tree)
        made.append(
            Built(
                method=method,
                pages=tree.pages,
                height=tree.height,
                page_writes=tree.page_writes,
                splits=tree.splits,
                mean_leaf_fill=round(statistics.mean(fills), 3),
            )
        )
    return made


@functools.cache
def both_trees_answer_identically() -> bool:
    """Every key reads the same through both constructions, and both scan in order.

    The differential license: construction is an implementation detail, and any divergence
    is a bug in the packing, most likely in the separator arithmetic, which is exactly where
    bottom-up builds go wrong.
    """
    inserted = _inserted(8000)
    built = bulk_build(list(_pairs(8000)))
    if any(built.get(key) != inserted.get(key) for key, _ in _pairs(8000)):
        return False
    return built.keys() == inserted.keys()


@functools.cache
def sorted_inserts_leave_half_empty_leaves() -> bool:
    """The insert path fills leaves to 50 percent; the bulk build to 89.

    Sorted input splits every leaf at the moment it fills and never returns to the left
    half, so the tree ends at the worst steady state fill. The bulk build packs to its knob.
    The same records in 644 pages one way and 359 the other: the space cost of building a
    tree by insertion is real and permanent, which is why every database ships a loader.
    """
    rows = {row.method: row for row in measure()}
    return rows["insert"].mean_leaf_fill < 0.55 and rows["bulk"].mean_leaf_fill > 0.85


@functools.cache
def the_bulk_build_never_splits_and_writes_once() -> bool:
    """Zero splits and one page write per page, against 21,923 writes and 641 splits.

    The insert path writes a page per record plus the splits; the build writes each page
    exactly once, which is the same one-touch property the external sort's run formation
    had, and for the same reason: sorted input means every page's contents are known the
    moment it is opened.
    """
    rows = {row.method: row for row in measure()}
    return (
        rows["bulk"].splits == 0
        and rows["bulk"].page_writes == rows["bulk"].pages
        and rows["insert"].page_writes > rows["bulk"].page_writes * 20
    )


@functools.cache
def a_full_pack_splits_on_the_first_insert() -> bool:
    """Packed to one hundred percent, the very next insert splits a leaf.

    The ninety percent custom is this measurement: headroom for the inserts that follow the
    load, priced at ten percent of the space. Packed full, the tree built without a single
    split performs its first insert as a split, and a load followed by uniform inserts
    splits across the whole width in the first pass.
    """
    full = bulk_build(list(_pairs(2000)), fill=1.0)
    before = full.splits
    full.put(b"k0000500x", b"v")
    slack = bulk_build(list(_pairs(2000)), fill=0.9)
    slack.put(b"k0000500x", b"v")
    return full.splits == before + 1 and slack.splits == 0


@functools.cache
def unsorted_input_is_refused() -> bool:
    """The build trusts nothing: unsorted or duplicated input raises before packing."""
    pairs = list(_pairs(100))
    pairs.reverse()
    try:
        bulk_build(pairs)
        return False
    except ConfigError:
        pass
    duplicated = list(_pairs(100))
    duplicated[5] = duplicated[6]
    try:
        bulk_build(sorted(duplicated))
        return False
    except ConfigError:
        return True


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "both_trees_answer_identically": both_trees_answer_identically(),
        "sorted_inserts_half_fill": sorted_inserts_leave_half_empty_leaves(),
        "the_build_writes_once": the_bulk_build_never_splits_and_writes_once(),
        "full_packs_split_immediately": a_full_pack_splits_on_the_first_insert(),
        "unsorted_is_refused": unsorted_input_is_refused(),
    }
