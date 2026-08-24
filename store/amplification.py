from __future__ import annotations

import functools
from dataclasses import dataclass

from store.btree import PAGE_BYTES, RECORD_BYTES, Tree
from store.compaction import (
    Levelled,
    Load,
    Tiered,
    amplification,
    read_cost,
    run_load,
    stale,
)

# The three amplifications, measured together, because they trade against each other.
#
# Write amplification is how many times a record is rewritten after being written. Read
# amplification is how many places a lookup has to check. Space amplification is how much
# larger the store is than the data it holds. The RUM conjecture says a design can be good at
# two of read, update and memory, and pays at the third, and this module puts the designs this
# package already has on one axis system to see how the conjecture holds up in the small.
#
# The designs on hand: a levelled LSM, a tiered LSM, and the B-tree. Each is driven by the same
# write stream and measured on all three axes in the same units. The point is not a winner. The
# point is that every one of them is at a different corner, which is what the conjecture
# predicts, and the corner each occupies follows from one design decision each.

WRITES = 40000
KEYS = 20000


@dataclass(frozen=True)
class Point:
    """One design's position: write, read and space amplification together."""

    design: str
    write: float
    read: float
    space: float

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "design": self.design,
            "write_amplification": self.write,
            "read_amplification": self.read,
            "space_amplification": self.space,
        }


@functools.cache
def _load() -> Load:
    """The one write stream every design is measured against."""
    return Load(keys=KEYS, writes=WRITES)


@functools.cache
def _probes() -> tuple[bytes, ...]:
    """The keys the read cost is averaged over."""
    return tuple(f"k{at:09d}".encode() for at in range(0, KEYS, 97))


@functools.cache
def levelled_point() -> Point:
    """The levelled LSM's corner."""
    store = run_load(Levelled(), _load())
    return Point(
        design="levelled",
        write=amplification(store, _load()),
        read=read_cost(store, list(_probes())),
        space=round(1 / (1 - stale(store)), 3),
    )


@functools.cache
def tiered_point() -> Point:
    """The tiered LSM's corner."""
    store = run_load(Tiered(), _load())
    return Point(
        design="tiered",
        write=amplification(store, _load()),
        read=read_cost(store, list(_probes())),
        space=round(1 / (1 - stale(store)), 3),
    )


@functools.cache
def btree_point() -> Point:
    """The B-tree's corner, in the same units.

    Write amplification for the tree is bytes moved per byte written, because pages are its
    unit. Read amplification is the height, which is the number of places a read checks. Space
    is pages held over pages needed, which counts the slack splits leave behind.
    """
    tree = Tree()
    for record in _load().records():
        tree.put(record.key, record.value or b"\x00")
    written_bytes = tree.page_writes * PAGE_BYTES
    ingested_bytes = WRITES * RECORD_BYTES
    needed = max(tree.records * RECORD_BYTES / PAGE_BYTES, 1)
    return Point(
        design="btree",
        write=round(written_bytes / ingested_bytes, 3),
        read=float(tree.height),
        space=round(tree.pages / needed, 3),
    )


@functools.cache
def every_design_sits_at_a_different_corner() -> bool:
    """No design wins two axes against both rivals, which is the conjecture in the small.

    Tiered has the best write amplification and the worst read and space. The tree has the
    best read and space against the LSMs paying for it in write. Levelled sits between on
    every axis. If any one design won two axes outright the conjecture would have a
    counterexample on this workload, and none does.
    """
    designs = (levelled_point(), tiered_point(), btree_point())
    for one in designs:
        others = [other for other in designs if other is not one]
        wins = sum(
            axis(one) < min(axis(other) for other in others)
            for axis in (lambda p: p.write, lambda p: p.read, lambda p: p.space)
        )
        if wins >= 2:
            return False
    return True


@functools.cache
def tiered_wins_writes_and_pays_twice() -> bool:
    """The cheapest writer is the most expensive reader and the largest store.

    One decision produces all three numbers: letting runs pile up before merging. Fewer merges
    is less rewriting, more runs is more places to look, and more runs is more stale versions
    held. The three amplifications are one design choice seen from three sides.
    """
    tiered, levelled = tiered_point(), levelled_point()
    return (
        tiered.write < levelled.write
        and tiered.read > levelled.read
        and tiered.space > levelled.space
    )


@functools.cache
def the_tree_wins_space_not_reads_which_was_not_the_guess() -> bool:
    """I expected the tree to win reads outright. It ties, and wins space instead.

    The folklore says B-trees read fast and LSMs read slow. Measured in places checked, the
    tree's height is 3.0 and the levelled LSM's read cost is 2.981, a tie to two figures,
    because a levelled store keeps one run per level and three levels is three places, the
    same count as a three level tree. The folklore is remembering tiered stores, which do
    read worse here, 3.981.

    Where the tree is actually alone is space: 1.01 against 1.421 and 1.695. Sorting at write
    time means no version ever exists twice, so the only slack is the half empty pages splits
    leave. The write side pays for this at 43.969, six times the levelled LSM, which is the
    page size over the record size doing exactly what the btree module measured.
    """
    tree, levelled, tiered = btree_point(), levelled_point(), tiered_point()
    return (
        abs(tree.read - levelled.read) < 0.1
        and tree.space < min(levelled.space, tiered.space)
        and tree.write > max(levelled.write, tiered.write) * 5
    )


def table() -> list[dict]:
    """The three corners, one row each."""
    return [one.as_dict() for one in (levelled_point(), tiered_point(), btree_point())]


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "different_corners": every_design_sits_at_a_different_corner(),
        "tiered_pays_twice": tiered_wins_writes_and_pays_twice(),
        "the_tree_wins_space": the_tree_wins_space_not_reads_which_was_not_the_guess(),
    }
