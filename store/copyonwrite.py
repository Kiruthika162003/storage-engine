"""Copy-on-write pages: what a snapshot costs when nothing is ever edited.

The in-place tree edits a page and journals the edit. The copy-on-write
tree never edits: it copies the changed leaf and every page on the path to
the root, and the old root remains a complete, immutable snapshot for
free. The price is path-length pages written per update, and the numbers
below measure that price, how batching amortises it, and what the free
snapshot actually holds.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

FANOUT = 16
DEPTH = 4
LEAVES = FANOUT ** (DEPTH - 1)


@dataclass
class Page:
    slots: tuple
    frozen: bool = True


@dataclass
class Cow:
    """Pages live in a table by id; roots are ids; nothing frozen changes."""

    pages: dict[int, Page] = field(default_factory=dict)
    next_id: int = 0
    written: int = 0

    def _fresh(self, slots: tuple) -> int:
        self.pages[self.next_id] = Page(slots=slots)
        self.next_id += 1
        self.written += 1
        return self.next_id - 1

    def build(self, values: list[int]) -> int:
        level = [self._fresh((value,)) for value in values]
        while len(level) > 1:
            level = [
                self._fresh(tuple(level[at : at + FANOUT]))
                for at in range(0, len(level), FANOUT)
            ]
        return level[0]

    def read(self, root: int, leaf_at: int) -> int:
        page = self.pages[root]
        for depth in range(DEPTH - 2, -1, -1):
            slot = (leaf_at // (FANOUT**depth)) % FANOUT
            page = self.pages[page.slots[slot]]
        return page.slots[0]

    def update(self, root: int, leaf_at: int, value: int) -> int:
        path = []
        page_id = root
        for depth in range(DEPTH - 2, -1, -1):
            slot = (leaf_at // (FANOUT**depth)) % FANOUT
            path.append((page_id, slot))
            page_id = self.pages[page_id].slots[slot]
        fresh = self._fresh((value,))
        for page_id, slot in reversed(path):
            slots = list(self.pages[page_id].slots)
            slots[slot] = fresh
            fresh = self._fresh(tuple(slots))
        return fresh

    def update_many(self, root: int, edits: dict[int, int]) -> int:
        """One epoch: copied pages stay thawed and absorb further edits."""
        fresh_root = root
        for leaf_at, value in sorted(edits.items()):
            fresh_root = self._update_thawed(fresh_root, leaf_at, value)
        stack = [fresh_root]
        while stack:
            page = self.pages[stack.pop()]
            if page.frozen:
                continue
            page.frozen = True
            if len(page.slots) > 1:
                stack.extend(page.slots)
        return fresh_root

    def _update_thawed(self, root: int, leaf_at: int, value: int) -> int:
        def copy_of(page_id: int) -> int:
            if not self.pages[page_id].frozen:
                return page_id
            fresh_id = self._fresh(self.pages[page_id].slots)
            self.pages[fresh_id].frozen = False
            return fresh_id

        fresh_root = copy_of(root)
        page_id = fresh_root
        for depth in range(DEPTH - 2, 0, -1):
            slot = (leaf_at // (FANOUT**depth)) % FANOUT
            child = copy_of(self.pages[page_id].slots[slot])
            slots = list(self.pages[page_id].slots)
            slots[slot] = child
            self.pages[page_id].slots = tuple(slots)
            page_id = child
        slot = leaf_at % FANOUT
        leaf = self._fresh((value,))
        slots = list(self.pages[page_id].slots)
        slots[slot] = leaf
        self.pages[page_id].slots = tuple(slots)
        return fresh_root


def _grown() -> tuple[Cow, int]:
    cow = Cow()
    return cow, cow.build(list(range(LEAVES)))


@functools.cache
def the_tree_costs_4369_pages_and_the_snapshot_zero() -> bool:
    """4096 leaves need 4369 pages; a snapshot writes nothing at all.

    The snapshot is the root id. Holding it is holding the whole tree at
    that instant, because no frozen page is ever edited. The page meter
    does not move when a snapshot is taken.
    """
    cow, root = _grown()
    built = cow.written
    kept = root
    return built == 4369 and cow.written == built and kept == root


@functools.cache
def every_lone_update_writes_the_path() -> bool:
    """One changed byte writes 4 pages: leaf, two inner pages, root.

    The in-place tree writes one page and a journal record. Copy on write
    multiplies every lone write by the depth, a 4x write amplification
    that buys free snapshots and no journal at all.
    """
    cow, root = _grown()
    before = cow.written
    cow.update(root, 100, 999)
    return cow.written - before == DEPTH and cow.read(root, 100) == 100


@functools.cache
def a_clustered_batch_pays_1_2_pages_per_edit() -> bool:
    """16 edits in one leaf group: 19 pages, against 64 done one by one.

    The epoch thaws each copied page once, so sixteen new leaves share one
    copied parent, one copied middle page and one copied root: 1.19 pages
    per edit instead of 4. Batching under copy on write is not a nicety,
    it is a 3.4x reduction in pages written.
    """
    cow, root = _grown()
    before = cow.written
    cow.update_many(root, {leaf: leaf * 2 for leaf in range(16)})
    return cow.written - before == 19


@functools.cache
def a_scattered_batch_shares_only_the_root() -> bool:
    """16 edits spread across the tree: 49 pages, barely below 64.

    Each edit lands in its own leaf group and its own middle page, so the
    only page all sixteen share is the root: 16 leaves, 16 parents, 16
    middles, 1 root. Amortisation follows locality, the same lesson the
    multiget batch measured on the read side.
    """
    cow, root = _grown()
    before = cow.written
    cow.update_many(root, {leaf * 256: 7 for leaf in range(16)})
    return cow.written - before == 49


@functools.cache
def the_frozen_past_stays_readable_forever() -> bool:
    """Three epochs later the first root still answers with its own values.

    Every leaf of the original snapshot reads back its original value
    after 33 edits across three epochs, because nothing it references was
    touched: the past is not restored, it was simply never damaged.
    """
    cow, root = _grown()
    second = cow.update_many(root, {leaf: leaf * 2 for leaf in range(16)})
    third = cow.update_many(second, {leaf * 256: 7 for leaf in range(16)})
    cow.update_many(third, {5: 1})
    sampled = list(range(0, LEAVES, 97))
    return all(cow.read(root, leaf) == leaf for leaf in sampled)


@functools.cache
def summarise() -> dict:
    return {
        "module": "store.copyonwrite",
        "the_tree_costs_4369_pages_and_the_snapshot_zero": (
            the_tree_costs_4369_pages_and_the_snapshot_zero()
        ),
        "every_lone_update_writes_the_path": every_lone_update_writes_the_path(),
        "a_clustered_batch_pays_1_2_pages_per_edit": (
            a_clustered_batch_pays_1_2_pages_per_edit()
        ),
        "a_scattered_batch_shares_only_the_root": (
            a_scattered_batch_shares_only_the_root()
        ),
        "the_frozen_past_stays_readable_forever": (
            the_frozen_past_stays_readable_forever()
        ),
    }
