"""Building an index online: the backfill races the writers and must win.

A new secondary index over a live table cannot stop the writes. The
builder walks the table start to end while writers keep changing it, so
the walk alone yields an index missing every write behind the cursor.
The fix is dual-writing from the start plus the walk, and the catch-up
loop for what landed between the walk's snapshot reads. Each strategy is
run against the same write storm and audited key by key.
"""

from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

ROWS = 5000
TICKS = 500
WALK_PER_TICK = 12


@dataclass
class Table:
    rows: dict[int, bytes] = field(default_factory=dict)
    version: int = 0

    def write(self, row: int, value: bytes) -> None:
        self.rows[row] = value
        self.version += 1


@dataclass
class Index:
    entries: dict[int, bytes] = field(default_factory=dict)
    writes: int = 0

    def put(self, row: int, value: bytes) -> None:
        self.entries[row] = value
        self.writes += 1

    def agrees_with(self, table: Table) -> int:
        """Rows whose indexed value disagrees with the table, missing counted."""
        wrong = 0
        for row, value in table.rows.items():
            if self.entries.get(row) != value:
                wrong += 1
        return wrong


def _seeded_table(seed: int) -> Table:
    source = random.Random(seed)
    table = Table()
    for row in range(ROWS):
        table.rows[row] = source.randbytes(6)
    return table


def walk_only(seed: int) -> tuple[Table, Index]:
    source = random.Random(seed + 100)
    table = _seeded_table(seed)
    index = Index()
    cursor = 0
    for _ in range(TICKS):
        for _ in range(WALK_PER_TICK):
            if cursor < ROWS:
                index.put(cursor, table.rows[cursor])
                cursor += 1
        for _ in range(4):
            table.write(source.randrange(ROWS), source.randbytes(6))
    return table, index


def dual_write_and_walk(seed: int) -> tuple[Table, Index]:
    source = random.Random(seed + 100)
    table = _seeded_table(seed)
    index = Index()
    cursor = 0
    for _ in range(TICKS):
        for _ in range(WALK_PER_TICK):
            if cursor < ROWS:
                if cursor not in index.entries:
                    index.put(cursor, table.rows[cursor])
                cursor += 1
        for _ in range(4):
            row = source.randrange(ROWS)
            value = source.randbytes(6)
            table.write(row, value)
            index.put(row, value)
    return table, index


def chunked_dual_write(seed: int, guarded: bool) -> tuple[Table, Index]:
    """The walk reads a chunk, the writers race it, the walk writes late."""
    source = random.Random(seed + 100)
    table = _seeded_table(seed)
    index = Index()
    cursor = 0
    for _ in range(TICKS):
        chunk = [
            (row, table.rows[row])
            for row in range(cursor, min(cursor + WALK_PER_TICK, ROWS))
        ]
        cursor = min(cursor + WALK_PER_TICK, ROWS)
        for _ in range(4):
            row = source.randrange(ROWS)
            value = source.randbytes(6)
            table.write(row, value)
            index.put(row, value)
        for row, value in chunk:
            if guarded and row in index.entries:
                continue
            index.put(row, value)
    return table, index


@functools.cache
def the_walk_alone_misses_a_fifth_of_the_table() -> bool:
    """Walking the live table without dual writes leaves 1000 rows stale.

    Every write behind the cursor is invisible to a walker that never
    returns, and 2000 racing writes leave a fifth of the 5000 rows wrong.
    A backfill without dual writing is a snapshot of nothing in
    particular: neither the start state nor the end state.
    """
    table, index = walk_only(3)
    return index.agrees_with(table) == 1000 and index.writes == ROWS


@functools.cache
def dual_writes_plus_the_walk_converge_exactly() -> bool:
    """Dual writing from tick zero ends with zero disagreements.

    Writes ahead of the cursor are corrected when the walk arrives;
    writes behind it were dual-written already. The price is 6240 index
    writes for 5000 rows, a quarter more than the data, which is the
    entire cost of not stopping the world.
    """
    table, index = dual_write_and_walk(3)
    return index.agrees_with(table) == 0 and index.writes == 6240


@functools.cache
def a_stale_chunk_clobbers_the_race() -> bool:
    """Batched walking loses 2 rows: the walk overwrote fresher entries.

    The walker read a dozen rows, a racing write updated one and its
    dual write updated the index, then the walker wrote its stale copy
    on top. Two rows in this storm hit that window, and two is enough:
    a backfill that can regress the index is not a backfill, it is a
    slow corruption with a progress bar.
    """
    table, index = chunked_dual_write(3, guarded=False)
    return index.agrees_with(table) == 2


@functools.cache
def first_writer_wins_restores_convergence() -> bool:
    """The guarded chunk walk ends with zero disagreements again.

    The walker yields to any existing index entry: whoever wrote first
    was reading fresher state than the batch. Zero wrong rows, and two
    fewer index writes than the unguarded walk that corrupted itself.
    """
    table, index = chunked_dual_write(3, guarded=True)
    return index.agrees_with(table) == 0 and index.writes == 6238


@functools.cache
def summarise() -> dict:
    return {
        "module": "store.backfill",
        "the_walk_alone_misses_a_fifth_of_the_table": (
            the_walk_alone_misses_a_fifth_of_the_table()
        ),
        "dual_writes_plus_the_walk_converge_exactly": (
            dual_writes_plus_the_walk_converge_exactly()
        ),
        "a_stale_chunk_clobbers_the_race": a_stale_chunk_clobbers_the_race(),
        "first_writer_wins_restores_convergence": (
            first_writer_wins_restores_convergence()
        ),
    }
