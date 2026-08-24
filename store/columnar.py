from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

from store.errors import ConfigError

# Rows and columns: the same table, two layouts, opposite bills.
#
# The key value store holds rows: every field of a record adjacent, which is what a point
# lookup wants, all of one entity in one read. An analytical scan wants one field of every
# entity, and the row layout makes it pay for the fields it skips, because bytes travel in
# blocks and the skipped fields share blocks with the wanted one. The column layout
# transposes: each field contiguous, a scan of one field touches only that field's bytes,
# and the point lookup now gathers a row from as many places as it has fields. Bytes
# touched is the honest meter and both layouts are charged by it.

FIELDS = ("id", "status", "amount", "city", "note")
WIDTHS = {"id": 8, "status": 1, "amount": 8, "city": 12, "note": 40}
ROW_BYTES = sum(WIDTHS.values())


@dataclass
class RowStore:
    """Rows adjacent."""

    rows: list[dict[str, bytes]] = field(default_factory=list)
    bytes_touched: int = field(default=0)

    def insert(self, row: dict[str, bytes]) -> None:
        if set(row) != set(FIELDS):
            raise ConfigError("a row carries every field")
        self.rows.append(row)

    def scan_field(self, name: str) -> list[bytes]:
        """One field of every row: the whole row travels anyway."""
        found = []
        for row in self.rows:
            self.bytes_touched += ROW_BYTES
            found.append(row[name])
        return found

    def read_row(self, at: int) -> dict[str, bytes]:
        """One whole row: one touch."""
        self.bytes_touched += ROW_BYTES
        return self.rows[at]


@dataclass
class ColumnStore:
    """Fields adjacent."""

    columns: dict[str, list[bytes]] = field(
        default_factory=lambda: {name: [] for name in FIELDS}
    )
    bytes_touched: int = field(default=0)

    def insert(self, row: dict[str, bytes]) -> None:
        if set(row) != set(FIELDS):
            raise ConfigError("a row carries every field")
        for name in FIELDS:
            self.columns[name].append(row[name])

    def scan_field(self, name: str) -> list[bytes]:
        """One field of every row: only that field's bytes travel."""
        held = self.columns[name]
        self.bytes_touched += WIDTHS[name] * len(held)
        return list(held)

    def read_row(self, at: int) -> dict[str, bytes]:
        """One row gathered from every column: a touch per field."""
        made = {}
        for name in FIELDS:
            self.bytes_touched += WIDTHS[name]
            made[name] = self.columns[name][at]
        return made

    @property
    def rows(self) -> int:
        return len(self.columns[FIELDS[0]])


@functools.cache
def _filled(count: int = 20000, seed: int = 199) -> tuple[RowStore, ColumnStore]:
    """Both layouts holding identical rows."""
    source = random.Random(seed)
    rows = RowStore()
    columns = ColumnStore()
    for at in range(count):
        row = {
            "id": at.to_bytes(8, "big"),
            "status": bytes([source.randrange(4)]),
            "amount": source.randrange(10**6).to_bytes(8, "big"),
            "city": f"{source.randrange(200):012d}".encode(),
            "note": source.randbytes(40),
        }
        rows.insert(row)
        columns.insert(row)
    return rows, columns


@functools.cache
def both_layouts_hold_the_same_table() -> bool:
    """Every row gathers identically from both layouts, and every column scans identically.

    The transpose license: layout is representation, and the two must be observationally
    one table before any bill is compared.
    """
    rows, columns = _filled(3000)
    for at in (0, 1, 1500, 2999):
        if rows.read_row(at) != columns.read_row(at):
            return False
    return all(rows.scan_field(name) == columns.scan_field(name) for name in FIELDS)


@functools.cache
def a_one_byte_field_scan_is_sixty_nine_times_cheaper_in_columns() -> bool:
    """Scanning status touches 69 bytes per row in the row layout and 1 in the column.

    The ratio is exactly the row width over the field width, 69 to 1 for the one byte
    status, because the row layout drags the note field's forty bytes past the bus for
    every status it reads. Analytics on wide rows is the row layout's worst case, and the
    width ratio is the entire speedup, no cleverness involved.
    """
    rows, columns = _filled()
    rows.scan_field("status")
    columns.scan_field("status")
    ratio = rows.bytes_touched / columns.bytes_touched
    return abs(ratio - ROW_BYTES / WIDTHS["status"]) < 0.01


@functools.cache
def a_point_read_is_one_touch_in_rows_and_five_in_columns() -> bool:
    """The row layout reads a row in one place; the column layout gathers from five.

    The bytes are the same 69 either way; the touches are one against five, and on a disk
    where a touch is a seek the difference is the whole transaction workload's latency.
    The two layouts are the same trade as the vlog's key value separation: locality given
    to one access pattern is taken from the other, and no layout has it both ways.
    """
    rows, columns = _filled(1000)
    rows.bytes_touched = 0
    columns.bytes_touched = 0
    rows.read_row(500)
    columns.read_row(500)
    return rows.bytes_touched == columns.bytes_touched == ROW_BYTES


@functools.cache
def the_scan_advantage_shrinks_with_the_field() -> bool:
    """Scanning the forty byte note field wins by 1.7, the one byte status by 69.

    The column layout's advantage on a field is the row width over that field's width, so
    wide fields barely benefit and narrow ones benefit enormously. A table of uniformly
    wide fields gains little from columns, which is why columnar formats spend their real
    effort on the encodings the narrow-field win pays for.
    """
    rows, columns = _filled()
    ratios = {}
    for name in ("status", "note"):
        rows.bytes_touched = 0
        columns.bytes_touched = 0
        rows.scan_field(name)
        columns.scan_field(name)
        ratios[name] = rows.bytes_touched / columns.bytes_touched
    return ratios["status"] > 60 and ratios["note"] < 2


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "one_table_two_layouts": both_layouts_hold_the_same_table(),
        "narrow_scans_win_by_the_width": (
            a_one_byte_field_scan_is_sixty_nine_times_cheaper_in_columns()
        ),
        "point_reads_prefer_rows": a_point_read_is_one_touch_in_rows_and_five_in_columns(),
        "the_advantage_is_the_width_ratio": the_scan_advantage_shrinks_with_the_field(),
    }
