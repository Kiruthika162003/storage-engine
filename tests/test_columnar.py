from __future__ import annotations

import pytest

from store import columnar as mod
from store.columnar import FIELDS, ROW_BYTES, WIDTHS, ColumnStore, RowStore
from store.errors import ConfigError


def a_row(at: int = 0) -> dict[str, bytes]:
    return {
        "id": at.to_bytes(8, "big"),
        "status": b"\x01",
        "amount": (100).to_bytes(8, "big"),
        "city": b"012345678901",
        "note": bytes(40),
    }


class TestRowStore:
    def test_an_incomplete_row_is_refused(self):
        with pytest.raises(ConfigError):
            RowStore().insert({"id": b"x"})

    def test_an_inserted_row_reads_back(self):
        store = RowStore()
        store.insert(a_row())
        assert store.read_row(0) == a_row()

    def test_a_field_scan_returns_every_value(self):
        store = RowStore()
        for at in range(5):
            store.insert(a_row(at))
        assert store.scan_field("id") == [at.to_bytes(8, "big") for at in range(5)]

    def test_a_field_scan_touches_whole_rows(self):
        store = RowStore()
        for at in range(5):
            store.insert(a_row(at))
        store.scan_field("status")
        assert store.bytes_touched == 5 * ROW_BYTES

    def test_a_point_read_touches_one_row(self):
        store = RowStore()
        store.insert(a_row())
        store.read_row(0)
        assert store.bytes_touched == ROW_BYTES


class TestColumnStore:
    def test_an_incomplete_row_is_refused(self):
        with pytest.raises(ConfigError):
            ColumnStore().insert({"id": b"x"})

    def test_an_inserted_row_gathers_back(self):
        store = ColumnStore()
        store.insert(a_row())
        assert store.read_row(0) == a_row()

    def test_a_field_scan_touches_only_the_field(self):
        store = ColumnStore()
        for at in range(5):
            store.insert(a_row(at))
        store.scan_field("status")
        assert store.bytes_touched == 5 * WIDTHS["status"]

    def test_a_point_read_touches_every_column_once(self):
        store = ColumnStore()
        store.insert(a_row())
        store.read_row(0)
        assert store.bytes_touched == ROW_BYTES

    def test_the_row_count_tracks_inserts(self):
        store = ColumnStore()
        for at in range(7):
            store.insert(a_row(at))
        assert store.rows == 7


class TestAgreement:
    def test_the_filled_stores_agree_on_rows(self):
        rows, columns = mod._filled(500)
        assert rows.read_row(250) == columns.read_row(250)

    def test_the_filled_stores_agree_on_columns(self):
        rows, columns = mod._filled(500)
        for name in FIELDS:
            assert rows.scan_field(name) == columns.scan_field(name)


class TestMeasurements:
    def test_one_table_two_layouts(self):
        assert mod.both_layouts_hold_the_same_table()

    def test_narrow_scans_win_by_the_width(self):
        assert mod.a_one_byte_field_scan_is_sixty_nine_times_cheaper_in_columns()

    def test_point_reads_prefer_rows(self):
        assert mod.a_point_read_is_one_touch_in_rows_and_five_in_columns()

    def test_the_advantage_is_the_width_ratio(self):
        assert mod.the_scan_advantage_shrinks_with_the_field()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4

    def test_the_filled_pair_is_cached(self):
        assert mod._filled(100) is mod._filled(100)
