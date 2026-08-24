from __future__ import annotations

import pytest

from store.backfill import (
    ROWS,
    Index,
    Table,
    _seeded_table,
    a_stale_chunk_clobbers_the_race,
    chunked_dual_write,
    dual_write_and_walk,
    dual_writes_plus_the_walk_converge_exactly,
    first_writer_wins_restores_convergence,
    summarise,
    the_walk_alone_misses_a_fifth_of_the_table,
    walk_only,
)


class TestTable:
    def test_a_write_bumps_the_version(self):
        table = Table()
        table.write(1, b"v")
        table.write(1, b"w")
        assert table.version == 2 and table.rows[1] == b"w"

    def test_the_seeded_table_is_deterministic(self):
        assert _seeded_table(3).rows == _seeded_table(3).rows

    def test_the_seeded_table_is_full(self):
        assert len(_seeded_table(3).rows) == ROWS


class TestIndex:
    def test_agrees_when_identical(self):
        table = Table()
        table.write(1, b"v")
        index = Index()
        index.put(1, b"v")
        assert index.agrees_with(table) == 0

    def test_a_missing_row_counts_as_wrong(self):
        table = Table()
        table.write(1, b"v")
        assert Index().agrees_with(table) == 1

    def test_a_stale_row_counts_as_wrong(self):
        table = Table()
        table.write(1, b"new")
        index = Index()
        index.put(1, b"old")
        assert index.agrees_with(table) == 1


class TestStrategies:
    def test_walk_only_covers_every_row_once(self):
        _, index = walk_only(3)
        assert len(index.entries) == ROWS

    def test_dual_write_leaves_nothing_wrong(self):
        table, index = dual_write_and_walk(9)
        assert index.agrees_with(table) == 0

    def test_the_guard_never_does_worse_than_no_guard(self):
        for seed in (3, 9, 21):
            table_u, unguarded = chunked_dual_write(seed, guarded=False)
            table_g, guarded = chunked_dual_write(seed, guarded=True)
            assert guarded.agrees_with(table_g) <= unguarded.agrees_with(table_u)

    def test_the_guard_converges_on_other_seeds_too(self):
        for seed in (9, 21):
            table, index = chunked_dual_write(seed, guarded=True)
            assert index.agrees_with(table) == 0


class TestClaims:
    @pytest.mark.parametrize(
        "claim",
        [
            the_walk_alone_misses_a_fifth_of_the_table,
            dual_writes_plus_the_walk_converge_exactly,
            a_stale_chunk_clobbers_the_race,
            first_writer_wins_restores_convergence,
        ],
    )
    def test_claim_holds(self, claim):
        assert claim() is True

    def test_summary_is_all_true(self):
        told = summarise()
        assert all(value for name, value in told.items() if name != "module")
