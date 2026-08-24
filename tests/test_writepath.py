from __future__ import annotations

import pytest

from store.eval.writepath import (
    APPEND,
    SYNC,
    WritePath,
    a_bigger_memtable_trades_replay_for_flushes,
    every_write_synced_costs_thirteen_times,
    render,
    run,
    summarise,
    sweep,
    the_first_grouping_buys_almost_everything,
    the_knobs_do_not_touch,
    the_window_is_the_group_minus_one,
    worst_case_lost,
)


class TestWritePath:
    def test_a_single_synced_put_charges_append_plus_sync(self):
        path = WritePath(sync_every=1, flush_at=10**9)
        path.put()
        assert path.charges == APPEND + SYNC

    def test_grouped_puts_defer_the_sync(self):
        path = WritePath(sync_every=4, flush_at=10**9)
        for _ in range(3):
            path.put()
        assert path.charges == 3 * APPEND
        path.put()
        assert path.charges == 4 * APPEND + SYNC

    def test_lost_in_a_crash_counts_unsynced_acks(self):
        path = WritePath(sync_every=4, flush_at=10**9)
        for _ in range(3):
            path.put()
        assert path.lost_in_a_crash() == 3

    def test_a_flush_syncs_what_it_covers(self):
        path = WritePath(sync_every=100, flush_at=5)
        for _ in range(5):
            path.put()
        assert path.lost_in_a_crash() == 0
        assert path.flushes == 1

    def test_an_empty_flush_charges_nothing(self):
        path = WritePath(sync_every=1, flush_at=10)
        path.flush()
        assert path.charges == 0

    def test_replay_covers_synced_but_unflushed(self):
        path = WritePath(sync_every=1, flush_at=10**9)
        for _ in range(7):
            path.put()
        assert path.replay_length() == 7


class TestSweep:
    def test_the_sweep_covers_six_cells(self):
        assert len(sweep()) == 6

    def test_run_is_deterministic(self):
        assert run(8, 128).charges == run(8, 128).charges

    def test_charges_fall_as_grouping_grows(self):
        by_group = [run(group, 128).charges for group in (1, 8, 64)]
        assert by_group == sorted(by_group, reverse=True)

    def test_worst_case_never_reaches_the_group(self):
        for group in (1, 4, 16):
            assert worst_case_lost(group, 128) < group

    def test_render_has_a_row_per_cell(self):
        assert len(render().splitlines()) == 7


class TestClaims:
    @pytest.mark.parametrize(
        "claim",
        [
            every_write_synced_costs_thirteen_times,
            the_first_grouping_buys_almost_everything,
            the_window_is_the_group_minus_one,
            a_bigger_memtable_trades_replay_for_flushes,
            the_knobs_do_not_touch,
        ],
    )
    def test_claim_holds(self, claim):
        assert claim() is True

    def test_summary_is_all_true(self):
        told = summarise()
        assert all(value for name, value in told.items() if name != "module")
