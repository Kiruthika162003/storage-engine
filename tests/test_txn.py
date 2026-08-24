from __future__ import annotations

import pytest

from store import txn as mod
from store.errors import Closed, ConfigError, Conflict
from store.txn import Manager, balance, transfer


def funded(accounts: int = 3, amount: int = 100) -> Manager:
    manager = Manager()
    setup = manager.begin()
    for at in range(accounts):
        setup.put(f"acct:{at}".encode(), amount.to_bytes(8, "big"))
    manager.commit(setup)
    return manager


class TestBasics:
    def test_a_committed_write_is_visible(self):
        manager = Manager()
        txn = manager.begin()
        txn.put(b"k", b"v")
        manager.commit(txn)
        assert manager.history.value(b"k") == b"v"

    def test_an_aborted_write_is_not(self):
        manager = Manager()
        txn = manager.begin()
        txn.put(b"k", b"v")
        manager.abort(txn)
        assert manager.history.value(b"k") is None

    def test_an_empty_key_is_refused(self):
        manager = Manager()
        txn = manager.begin()
        with pytest.raises(ConfigError):
            txn.put(b"", b"v")
        manager.abort(txn)

    def test_a_buffered_delete_lands(self):
        manager = Manager()
        first = manager.begin()
        first.put(b"k", b"v")
        manager.commit(first)
        second = manager.begin()
        second.delete(b"k")
        manager.commit(second)
        assert manager.history.value(b"k") is None

    def test_a_transaction_reads_its_own_write(self):
        manager = Manager()
        txn = manager.begin()
        txn.put(b"k", b"mine")
        assert txn.get(b"k") == b"mine"
        manager.abort(txn)

    def test_a_transaction_reads_its_own_delete(self):
        manager = funded()
        txn = manager.begin()
        txn.delete(b"acct:0")
        assert txn.get(b"acct:0") is None
        manager.abort(txn)

    def test_a_transaction_reads_the_snapshot_otherwise(self):
        manager = funded()
        txn = manager.begin()
        assert txn.get(b"acct:0") is not None
        manager.abort(txn)

    def test_a_snapshot_is_released_on_commit(self):
        manager = Manager()
        txn = manager.begin()
        manager.commit(txn)
        assert not manager.history.open_snapshots

    def test_a_snapshot_is_released_on_abort(self):
        manager = Manager()
        txn = manager.begin()
        manager.abort(txn)
        assert not manager.history.open_snapshots

    def test_the_counters_track_outcomes(self):
        manager = Manager()
        manager.commit(manager.begin())
        manager.abort(manager.begin())
        assert manager.committed == 1 and manager.aborted == 1

    def test_as_dict_carries_the_state(self):
        manager = Manager()
        txn = manager.begin()
        assert txn.as_dict()["state"] == "open"
        manager.abort(txn)


class TestIsolation:
    def test_a_buffer_is_invisible_to_others(self):
        manager = Manager()
        writer, reader = manager.begin(), manager.begin()
        writer.put(b"k", b"v")
        assert reader.get(b"k") is None
        manager.abort(writer)
        manager.abort(reader)

    def test_a_commit_is_invisible_to_an_older_snapshot(self):
        manager = Manager()
        early = manager.begin()
        writer = manager.begin()
        writer.put(b"k", b"v")
        manager.commit(writer)
        assert early.get(b"k") is None
        manager.abort(early)

    def test_a_commit_is_visible_to_a_later_snapshot(self):
        manager = Manager()
        writer = manager.begin()
        writer.put(b"k", b"v")
        manager.commit(writer)
        later = manager.begin()
        assert later.get(b"k") == b"v"
        manager.abort(later)


class TestConflicts:
    def test_a_read_of_a_changed_key_conflicts(self):
        manager = funded()
        loser = manager.begin()
        loser.get(b"acct:0")
        winner = manager.begin()
        winner.put(b"acct:0", (999).to_bytes(8, "big"))
        manager.commit(winner)
        loser.put(b"other", b"x")
        with pytest.raises(Conflict):
            manager.commit(loser)

    def test_the_conflicted_transaction_is_aborted(self):
        manager = funded()
        loser = manager.begin()
        loser.get(b"acct:0")
        winner = manager.begin()
        winner.put(b"acct:0", (999).to_bytes(8, "big"))
        manager.commit(winner)
        with pytest.raises(Conflict):
            manager.commit(loser)
        assert loser.state == "aborted" and manager.conflicts == 1

    def test_a_conflicted_commit_writes_nothing(self):
        manager = funded()
        loser = manager.begin()
        loser.get(b"acct:0")
        loser.put(b"marker", b"x")
        winner = manager.begin()
        winner.put(b"acct:0", (999).to_bytes(8, "big"))
        manager.commit(winner)
        with pytest.raises(Conflict):
            manager.commit(loser)
        assert manager.history.value(b"marker") is None

    def test_a_read_of_an_unchanged_key_commits(self):
        manager = funded()
        txn = manager.begin()
        txn.get(b"acct:0")
        other = manager.begin()
        other.put(b"acct:1", (5).to_bytes(8, "big"))
        manager.commit(other)
        txn.put(b"acct:2", (7).to_bytes(8, "big"))
        manager.commit(txn)
        assert txn.state == "committed"

    def test_blind_writes_do_not_conflict(self):
        manager = Manager()
        first, second = manager.begin(), manager.begin()
        first.put(b"k", b"1")
        second.put(b"k", b"2")
        manager.commit(first)
        manager.commit(second)
        assert manager.conflicts == 0

    def test_a_delete_conflicts_a_reader(self):
        manager = funded()
        reader = manager.begin()
        reader.get(b"acct:0")
        deleter = manager.begin()
        deleter.delete(b"acct:0")
        manager.commit(deleter)
        reader.put(b"out", b"x")
        with pytest.raises(Conflict):
            manager.commit(reader)


class TestLifecycle:
    def test_a_committed_transaction_refuses_reads(self):
        manager = Manager()
        txn = manager.begin()
        manager.commit(txn)
        with pytest.raises(Closed):
            txn.get(b"k")

    def test_a_committed_transaction_refuses_writes(self):
        manager = Manager()
        txn = manager.begin()
        manager.commit(txn)
        with pytest.raises(Closed):
            txn.put(b"k", b"v")

    def test_a_committed_transaction_refuses_a_second_commit(self):
        manager = Manager()
        txn = manager.begin()
        manager.commit(txn)
        with pytest.raises(Closed):
            manager.commit(txn)

    def test_an_aborted_transaction_refuses_a_commit(self):
        manager = Manager()
        txn = manager.begin()
        manager.abort(txn)
        with pytest.raises(Closed):
            manager.commit(txn)

    def test_an_aborted_transaction_refuses_a_second_abort(self):
        manager = Manager()
        txn = manager.begin()
        manager.abort(txn)
        with pytest.raises(Closed):
            manager.abort(txn)


class TestTransfer:
    def test_a_transfer_moves_the_amount(self):
        manager = funded(2, 100)
        assert transfer(manager, b"acct:0", b"acct:1", 30)
        assert balance(manager, b"acct:0") == 70
        assert balance(manager, b"acct:1") == 130

    def test_an_overdraft_is_refused(self):
        manager = funded(2, 10)
        assert not transfer(manager, b"acct:0", b"acct:1", 50)
        assert balance(manager, b"acct:0") == 10

    def test_a_refused_transfer_writes_nothing(self):
        manager = funded(2, 10)
        transfer(manager, b"acct:0", b"acct:1", 50)
        assert balance(manager, b"acct:1") == 10

    def test_a_missing_account_reads_as_zero(self):
        manager = Manager()
        assert balance(manager, b"acct:none") == 0

    def test_transfers_chain(self):
        manager = funded(3, 100)
        transfer(manager, b"acct:0", b"acct:1", 100)
        transfer(manager, b"acct:1", b"acct:2", 200)
        assert balance(manager, b"acct:2") == 300

    def test_the_total_survives_a_chain(self):
        manager = funded(3, 100)
        transfer(manager, b"acct:0", b"acct:1", 40)
        transfer(manager, b"acct:1", b"acct:2", 90)
        total = sum(balance(manager, f"acct:{at}".encode()) for at in range(3))
        assert total == 300


class TestMeasurements:
    def test_money_is_conserved(self):
        assert mod.money_is_conserved_under_interleaving()

    def test_no_lost_updates(self):
        assert mod.a_lost_update_is_impossible_by_construction()

    def test_blind_writes_land(self):
        assert mod.blind_writes_do_not_conflict()

    def test_own_writes_are_visible(self):
        assert mod.a_transaction_reads_its_own_writes_and_nobody_elses()

    def test_finished_means_finished(self):
        assert mod.a_finished_transaction_refuses_everything()

    def test_conflicts_track_contention(self):
        assert mod.the_conflict_rate_is_the_contention_not_the_load()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_six_claims(self):
        assert len(mod.summarise()) == 6

    def test_the_contention_table_has_four_rows(self):
        assert len(mod.compare_the_contention()) == 4

    def test_the_conflict_rate_falls_with_spread_past_ten(self):
        rows = mod.compare_the_contention()
        rates = [row["conflict_rate"] for row in rows[1:]]
        assert rates == sorted(rates, reverse=True)

    def test_the_two_account_row_is_thinned_by_the_self_transfer_guard(self):
        rows = mod.compare_the_contention()
        assert rows[0]["conflict_rate"] < rows[1]["conflict_rate"]

    def test_commits_rise_with_spread(self):
        rows = mod.compare_the_contention()
        assert rows[-1]["committed"] > rows[0]["committed"]

    def test_the_contended_manager_is_cached(self):
        assert mod._contended(10, 100, 1) is mod._contended(10, 100, 1)

    def test_the_storm_leaves_no_open_snapshots(self):
        manager = mod._contended(10, 500, 2)
        assert not manager.history.open_snapshots
