from __future__ import annotations

import pytest

from store import writeskew as mod
from store.errors import Conflict
from store.txn import Manager
from store.writeskew import (
    commit_si,
    on_call_count,
    roster,
    sign_off_guarded,
    sign_off_snapshot,
)


class TestRoster:
    def test_the_roster_starts_full(self):
        manager = Manager()
        roster(manager)
        audit = manager.begin()
        assert on_call_count(audit) == 2
        manager.abort(audit)


class TestCommitSi:
    def test_a_plain_write_commits(self):
        manager = Manager()
        txn = manager.begin()
        txn.put(b"k", b"v")
        commit_si(manager, txn)
        assert manager.history.value(b"k") == b"v"

    def test_a_write_write_collision_conflicts(self):
        manager = Manager()
        first, second = manager.begin(), manager.begin()
        first.put(b"k", b"1")
        second.put(b"k", b"2")
        commit_si(manager, first)
        with pytest.raises(Conflict):
            commit_si(manager, second)

    def test_a_read_of_changed_data_does_not_conflict(self):
        manager = Manager()
        setup = manager.begin()
        setup.put(b"seen", b"old")
        commit_si(manager, setup)
        reader = manager.begin()
        reader.get(b"seen")
        writer = manager.begin()
        writer.put(b"seen", b"new")
        commit_si(manager, writer)
        reader.put(b"other", b"x")
        commit_si(manager, reader)
        assert reader.state == "committed"

    def test_a_finished_transaction_is_refused(self):
        manager = Manager()
        txn = manager.begin()
        txn.put(b"k", b"v")
        commit_si(manager, txn)
        with pytest.raises(Conflict):
            commit_si(manager, txn)


class TestProcedures:
    def test_a_lone_snapshot_sign_off_succeeds(self):
        manager = Manager()
        roster(manager)
        assert sign_off_snapshot(manager, b"alice")

    def test_the_second_serial_sign_off_aborts_itself(self):
        manager = Manager()
        roster(manager)
        sign_off_snapshot(manager, b"alice")
        assert not sign_off_snapshot(manager, b"bob")

    def test_the_guarded_procedure_also_works_alone(self):
        manager = Manager()
        roster(manager)
        assert sign_off_guarded(manager, b"alice")


class TestMeasurements:
    def test_the_ward_empties_under_si(self):
        assert mod.true_snapshot_isolation_empties_the_ward()

    def test_read_validation_refuses_it(self):
        assert mod.the_packages_own_manager_already_refuses_it()

    def test_the_guard_repairs_si(self):
        assert mod.the_materialised_guard_repairs_true_si()

    def test_serial_runs_hide_the_anomaly(self):
        assert mod.sequential_sign_offs_never_needed_the_guard()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4
