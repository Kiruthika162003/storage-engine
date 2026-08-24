from __future__ import annotations

import pytest

from store import lockmanager as mod
from store.errors import Closed, Conflict
from store.lockmanager import Table


class TestAcquire:
    def test_a_free_lock_grants(self):
        table = Table()
        locker = table.begin()
        assert table.acquire(locker, b"k")

    def test_a_grant_records_the_owner(self):
        table = Table()
        locker = table.begin()
        table.acquire(locker, b"k")
        assert table.owners[b"k"] == locker.name

    def test_reacquiring_your_own_lock_grants(self):
        table = Table()
        locker = table.begin()
        table.acquire(locker, b"k")
        assert table.acquire(locker, b"k")

    def test_a_held_lock_parks_the_second_comer(self):
        table = Table()
        first, second = table.begin(), table.begin()
        table.acquire(first, b"k")
        assert not table.acquire(second, b"k")

    def test_a_parked_locker_is_waiting(self):
        table = Table()
        first, second = table.begin(), table.begin()
        table.acquire(first, b"k")
        table.acquire(second, b"k")
        assert second.waiting_for == b"k"

    def test_a_release_lets_the_waiter_in(self):
        table = Table()
        first, second = table.begin(), table.begin()
        table.acquire(first, b"k")
        table.acquire(second, b"k")
        table.release(first)
        assert table.acquire(second, b"k")

    def test_a_closed_locker_cannot_acquire(self):
        table = Table()
        locker = table.begin()
        table.release(locker)
        with pytest.raises(Closed):
            table.acquire(locker, b"k")


class TestDeadlock:
    def build_cycle(self):
        table = Table()
        older, younger = table.begin(), table.begin()
        table.acquire(older, b"x")
        table.acquire(younger, b"y")
        table.acquire(older, b"y")
        return table, older, younger

    def test_the_cycle_is_detected(self):
        table, _, younger = self.build_cycle()
        with pytest.raises(Conflict):
            table.acquire(younger, b"x")
        assert table.deadlocks == 1

    def test_the_youngest_is_the_victim(self):
        table, older, younger = self.build_cycle()
        with pytest.raises(Conflict):
            table.acquire(younger, b"x")
        assert younger.state == "aborted" and older.state == "open"

    def test_the_survivor_finishes(self):
        table, older, younger = self.build_cycle()
        with pytest.raises(Conflict):
            table.acquire(younger, b"x")
        assert table.acquire(older, b"y")

    def test_the_victims_locks_are_freed(self):
        table, _, younger = self.build_cycle()
        with pytest.raises(Conflict):
            table.acquire(younger, b"x")
        assert b"y" not in table.owners or table.owners[b"y"] != younger.name

    def test_no_cycle_no_deadlock(self):
        table = Table()
        first, second = table.begin(), table.begin()
        table.acquire(first, b"x")
        table.acquire(second, b"x")
        assert table.deadlocks == 0


class TestRelease:
    def test_release_frees_everything(self):
        table = Table()
        locker = table.begin()
        table.acquire(locker, b"a")
        table.acquire(locker, b"b")
        table.release(locker)
        assert not table.owners

    def test_release_commits_the_locker(self):
        table = Table()
        locker = table.begin()
        table.release(locker)
        assert locker.state == "committed"

    def test_a_double_release_is_refused(self):
        table = Table()
        locker = table.begin()
        table.release(locker)
        with pytest.raises(Closed):
            table.release(locker)

    def test_as_dict_counts_the_held(self):
        table = Table()
        locker = table.begin()
        table.acquire(locker, b"a")
        assert table.as_dict()["held"] == 1


class TestMeasurements:
    def test_free_locks_grant_immediately(self):
        assert mod.a_free_lock_grants_immediately()

    def test_the_youngest_dies(self):
        assert mod.the_textbook_deadlock_is_detected_and_the_youngest_dies()

    def test_aborts_free_the_locks(self):
        assert mod.an_aborted_locker_frees_everything_it_held()

    def test_deadlocks_track_contention(self):
        assert mod.deadlocks_track_contention_like_conflicts_did()

    def test_finished_means_finished(self):
        assert mod.a_finished_locker_refuses_more_work()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5
