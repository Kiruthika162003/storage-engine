from __future__ import annotations

import pytest

from store import undo as mod
from store.errors import ConfigError
from store.undo import System


class TestSystem:
    def test_two_transactions_at_once_are_refused(self):
        system = System(steal=False, force=False)
        system.begin("a")
        with pytest.raises(ConfigError):
            system.begin("b")

    def test_a_write_outside_a_transaction_is_refused(self):
        with pytest.raises(ConfigError):
            System(steal=False, force=False).write("k", 1)

    def test_a_commit_outside_a_transaction_is_refused(self):
        with pytest.raises(ConfigError):
            System(steal=False, force=False).commit()

    def test_a_write_lands_in_the_buffer(self):
        system = System(steal=False, force=False)
        system.begin("t")
        system.write("k", 5)
        assert system.buffer["k"] == 5 and "k" not in system.disk

    def test_undo_records_the_before_image(self):
        system = System(steal=False, force=False)
        system.disk["k"] = 3
        system.begin("t")
        system.write("k", 5)
        assert system.undo_log == [("t", "k", 3)]

    def test_no_steal_keeps_dirty_pages_in_memory(self):
        system = System(steal=False, force=False)
        system.begin("t")
        system.write("k", 5)
        system.maybe_steal("k")
        assert "k" not in system.disk

    def test_steal_writes_the_dirty_page_home(self):
        system = System(steal=True, force=False)
        system.begin("t")
        system.write("k", 5)
        system.maybe_steal("k")
        assert system.disk["k"] == 5

    def test_force_flushes_at_commit(self):
        system = System(steal=False, force=True)
        system.begin("t")
        system.write("k", 5)
        system.commit()
        assert system.disk["k"] == 5 and not system.buffer

    def test_no_force_leaves_the_buffer_dirty(self):
        system = System(steal=False, force=False)
        system.begin("t")
        system.write("k", 5)
        system.commit()
        assert "k" not in system.disk and system.buffer["k"] == 5


class TestRecovery:
    def test_a_committed_write_survives(self):
        system = System(steal=False, force=False)
        system.begin("t")
        system.write("k", 5)
        system.commit()
        assert system.crash_and_recover()["k"] == 5

    def test_an_uncommitted_write_vanishes(self):
        system = System(steal=False, force=False)
        system.disk["k"] = 1
        system.begin("t")
        system.write("k", 5)
        assert system.crash_and_recover()["k"] == 1

    def test_a_stolen_uncommitted_write_is_undone(self):
        system = System(steal=True, force=False)
        system.disk["k"] = 1
        system.begin("t")
        system.write("k", 5)
        system.maybe_steal("k")
        assert system.crash_and_recover()["k"] == 1

    def test_winner_and_loser_separate_correctly(self):
        state = mod._run(steal=True, force=False, crash_mid=True)
        assert state == {"a": 11, "b": 20}


class TestMeasurements:
    def test_the_engines_cell_is_redo_only(self):
        assert mod.no_steal_no_force_needs_only_redo()

    def test_steal_without_undo_corrupts(self):
        assert mod.steal_without_undo_persists_the_losers_write()

    def test_steal_with_undo_recovers(self):
        assert mod.steal_with_undo_reverses_the_losers_write()

    def test_force_prepays_redo(self):
        assert mod.force_makes_redo_redundant_and_slow()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4
