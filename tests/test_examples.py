from __future__ import annotations

from examples import bank, lifecycle, sessions, telemetry, tenants
from store.composite import encode_field
from store.rangedel import Ranged
from store.ttl import Shelf


class TestBank:
    def test_the_bank_runs_clean(self, capsys):
        assert bank.main() == 0
        out = capsys.readouterr().out
        assert "'balanced': True" in out

    def test_the_audit_balances_after_a_day(self):
        manager = bank.open_the_bank()
        bank.a_day_of_business(manager, seed=99)
        assert bank.audit(manager)["balanced"]

    def test_a_statement_reads_one_account(self):
        manager = bank.open_the_bank()
        made = bank.statement(manager, 3)
        assert made == {"account": 3, "balance": bank.OPENING}

    def test_statements_release_their_snapshots(self):
        manager = bank.open_the_bank()
        bank.statement(manager, 1)
        assert not manager.history.open_snapshots


class TestSessions:
    def test_the_sessions_run_clean(self, capsys):
        assert sessions.main() == 0
        assert "logins" in capsys.readouterr().out

    def test_a_login_creates_a_live_session(self):
        shelf = Shelf()
        token = sessions.login(shelf, 7, 0)
        assert shelf.get(token) == b"user-7"

    def test_a_touch_slides_the_expiry(self):
        shelf = Shelf()
        token = sessions.login(shelf, 7, 0)
        shelf.tick(sessions.SESSION_TICKS - 1)
        assert sessions.touch(shelf, token)
        shelf.tick(sessions.SESSION_TICKS - 1)
        assert shelf.get(token) is not None

    def test_an_expired_touch_reports_failure(self):
        shelf = Shelf()
        token = sessions.login(shelf, 7, 0)
        shelf.tick(sessions.SESSION_TICKS)
        assert not sessions.touch(shelf, token)


class TestTenants:
    def test_the_tenants_run_clean(self, capsys):
        assert tenants.main() == 0
        out = capsys.readouterr().out
        assert "1 record(s)" in out and "1 range shard" in out

    def test_the_drop_removes_only_its_tenant(self):
        store = Ranged()
        tenants.load(store)
        tenants.drop_tenant(store, 2)
        keys = store.keys()
        assert not any(key.startswith(encode_field(b"t002")) for key in keys)
        assert any(key.startswith(encode_field(b"t003")) for key in keys)


class TestTelemetry:
    def test_the_telemetry_runs_clean(self, capsys):
        assert telemetry.main() == 0
        assert "sketch" in capsys.readouterr().out


class TestLifecycle:
    def test_the_lifecycle_runs_clean(self, capsys):
        assert lifecycle.main() == 0
        out = capsys.readouterr().out
        assert "agrees=True" in out and "invariants: True" in out
