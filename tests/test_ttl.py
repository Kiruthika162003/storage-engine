from __future__ import annotations

import pytest

from store import ttl as mod
from store.errors import ConfigError
from store.ttl import Entry, Shelf


class TestEntry:
    def test_no_deadline_is_always_alive(self):
        assert Entry(value=b"v").alive(10**9)

    def test_before_the_deadline_is_alive(self):
        assert Entry(value=b"v", deadline=10).alive(9)

    def test_at_the_deadline_is_dead(self):
        assert not Entry(value=b"v", deadline=10).alive(10)


class TestShelf:
    def test_a_put_reads_back(self):
        shelf = Shelf()
        shelf.put(b"k", b"v")
        assert shelf.get(b"k") == b"v"

    def test_a_missing_key_is_absent(self):
        assert Shelf().get(b"k") is None

    def test_a_zero_ttl_is_refused(self):
        with pytest.raises(ConfigError):
            Shelf().put(b"k", b"v", ttl=0)

    def test_time_does_not_reverse(self):
        with pytest.raises(ConfigError):
            Shelf().tick(-1)

    def test_a_key_without_a_ttl_survives_forever(self):
        shelf = Shelf()
        shelf.put(b"k", b"v")
        shelf.tick(10**6)
        assert shelf.get(b"k") == b"v"

    def test_a_key_with_a_ttl_expires(self):
        shelf = Shelf()
        shelf.put(b"k", b"v", ttl=3)
        shelf.tick(3)
        assert shelf.get(b"k") is None

    def test_an_expired_read_removes_the_entry(self):
        shelf = Shelf()
        shelf.put(b"k", b"v", ttl=3)
        shelf.tick(3)
        shelf.get(b"k")
        assert shelf.held == 0 and shelf.lazy_removals == 1

    def test_an_unread_expired_key_stays_held(self):
        shelf = Shelf()
        shelf.put(b"k", b"v", ttl=3)
        shelf.tick(3)
        assert shelf.held == 1 and shelf.live() == 0

    def test_a_sweep_removes_the_unread(self):
        shelf = Shelf()
        shelf.put(b"k", b"v", ttl=3)
        shelf.tick(3)
        assert shelf.sweep() == 1 and shelf.held == 0

    def test_a_sweep_spares_the_living(self):
        shelf = Shelf()
        shelf.put(b"a", b"v", ttl=3)
        shelf.put(b"b", b"v", ttl=100)
        shelf.tick(3)
        shelf.sweep()
        assert shelf.get(b"b") == b"v"

    def test_a_sweep_of_nothing_removes_nothing(self):
        shelf = Shelf()
        shelf.put(b"k", b"v", ttl=100)
        assert shelf.sweep() == 0

    def test_a_rewrite_extends_the_life(self):
        shelf = Shelf()
        shelf.put(b"k", b"v", ttl=3)
        shelf.tick(2)
        shelf.put(b"k", b"v", ttl=3)
        shelf.tick(2)
        assert shelf.get(b"k") == b"v"

    def test_as_dict_counts_the_dead(self):
        shelf = Shelf()
        shelf.put(b"k", b"v", ttl=1)
        shelf.tick(1)
        made = shelf.as_dict()
        assert made["held"] == 1 and made["live"] == 0


class TestMeasurements:
    def test_reads_enforce_the_deadline(self):
        assert mod.an_expired_key_reads_as_absent_before_any_cleanup()

    def test_the_boundary_is_exclusive(self):
        assert mod.a_key_read_at_the_last_tick_is_alive()

    def test_lazy_leaves_the_unread(self):
        assert mod.lazy_expiry_leaves_the_unread_dead_forever()

    def test_a_sweep_reclaims_the_rest(self):
        assert mod.a_sweep_reclaims_what_reads_never_will()

    def test_a_rewrite_clears_the_deadline(self):
        assert mod.a_rewrite_clears_the_deadline()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5

    def test_the_policy_table_has_two_rows(self):
        rows = mod.compare_the_policies(1000)
        assert [row["policy"] for row in rows] == ["lazy", "sweep"]

    def test_the_swept_shelf_ends_empty(self):
        rows = mod.compare_the_policies(1000)
        assert rows[1]["held"] == 0 and rows[0]["held"] > 0

    def test_the_abandoned_shelf_is_cached(self):
        assert mod._abandoned(100, 0.2, 1) is mod._abandoned(100, 0.2, 1)
