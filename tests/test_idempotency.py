from __future__ import annotations

import pytest

from store import idempotency as mod
from store.errors import ConfigError
from store.idempotency import Keyed, Naive


class TestKeyed:
    def test_a_zero_window_is_refused(self):
        with pytest.raises(ConfigError):
            Keyed(window=0)

    def test_a_first_deposit_applies(self):
        store = Keyed()
        assert store.deposit(b"a", 10) == 10 and store.balance == 10

    def test_a_repeat_does_not_reapply(self):
        store = Keyed()
        store.deposit(b"a", 10)
        store.deposit(b"a", 10)
        assert store.balance == 10 and store.replayed == 1

    def test_distinct_keys_both_apply(self):
        store = Keyed()
        store.deposit(b"a", 10)
        store.deposit(b"b", 5)
        assert store.balance == 15

    def test_the_replay_answer_is_frozen(self):
        store = Keyed()
        store.deposit(b"a", 10)
        store.deposit(b"b", 100)
        assert store.deposit(b"a", 10) == 10

    def test_the_window_evicts_oldest_first(self):
        store = Keyed(window=2)
        store.deposit(b"a", 1)
        store.deposit(b"b", 1)
        store.deposit(b"c", 1)
        store.deposit(b"a", 1)
        assert store.executed == 4

    def test_inside_the_window_stays_remembered(self):
        store = Keyed(window=3)
        store.deposit(b"a", 1)
        store.deposit(b"b", 1)
        store.deposit(b"a", 1)
        assert store.executed == 2


class TestNaive:
    def test_every_delivery_applies(self):
        store = Naive()
        store.deposit(b"a", 10)
        store.deposit(b"a", 10)
        assert store.balance == 20


class TestStorm:
    def test_the_storm_duplicates(self):
        store = Naive()
        deliveries = mod._storm(store, 200, 7)
        assert deliveries > 200

    def test_the_storm_is_deterministic(self):
        first = Naive()
        second = Naive()
        assert mod._storm(first, 200, 7) == mod._storm(second, 200, 7)


class TestMeasurements:
    def test_naive_stores_apply_deliveries(self):
        assert mod.the_naive_store_applies_every_delivery()

    def test_keyed_stores_apply_intents(self):
        assert mod.the_keyed_store_applies_every_intent_once()

    def test_replays_answer_the_original(self):
        assert mod.a_replay_returns_the_original_answer_not_the_current_one()

    def test_the_window_is_the_budget(self):
        assert mod.a_retry_after_the_window_reapplies()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4
