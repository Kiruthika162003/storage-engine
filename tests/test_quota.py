from __future__ import annotations

import pytest

from store import quota as mod
from store.errors import ConfigError
from store.quota import Bucket, Shared


class TestBucket:
    def test_a_zero_capacity_is_refused(self):
        with pytest.raises(ConfigError):
            Bucket(capacity=0, refill_per_tick=1)

    def test_a_zero_refill_is_refused(self):
        with pytest.raises(ConfigError):
            Bucket(capacity=1, refill_per_tick=0)

    def test_a_fresh_bucket_is_full(self):
        assert Bucket(capacity=5, refill_per_tick=1).tokens == 5.0

    def test_a_spend_takes_one_token(self):
        bucket = Bucket(capacity=5, refill_per_tick=1)
        assert bucket.try_spend() and bucket.tokens == 4.0

    def test_an_empty_bucket_defers(self):
        bucket = Bucket(capacity=1, refill_per_tick=0.1)
        bucket.try_spend()
        assert not bucket.try_spend() and bucket.deferred == 1

    def test_the_refill_restores(self):
        bucket = Bucket(capacity=2, refill_per_tick=1)
        bucket.try_spend()
        bucket.try_spend()
        bucket.tick()
        assert bucket.try_spend()

    def test_the_refill_never_exceeds_capacity(self):
        bucket = Bucket(capacity=2, refill_per_tick=10)
        bucket.tick()
        assert bucket.tokens == 2.0

    def test_fractional_refills_accumulate(self):
        bucket = Bucket(capacity=2, refill_per_tick=0.5)
        bucket.try_spend()
        bucket.try_spend()
        bucket.tick()
        assert not bucket.try_spend()
        bucket.tick()
        assert bucket.try_spend()

    def test_spends_are_counted(self):
        bucket = Bucket(capacity=3, refill_per_tick=1)
        bucket.try_spend()
        bucket.try_spend()
        assert bucket.spent == 2


class TestShared:
    def test_a_zero_budget_is_refused(self):
        with pytest.raises(ConfigError):
            Shared(per_tick=0)

    def test_spends_draw_down_the_tick(self):
        shared = Shared(per_tick=2)
        assert shared.try_spend("a") and shared.try_spend("b")
        assert not shared.try_spend("c")

    def test_the_tick_restores_the_budget(self):
        shared = Shared(per_tick=1)
        shared.try_spend("a")
        shared.tick()
        assert shared.try_spend("a")

    def test_spending_is_attributed(self):
        shared = Shared(per_tick=5)
        shared.try_spend("a")
        shared.try_spend("a")
        shared.try_spend("b")
        assert shared.spent_by == {"a": 2, "b": 1}

    def test_deferrals_are_attributed(self):
        shared = Shared(per_tick=1)
        shared.try_spend("a")
        shared.try_spend("b")
        assert shared.deferred_by == {"b": 1}


class TestMeasurements:
    def test_shared_budgets_starve(self):
        assert mod.the_shared_budget_starves_the_quiet()

    def test_buckets_hold_the_floor(self):
        assert mod.buckets_hold_every_quiet_tenant_at_its_floor()

    def test_capacity_is_the_burst(self):
        assert mod.the_burst_allowance_is_the_capacity()

    def test_history_is_capped(self):
        assert mod.unused_allowance_does_not_bank_past_the_cap()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4
