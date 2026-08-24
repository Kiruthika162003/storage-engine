from __future__ import annotations

import pytest

from store import bufferpool as mod
from store.bufferpool import Pool
from store.errors import Closed, ConfigError, TooLarge


class TestPin:
    def test_a_zero_capacity_is_refused(self):
        with pytest.raises(ConfigError):
            Pool(capacity=0)

    def test_a_pin_fetches_the_page(self):
        pool = Pool(capacity=2)
        page = pool.pin(7)
        assert page.number == 7 and page.payload == (7).to_bytes(8, "little") * 512

    def test_the_first_pin_is_a_miss(self):
        pool = Pool(capacity=2)
        pool.pin(7)
        assert pool.misses == 1 and pool.hits == 0

    def test_the_second_pin_is_a_hit(self):
        pool = Pool(capacity=2)
        pool.pin(7)
        pool.pin(7)
        assert pool.hits == 1

    def test_pins_stack(self):
        pool = Pool(capacity=2)
        page = pool.pin(7)
        pool.pin(7)
        assert page.pins == 2

    def test_holders_are_recorded(self):
        pool = Pool(capacity=2)
        page = pool.pin(7, holder="scan-1")
        assert "scan-1" in page.holders

    def test_a_fetch_happens_once_per_residency(self):
        pool = Pool(capacity=2)
        pool.pin(7)
        pool.pin(7)
        assert pool.fetches == [7]


class TestUnpin:
    def test_an_unpin_releases(self):
        pool = Pool(capacity=2)
        page = pool.pin(7)
        pool.unpin(page)
        assert page.pins == 0

    def test_an_unpin_below_zero_is_refused(self):
        pool = Pool(capacity=2)
        page = pool.pin(7)
        pool.unpin(page)
        with pytest.raises(Closed):
            pool.unpin(page)

    def test_an_unpin_can_mark_dirty(self):
        pool = Pool(capacity=2)
        page = pool.pin(7)
        pool.unpin(page, dirty=True)
        assert page.dirty

    def test_dirt_is_sticky(self):
        pool = Pool(capacity=2)
        page = pool.pin(7)
        pool.unpin(page, dirty=True)
        again = pool.pin(7)
        pool.unpin(again, dirty=False)
        assert page.dirty

    def test_the_holder_is_removed(self):
        pool = Pool(capacity=2)
        page = pool.pin(7, holder="scan-1")
        pool.unpin(page, holder="scan-1")
        assert "scan-1" not in page.holders


class TestEviction:
    def test_an_unpinned_page_can_be_evicted(self):
        pool = Pool(capacity=1)
        page = pool.pin(1)
        pool.unpin(page)
        pool.pin(2)
        assert 1 not in pool.slots

    def test_a_pinned_page_cannot(self):
        pool = Pool(capacity=1)
        pool.pin(1)
        with pytest.raises(TooLarge):
            pool.pin(2)

    def test_exhaustion_is_counted(self):
        pool = Pool(capacity=1)
        pool.pin(1)
        with pytest.raises(TooLarge):
            pool.pin(2)
        assert pool.exhausted == 1

    def test_a_dirty_eviction_writes_back(self):
        pool = Pool(capacity=1)
        page = pool.pin(1)
        pool.unpin(page, dirty=True)
        pool.pin(2)
        assert pool.write_backs == 1

    def test_a_clean_eviction_does_not(self):
        pool = Pool(capacity=1)
        page = pool.pin(1)
        pool.unpin(page)
        pool.pin(2)
        assert pool.write_backs == 0

    def test_as_dict_counts_the_pinned(self):
        pool = Pool(capacity=3)
        pool.pin(1)
        page = pool.pin(2)
        pool.unpin(page)
        assert pool.as_dict()["pinned"] == 1


class TestMeasurements:
    def test_pins_survive_pressure(self):
        assert mod.a_pinned_page_survives_any_pressure()

    def test_exhaustion_names_names(self):
        assert mod.exhaustion_names_the_holders()

    def test_dirty_pages_write_back_once(self):
        assert mod.a_dirty_page_is_written_back_exactly_once()

    def test_double_unpin_is_refused(self):
        assert mod.a_double_unpin_is_refused()

    def test_repinning_shares(self):
        assert mod.repinning_is_free_and_counted()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5
