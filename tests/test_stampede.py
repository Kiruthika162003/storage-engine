from __future__ import annotations

from store import stampede as mod
from store.stampede import Backing, Cache


def fresh(coalesce: bool, ttl: int = 50) -> Cache:
    return Cache(backing=Backing(), coalesce=coalesce, ttl=ttl)


class TestCache:
    def test_the_first_read_misses_and_rebuilds(self):
        cache = fresh(False)
        cache.tick()
        assert cache.get(b"k") == b"value-of-k"
        assert cache.backing.queries == 1

    def test_the_rebuild_lands_next_tick(self):
        cache = fresh(False)
        cache.tick()
        cache.get(b"k")
        cache.get(b"k")
        assert cache.backing.queries == 2
        cache.tick()
        cache.get(b"k")
        assert cache.backing.queries == 2

    def test_a_fresh_entry_hits_until_expiry(self):
        cache = fresh(False, ttl=3)
        cache.tick()
        cache.get(b"k")
        cache.tick()
        for _ in range(5):
            cache.get(b"k")
        assert cache.backing.queries == 1

    def test_expiry_forces_a_rebuild(self):
        cache = fresh(False, ttl=2)
        cache.tick()
        cache.get(b"k")
        cache.tick()
        cache.tick()
        cache.tick()
        cache.get(b"k")
        assert cache.backing.queries == 2

    def test_coalescing_blocks_the_second_misser(self):
        cache = fresh(True)
        cache.tick()
        assert cache.get(b"k") is not None
        assert cache.get(b"k") is None
        assert cache.waited == 1 and cache.backing.queries == 1

    def test_without_coalescing_every_misser_rebuilds(self):
        cache = fresh(False)
        cache.tick()
        cache.get(b"k")
        cache.get(b"k")
        cache.get(b"k")
        assert cache.backing.queries == 3

    def test_the_peak_meter_tracks_within_tick_load(self):
        cache = fresh(False)
        cache.tick()
        for _ in range(7):
            cache.get(b"k")
        assert cache.backing.peak_in_tick == 7


class TestMeasurements:
    def test_the_stampede_multiplies(self):
        assert mod.the_stampede_multiplies_backing_load_by_the_reader_count()

    def test_coalescing_cuts_to_one(self):
        assert mod.coalescing_cuts_each_expiry_to_one_rebuild()

    def test_jitter_spreads_the_fleet(self):
        assert mod.synchronized_ttls_expire_together_and_jitter_spreads_them()

    def test_fresh_entries_are_free(self):
        assert mod.fresh_entries_never_touch_the_backing_store()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4
