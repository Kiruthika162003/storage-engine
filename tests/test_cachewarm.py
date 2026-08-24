from __future__ import annotations

from store import cachewarm as mod
from store.cache import Recent
from store.cachewarm import Ramp, run, save_keys, warm


class TestRamp:
    def test_windows_close_at_the_size(self):
        ramp = Ramp(window=2)
        ramp.note(True)
        ramp.note(False)
        assert ramp.trajectory == [0.5]

    def test_partial_windows_stay_open(self):
        ramp = Ramp(window=3)
        ramp.note(True)
        assert ramp.trajectory == []

    def test_the_trajectory_accumulates(self):
        ramp = Ramp(window=1)
        for hit in (True, False, True):
            ramp.note(hit)
        assert ramp.trajectory == [1.0, 0.0, 1.0]


class TestWarm:
    def test_save_keys_reports_the_residents(self):
        cache = Recent(capacity=4)
        cache.put(1, b"a")
        cache.put(2, b"b")
        assert sorted(save_keys(cache)) == [1, 2]

    def test_warm_loads_the_saved_set(self):
        cache = Recent(capacity=4)
        loaded = warm(cache, [7, 8])
        assert loaded == 2 and cache.get(7) is not None

    def test_run_fills_and_records(self):
        cache = Recent(capacity=8)
        ramp = run(cache, [1, 1, 2], Ramp(window=1))
        assert ramp.trajectory == [0.0, 1.0, 0.0]


class TestMeasurements:
    def test_warmup_skips_the_ramp(self):
        assert mod.a_warmed_cache_skips_the_ramp()

    def test_the_plateau_is_the_workload(self):
        assert mod.the_cold_ramp_ends_at_the_same_plateau()

    def test_shifted_sets_void_the_bet(self):
        assert mod.a_shifted_working_set_makes_warmup_worthless()

    def test_the_cost_is_the_capacity(self):
        assert mod.the_warmup_cost_is_the_capacity()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4

    def test_the_stream_is_cached(self):
        assert mod._hot_stream(100) is mod._hot_stream(100)
