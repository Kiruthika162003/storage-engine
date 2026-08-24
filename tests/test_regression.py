from __future__ import annotations

from store.eval import regression as mod


class TestBaselines:
    def test_every_baseline_is_observed(self):
        observed = mod.observe()
        assert set(observed) == set(mod.BASELINES)

    def test_no_quantity_drifts(self):
        assert mod.drifts() == []

    def test_the_report_is_clean(self):
        made = mod.report()
        assert made["clean"] and made["drifting"] == 0

    def test_the_report_counts_the_baselines(self):
        assert mod.report()["baselines"] == len(mod.BASELINES)

    def test_tolerances_are_positive(self):
        assert all(tolerance > 0 for _, tolerance in mod.BASELINES.values())

    def test_a_planted_drift_is_named(self):
        observed = mod.observe()
        name = "compaction.levelled_amplification"
        expected, tolerance = mod.BASELINES[name]
        relative = abs(observed[name] * 2 - expected) / expected
        assert relative > tolerance

    def test_the_observation_is_cached(self):
        assert mod.observe() is mod.observe()

    def test_the_baseline_set_is_substantial(self):
        assert len(mod.BASELINES) >= 10
