from __future__ import annotations

from store.eval import recovery as mod


class TestReplayCosts:
    def test_costs_are_sampled(self):
        assert len(mod._replay_cost(200, 2000)) == 40

    def test_no_cost_exceeds_the_threshold(self):
        assert max(mod._replay_cost(200, 2000)) <= 200

    def test_costs_are_never_negative(self):
        assert min(mod._replay_cost(200, 2000)) >= 0

    def test_the_costs_are_cached(self):
        assert mod._replay_cost(200, 2000) is mod._replay_cost(200, 2000)


class TestMeasurements:
    def test_replay_averages_half(self):
        assert mod.replay_averages_half_the_threshold()

    def test_the_trade_is_linear(self):
        assert mod.a_tighter_threshold_buys_faster_recovery_with_more_flushes()

    def test_the_worst_case_is_the_threshold(self):
        assert mod.recovery_is_bounded_by_the_threshold_always()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_three_claims(self):
        assert len(mod.summarise()) == 3

    def test_the_threshold_table_has_five_rows(self):
        assert len(mod.compare_the_thresholds(2000)) == 5

    def test_the_aliased_row_reads_zero(self):
        rows = {row["flush_at"]: row for row in mod.compare_the_thresholds(2000)}
        assert rows[50]["mean_replay"] == 0.0

    def test_the_unaliased_rows_grow_with_the_threshold(self):
        rows = [row for row in mod.compare_the_thresholds(4000) if row["flush_at"] > 50]
        means = [row["mean_replay"] for row in rows]
        assert means == sorted(means)

    def test_worst_cases_track_the_threshold(self):
        rows = [row for row in mod.compare_the_thresholds(4000) if row["flush_at"] > 50]
        assert all(row["worst_replay"] <= row["flush_at"] for row in rows)
