from __future__ import annotations

import random

import pytest

from store.errors import ConfigError
from store.eval import latency
from store.verify import metamorphic


class TestLatency:
    def test_an_impossible_utilisation_is_refused(self):
        with pytest.raises(ConfigError):
            latency.simulate(1.5)

    def test_a_full_utilisation_is_refused(self):
        with pytest.raises(ConfigError):
            latency.simulate(1.0)

    def test_the_served_count_tracks_the_load(self):
        low = latency.simulate(0.5, 50000)
        high = latency.simulate(0.9, 50000)
        assert high.served > low.served

    def test_waits_grow_with_utilisation(self):
        waits = [latency.simulate(u).mean_wait for u in (0.5, 0.8, 0.9, 0.95)]
        assert waits == sorted(waits)

    def test_the_p99_exceeds_the_median(self):
        made = latency.simulate(0.9)
        assert made.p99 > made.p50

    def test_the_peak_queue_exceeds_the_mean_wait(self):
        made = latency.simulate(0.9)
        assert made.peak_queue > made.mean_wait

    def test_the_simulation_is_cached(self):
        assert latency.simulate(0.8) is latency.simulate(0.8)

    def test_the_hockey_stick_is_real(self):
        assert latency.the_wait_doubles_between_ninety_and_ninety_five()

    def test_the_tail_tracks_the_mean(self):
        assert latency.the_tail_is_worse_than_the_mean_everywhere()

    def test_half_idle_is_unqueued(self):
        assert latency.half_idle_is_effectively_unqueued()

    def test_every_latency_claim_holds(self):
        assert all(latency.summarise().values())

    def test_the_utilisation_table_has_six_rows(self):
        assert len(latency.compare_the_utilisations()) == 6

    def test_the_table_waits_are_monotonic(self):
        rows = latency.compare_the_utilisations()
        waits = [row["mean_wait"] for row in rows]
        assert waits == sorted(waits)

    def test_poisson_draws_are_non_negative(self):
        source = random.Random(1)
        assert all(latency._poisson(source, 0.9) >= 0 for _ in range(1000))

    def test_poisson_means_are_near_the_rate(self):
        source = random.Random(2)
        draws = [latency._poisson(source, 0.9) for _ in range(20000)]
        assert 0.85 < sum(draws) / len(draws) < 0.95


class TestMetamorphic:
    def test_maintenance_is_invisible(self):
        assert metamorphic.maintenance_history_is_invisible()

    def test_a_flushed_crash_is_invisible(self):
        assert metamorphic.a_crash_after_a_flush_is_invisible()

    def test_the_live_set_is_the_store(self):
        assert metamorphic.rewriting_the_live_set_into_a_fresh_store_is_a_fixed_point()

    def test_delete_all_rewrite_restores(self):
        assert metamorphic.deleting_everything_and_rewriting_restores_the_contents()

    def test_a_scan_is_its_gets(self):
        assert metamorphic.a_scan_is_its_gets()

    def test_every_metamorphic_claim_holds(self):
        assert all(metamorphic.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(metamorphic.summarise()) == 5
