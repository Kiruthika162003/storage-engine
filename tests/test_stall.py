from __future__ import annotations

import pytest

from store import stall as mod
from store.errors import ConfigError
from store.stall import Meter, Simulation


class TestSimulation:
    def test_a_zero_rate_is_refused(self):
        with pytest.raises(ConfigError):
            Simulation(arrival=0.0, retire=1.0)

    def test_inverted_thresholds_are_refused(self):
        with pytest.raises(ConfigError):
            Simulation(arrival=1.0, retire=1.0, slow_at=20, stop_at=8)

    def test_a_run_counts_its_ticks(self):
        assert Simulation(arrival=0.5, retire=1.0).run(100).ticks == 100

    def test_an_idle_store_accepts_everything(self):
        made = Simulation(arrival=0.5, retire=1.0).run(1000)
        assert made.accepted == 500

    def test_throughput_is_accepted_over_ticks(self):
        made = Simulation(arrival=0.5, retire=1.0).run(1000)
        assert made.throughput == 0.5

    def test_debt_never_goes_negative(self):
        made = Simulation(arrival=0.1, retire=5.0).run(500)
        assert made.end_debt == 0

    def test_a_healthy_run_never_slows(self):
        made = Simulation(arrival=0.9, retire=1.0).run(2000)
        assert made.slowed == 0 and made.stopped == 0

    def test_an_overloaded_run_slows(self):
        made = Simulation(arrival=1.5, retire=1.0).run(2000)
        assert made.slowed > 0

    def test_a_flood_hits_the_stop(self):
        made = Simulation(arrival=10.0, retire=1.0).run(2000)
        assert made.stopped > 0

    def test_the_peak_debt_is_recorded(self):
        made = Simulation(arrival=10.0, retire=1.0).run(2000)
        assert made.peak_debt >= made.end_debt

    def test_as_dict_carries_every_field(self):
        made = Simulation(arrival=1.0, retire=1.0).run(10)
        assert {"ticks", "accepted", "slowed", "stopped", "throughput"} <= set(made.as_dict())

    def test_the_meter_survives_zero_ticks(self):
        made = Meter(ticks=0, accepted=0, slowed=0, stopped=0, peak_debt=0, end_debt=0)
        assert made.throughput == 0.0


class TestMeasurements:
    def test_healthy_stores_never_wait(self):
        assert mod.a_sustainable_rate_never_touches_the_thresholds()

    def test_bursts_are_absorbed(self):
        assert mod.a_burst_is_absorbed_and_paid_back()

    def test_the_stop_boundary_is_sharp(self):
        assert mod.the_stop_appears_exactly_where_slowing_stops_sufficing()

    def test_band_width_needs_variance(self):
        assert mod.the_slow_bands_width_does_nothing_in_a_deterministic_model()

    def test_the_debt_is_bounded(self):
        assert mod.the_peak_debt_is_bounded_by_the_stop_threshold()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5

    def test_the_rate_table_has_six_rows(self):
        assert len(mod.compare_the_arrival_rates()) == 6

    def test_throughput_saturates_at_the_retire_rate(self):
        rows = mod.compare_the_arrival_rates()
        assert all(row["throughput"] <= 1.1 for row in rows)

    def test_underloaded_rows_never_stop(self):
        rows = mod.compare_the_arrival_rates()
        assert all(row["stopped"] == 0 for row in rows if row["arrival"] < 1.0)
