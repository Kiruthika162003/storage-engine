from __future__ import annotations

import pytest

from store import scrub as mod
from store.errors import BadChecksum, ConfigError
from store.scrub import Farm, run


class TestFarm:
    def test_a_zero_population_is_refused(self):
        with pytest.raises(ConfigError):
            Farm(blocks=0, rot_per_tick=0.0)

    def test_an_impossible_rot_rate_is_refused(self):
        with pytest.raises(ConfigError):
            Farm(blocks=1, rot_per_tick=1.5)

    def test_a_fresh_farm_is_intact(self):
        farm = Farm(blocks=100, rot_per_tick=0.0)
        assert farm.undetected() == 0

    def test_a_rotless_farm_stays_intact(self):
        farm = Farm(blocks=100, rot_per_tick=0.0)
        for _ in range(50):
            farm.tick()
        assert farm.undetected() == 0

    def test_rot_appears_over_time(self):
        farm = Farm(blocks=500, rot_per_tick=0.01)
        for _ in range(50):
            farm.tick()
        assert farm.undetected() > 0

    def test_a_rotted_block_fails_its_checksum(self):
        farm = Farm(blocks=10, rot_per_tick=0.0)
        farm.held[0].payload[0] ^= 0x01
        assert not farm.held[0].intact()

    def test_a_read_of_a_healthy_block_succeeds(self):
        farm = Farm(blocks=10, rot_per_tick=0.0)
        assert farm.read(2) == bytes(farm.held[2].payload)

    def test_a_read_of_a_rotted_block_raises(self):
        farm = Farm(blocks=10, rot_per_tick=0.0)
        farm.held[2].payload[0] ^= 0x01
        with pytest.raises(BadChecksum):
            farm.read(2)

    def test_a_scrub_finds_the_rot(self):
        farm = Farm(blocks=10, rot_per_tick=0.0)
        farm.held[3].payload[0] ^= 0x01
        assert farm.scrub() == 1

    def test_a_scrub_repairs_what_it_finds(self):
        farm = Farm(blocks=10, rot_per_tick=0.0)
        farm.held[3].payload[0] ^= 0x01
        farm.scrub()
        assert farm.undetected() == 0 and farm.read(3)

    def test_a_scrub_reads_every_block(self):
        farm = Farm(blocks=25, rot_per_tick=0.0)
        farm.scrub()
        assert farm.scrub_reads == 25

    def test_the_same_seed_rots_the_same_way(self):
        left = Farm(blocks=200, rot_per_tick=0.01, seed=9)
        right = Farm(blocks=200, rot_per_tick=0.01, seed=9)
        for _ in range(30):
            left.tick()
            right.tick()
        assert left.undetected() == right.undetected()


class TestRun:
    def test_a_run_reports_its_ticks(self):
        farm = Farm(blocks=50, rot_per_tick=0.001)
        assert run(farm, 20, 5)["ticks"] == 20

    def test_a_scrubbed_run_detects(self):
        farm = Farm(blocks=500, rot_per_tick=0.01)
        assert run(farm, 100, 10)["detected"] > 0

    def test_an_unscrubbed_run_detects_nothing(self):
        farm = Farm(blocks=500, rot_per_tick=0.01)
        assert run(farm, 100, 0)["detected"] == 0

    def test_the_peak_is_at_least_the_end(self):
        farm = Farm(blocks=500, rot_per_tick=0.005)
        made = run(farm, 100, 10)
        assert made["peak_undetected"] >= made["undetected_at_end"]


class TestMeasurements:
    def test_rot_scales_with_the_interval(self):
        assert mod.undetected_rot_scales_with_the_scrub_interval()

    def test_unscrubbed_rot_accumulates(self):
        assert mod.an_unscrubbed_farm_accumulates_rot_without_limit()

    def test_reads_refuse_rot(self):
        assert mod.a_foreground_read_refuses_rot_rather_than_serving_it()

    def test_scrubs_repair(self):
        assert mod.a_scrub_repairs_what_it_finds_and_the_farm_recovers()

    def test_the_price_is_reads(self):
        assert mod.scrub_reads_are_the_price_and_they_dwarf_the_finds()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5

    def test_the_interval_table_has_six_rows(self):
        assert len(mod.compare_the_intervals(100)) == 6

    def test_looser_intervals_peak_higher(self):
        rows = [row for row in mod.compare_the_intervals(400) if row["scrub_every"]]
        peaks = [row["peak_undetected"] for row in rows]
        assert peaks == sorted(peaks)

    def test_the_unscrubbed_row_peaks_highest(self):
        rows = mod.compare_the_intervals(400)
        bare = next(row for row in rows if row["scrub_every"] == 0)
        assert bare["peak_undetected"] == max(row["peak_undetected"] for row in rows)
