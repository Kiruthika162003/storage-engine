from __future__ import annotations

import random

import pytest

from store import prefetch as mod
from store.errors import ConfigError
from store.prefetch import Prefetcher, drive


class TestDetector:
    def test_a_trigger_below_two_is_refused(self):
        with pytest.raises(ConfigError):
            Prefetcher(trigger=1)

    def test_a_zero_window_is_refused(self):
        with pytest.raises(ConfigError):
            Prefetcher(max_window=0)

    def test_the_first_read_is_demand(self):
        assert Prefetcher().read(0) == "demand"

    def test_two_consecutive_reads_trigger_prefetch(self):
        made = Prefetcher()
        made.read(0)
        made.read(1)
        assert made.prefetches > 0

    def test_the_third_read_is_served_ahead(self):
        made = Prefetcher()
        made.read(0)
        made.read(1)
        assert made.read(2) == "ahead"

    def test_non_consecutive_reads_never_trigger(self):
        made = Prefetcher()
        made.read(0)
        made.read(5)
        made.read(11)
        assert made.prefetches == 0

    def test_the_window_starts_at_one(self):
        assert Prefetcher().window == 1

    def test_the_window_respects_its_cap(self):
        made = Prefetcher(max_window=4)
        for number in range(50):
            made.read(number)
        assert made.window == 4

    def test_a_jump_resets_the_window(self):
        made = Prefetcher()
        for number in range(10):
            made.read(number)
        made.read(999)
        assert made.window == 1


class TestAccounting:
    def test_coverage_is_served_over_reads(self):
        made = drive(Prefetcher(), range(100))
        assert made["coverage"] == round(made["served_ahead"] / made["reads"], 4)

    def test_waste_counts_the_stranded(self):
        made = Prefetcher()
        for number in range(10):
            made.read(number)
        assert made.wasted > 0

    def test_a_consumed_prefetch_is_not_waste(self):
        made = Prefetcher(max_window=2)
        for number in range(50):
            made.read(number)
        assert made.wasted <= 2

    def test_as_dict_adds_up(self):
        made = drive(Prefetcher(), range(200))
        assert made["demand"] + made["served_ahead"] == made["reads"]


class TestMeasurements:
    def test_scans_are_served_ahead(self):
        assert mod.a_long_scan_is_almost_entirely_served_ahead()

    def test_random_readers_cost_nothing(self):
        assert mod.a_random_reader_gets_no_help_and_causes_no_flood()

    def test_the_window_doubles_and_caps(self):
        assert mod.the_window_doubles_and_caps()

    def test_breaks_reset_spending_not_purchases(self):
        assert mod.a_break_resets_the_spending_but_not_the_bought_blocks()

    def test_interleaving_is_the_limit(self):
        assert mod.interleaved_scans_defeat_a_single_run_detector()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5

    def test_the_pattern_table_has_three_rows(self):
        rows = mod.compare_the_patterns()
        assert [row["pattern"] for row in rows] == ["scan", "random", "interleaved"]

    def test_only_the_scan_row_has_coverage(self):
        rows = {row["pattern"]: row for row in mod.compare_the_patterns()}
        assert rows["scan"]["coverage"] > 0.9
        assert rows["random"]["coverage"] < 0.01
        assert rows["interleaved"]["coverage"] < 0.01

    def test_a_seeded_random_stream_is_reproducible(self):
        source = random.Random(1)
        blocks = [source.randrange(1000) for _ in range(500)]
        assert drive(Prefetcher(), blocks) == drive(Prefetcher(), list(blocks))
