from __future__ import annotations

import itertools

import pytest

from store import zonemap as mod
from store.errors import ConfigError
from store.zonemap import Mapped, Zone


class TestZone:
    def test_an_overlapping_range_overlaps(self):
        assert Zone(block=0, low=10, high=20).overlaps(15, 25)

    def test_a_contained_range_overlaps(self):
        assert Zone(block=0, low=10, high=20).overlaps(12, 14)

    def test_a_containing_range_overlaps(self):
        assert Zone(block=0, low=10, high=20).overlaps(0, 100)

    def test_a_range_below_misses(self):
        assert not Zone(block=0, low=10, high=20).overlaps(0, 9)

    def test_a_range_above_misses(self):
        assert not Zone(block=0, low=10, high=20).overlaps(21, 30)

    def test_a_touching_boundary_overlaps(self):
        assert Zone(block=0, low=10, high=20).overlaps(20, 30)


class TestBuild:
    def test_a_zero_block_size_is_refused(self):
        with pytest.raises(ConfigError):
            Mapped.build([1], block_size=0)

    def test_the_block_count_follows_the_size(self):
        made = Mapped.build(list(range(1000)), block_size=100)
        assert len(made.blocks) == 10

    def test_a_partial_last_block_is_kept(self):
        made = Mapped.build(list(range(105)), block_size=100)
        assert len(made.blocks) == 2 and len(made.blocks[1]) == 5

    def test_the_zones_summarise_their_blocks(self):
        made = Mapped.build([5, 1, 9, 2], block_size=2)
        assert made.zones[0].low == 1 and made.zones[0].high == 5

    def test_the_map_costs_sixteen_bytes_a_block(self):
        made = Mapped.build(list(range(1000)), block_size=100)
        assert made.map_bytes == 160


class TestQuery:
    def test_a_query_finds_its_values(self):
        made = Mapped.build(list(range(100)), block_size=10)
        assert sorted(made.query(15, 25)) == list(range(15, 26))

    def test_the_range_is_closed_on_both_ends(self):
        made = Mapped.build([10, 20, 30], block_size=3)
        assert sorted(made.query(10, 30)) == [10, 20, 30]

    def test_a_miss_returns_nothing(self):
        made = Mapped.build(list(range(100)), block_size=10)
        assert made.query(500, 600) == []

    def test_sorted_data_prunes(self):
        made = Mapped.build(list(range(1000)), block_size=100)
        made.query(0, 50)
        assert made.blocks_skipped == 9

    def test_the_answer_matches_a_plain_filter(self):
        values = list(mod._values(3000))
        made = Mapped.build(values, block_size=100)
        wanted = sorted(value for value in values if 1000 <= value <= 2000)
        assert sorted(made.query(1000, 2000)) == wanted

    def test_counters_accumulate_across_queries(self):
        made = Mapped.build(list(range(1000)), block_size=100)
        made.query(0, 10)
        made.query(0, 10)
        assert made.blocks_read == 2


class TestMeasurements:
    def test_sorted_skips_nearly_all(self):
        assert mod.sorted_layout_skips_ninety_nine_percent()

    def test_shuffled_skips_nothing(self):
        assert mod.shuffled_layout_skips_nothing()

    def test_answers_agree_across_layouts(self):
        assert mod.the_answers_agree_across_layouts()

    def test_empty_ranges_are_free(self):
        assert mod.an_empty_range_reads_nothing_on_any_layout()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4

    def test_the_layout_table_has_two_rows(self):
        rows = mod.compare_the_layouts(20)
        assert [row["layout"] for row in rows] == ["sorted", "shuffled"]

    def test_the_map_costs_the_same_either_way(self):
        rows = mod.compare_the_layouts(20)
        assert rows[0]["map_bytes"] == rows[1]["map_bytes"]

    def test_only_the_sorted_row_skips(self):
        rows = {row["layout"]: row for row in mod.compare_the_layouts(20)}
        assert rows["sorted"]["skip_rate"] > 0.9 > rows["shuffled"]["skip_rate"]

    def test_the_values_are_cached(self):
        assert mod._values(100) is mod._values(100)

    def test_the_jittered_values_are_nearly_sorted(self):
        values = mod._values(1000)
        inversions = sum(1 for a, b in itertools.pairwise(values) if a > b)
        assert inversions < len(values) * 0.5
