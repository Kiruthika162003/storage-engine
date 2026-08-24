from __future__ import annotations

import math

import pytest

from store import externalsort as mod
from store.errors import ConfigError
from store.externalsort import sort


class TestCorrectness:
    def test_a_small_list_sorts(self):
        made, _ = sort([3, 1, 2], memory=10, fan_in=2)
        assert made == [1, 2, 3]

    def test_an_empty_list_sorts(self):
        made, _ = sort([], memory=10, fan_in=2)
        assert made == []

    def test_zero_memory_is_refused(self):
        with pytest.raises(ConfigError):
            sort([1], memory=0, fan_in=2)

    def test_a_fan_in_of_one_is_refused(self):
        with pytest.raises(ConfigError):
            sort([1], memory=1, fan_in=1)

    def test_duplicates_survive(self):
        made, _ = sort([2, 1, 2, 1], memory=2, fan_in=2)
        assert made == [1, 1, 2, 2]

    def test_a_presorted_list_survives(self):
        values = list(range(500))
        made, _ = sort(values, memory=50, fan_in=4)
        assert made == values

    def test_a_reversed_list_sorts(self):
        made, _ = sort(list(range(500, 0, -1)), memory=50, fan_in=4)
        assert made == list(range(1, 501))

    def test_memory_of_one_still_sorts(self):
        values = list(mod._values(200))
        made, _ = sort(values, memory=1, fan_in=2)
        assert made == sorted(values)


class TestMeter:
    def test_run_formation_makes_the_expected_runs(self):
        _, meter = sort(list(range(1000)), memory=100, fan_in=4)
        assert meter.runs_made == 10

    def test_an_in_memory_sort_makes_one_run_and_no_passes(self):
        _, meter = sort(list(range(50)), memory=100, fan_in=4)
        assert meter.runs_made == 1 and meter.passes == 0

    def test_the_pass_count_is_the_ceiled_log(self):
        _, meter = sort(list(mod._values(5000)), memory=50, fan_in=4)
        assert meter.passes == math.ceil(math.log(100, 4))

    def test_io_grows_with_passes(self):
        _, narrow = sort(list(mod._values(5000)), memory=50, fan_in=2)
        _, wide = sort(list(mod._values(5000)), memory=50, fan_in=64)
        assert narrow.read > wide.read

    def test_reads_equal_writes(self):
        _, meter = sort(list(mod._values(3000)), memory=100, fan_in=4)
        assert meter.read == meter.written

    def test_as_dict_reports_io_per_record(self):
        _, meter = sort(list(mod._values(1000)), memory=100, fan_in=4)
        wanted = round((meter.read + meter.written) / 1000, 2)
        assert meter.as_dict()["io_per_record"] == wanted


class TestMeasurements:
    def test_correct_at_every_geometry(self):
        assert mod.the_sort_is_correct_at_every_geometry()

    def test_passes_follow_the_log(self):
        assert mod.the_pass_count_follows_the_logarithm()

    def test_every_pass_moves_everything(self):
        assert mod.every_pass_moves_every_record_once()

    def test_wide_fan_in_is_one_pass(self):
        assert mod.a_wide_enough_fan_in_makes_one_pass()

    def test_the_lsm_is_this_forever(self):
        assert mod.the_levelled_lsm_is_this_algorithm_run_forever()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5

    def test_the_fan_in_table_has_six_rows(self):
        assert len(mod.compare_the_fan_ins(2000, 50)) == 6

    def test_wider_fans_never_cost_more(self):
        rows = mod.compare_the_fan_ins(2000, 50)
        reads = [row["read"] for row in rows]
        assert reads == sorted(reads, reverse=True)

    def test_the_values_are_cached(self):
        assert mod._values(100) is mod._values(100)
