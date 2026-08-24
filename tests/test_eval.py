from __future__ import annotations

import pytest

from store.errors import ConfigError
from store.eval import findings, run, scaling, workload
from store.eval.workload import MIXES, Mix


class TestWorkload:
    def test_every_mix_blends_to_one(self):
        for mix in MIXES:
            assert abs(mix.gets + mix.puts + mix.deletes + mix.scans - 1.0) < 1e-9

    def test_a_bad_blend_is_refused(self):
        with pytest.raises(ConfigError):
            Mix(name="x", gets=0.5, puts=0.1, deletes=0.0, scans=0.0, keys=1, operations=1)

    def test_a_stream_has_the_right_length(self):
        assert len(workload.stream("balanced")) == 20000

    def test_a_stream_is_cached(self):
        assert workload.stream("balanced") is workload.stream("balanced")

    def test_an_unknown_mix_is_refused(self):
        with pytest.raises(ConfigError):
            workload.stream("spiral")

    def test_the_blend_is_roughly_respected(self):
        made = workload.stream("read_heavy")
        gets = sum(1 for one in made if one.kind == "get")
        assert 0.9 < gets / len(made) < 1.0

    def test_puts_carry_values(self):
        made = workload.stream("balanced")
        assert all(one.value for one in made if one.kind == "put")

    def test_scans_carry_lengths(self):
        made = workload.stream("scan_heavy")
        assert all(one.length == 20 for one in made if one.kind == "scan")

    def test_hot_reads_concentrate(self):
        made = workload.stream("hot_reads")
        hot = sum(1 for one in made if int(one.key[1:]) < 250)
        assert hot > len(made) * 0.8

    def test_as_dict_names_the_mix(self):
        assert MIXES[0].as_dict()["name"] == "read_heavy"

    def test_the_names_are_distinct(self):
        names = [mix.name for mix in MIXES]
        assert len(names) == len(set(names))


class TestRun:
    def test_the_meter_counts_every_operation(self):
        meter = run.run("balanced")
        assert meter.operations == 20000

    def test_the_kinds_add_up(self):
        meter = run.run("balanced")
        assert meter.gets + meter.puts + meter.deletes + meter.scans == meter.operations

    def test_hits_and_misses_add_up(self):
        meter = run.run("read_heavy")
        assert meter.hits + meter.misses == meter.gets

    def test_the_hit_rate_is_a_fraction(self):
        assert 0.0 <= run.run("balanced").hit_rate <= 1.0

    def test_runs_are_cached(self):
        assert run.run("balanced") is run.run("balanced")

    def test_the_table_has_a_row_per_mix(self):
        assert len(run.table()) == len(MIXES)

    def test_scan_records_are_counted(self):
        assert run.run("scan_heavy").scan_records > 0

    def test_as_dict_names_the_mix(self):
        assert run.run("balanced").as_dict()["mix"] == "balanced"


class TestFindings:
    def test_read_heavy_barely_writes(self):
        assert findings.a_read_heavy_mix_barely_exercises_the_write_path()

    def test_insert_heavy_is_maintenance(self):
        assert findings.an_insert_mix_is_all_maintenance()

    def test_hit_rate_is_correlation(self):
        assert findings.hot_reads_hit_two_thirds_and_uniform_reads_one_tenth()

    def test_scans_keep_tables_alive(self):
        assert findings.scans_keep_more_tables_alive()

    def test_every_claim_holds(self):
        assert all(findings.summarise().values())

    def test_everything_carries_both_parts(self):
        made = findings.everything()
        assert set(made) == {"mixes", "claims"}


class TestScaling:
    def test_flushes_outrun_the_volume(self):
        assert scaling.flushes_grow_faster_than_the_volume_because_dedup_fades()

    def test_folds_converge_to_a_third(self):
        assert scaling.folds_converge_to_a_third_of_flushes_not_a_quarter()

    def test_misses_stay_cheap(self):
        assert scaling.a_miss_stays_cheap_at_every_size()

    def test_every_claim_holds(self):
        assert all(scaling.summarise().values())

    def test_the_table_has_a_row_per_size(self):
        assert len(scaling.table()) == len(scaling.SIZES)

    def test_flushes_rise_with_size(self):
        rows = scaling.table()
        flushes = [row["flushes"] for row in rows]
        assert flushes == sorted(flushes)

    def test_tables_stay_bounded(self):
        assert all(row["tables"] <= 4 for row in scaling.table())

    def test_measures_are_cached(self):
        assert scaling.measure(2000) is scaling.measure(2000)
