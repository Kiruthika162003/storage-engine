from __future__ import annotations

import pytest

from store.eval.readpath import (
    BLOCK,
    PROBE,
    READ,
    SPACE,
    TABLES,
    Path,
    _held,
    _workload,
    a_coarse_granule_multiplies_reads,
    a_hot_cache_makes_the_filter_overhead,
    render,
    summarise,
    sweep,
    the_filter_owns_the_absent_workload,
    the_filter_pays_on_present_keys_too,
    the_organs_overlap_rather_than_add,
)


class TestPath:
    def test_build_is_deterministic(self):
        one = Path.build(19, False, 0, BLOCK)
        two = Path.build(19, False, 0, BLOCK)
        assert [sorted(t)[:5] for t in one.tables] == [sorted(t)[:5] for t in two.tables]

    def test_a_present_key_is_found(self):
        path = Path.build(19, False, 0, BLOCK)
        key = next(iter(path.tables[3]))
        assert path.get(key)

    def test_an_absent_key_is_not(self):
        path = Path.build(19, False, 0, BLOCK)
        assert not path.get(SPACE + 5)

    def test_bare_absent_read_charges_every_table(self):
        path = Path.build(19, False, 0, BLOCK)
        path.get(SPACE + 5)
        assert path.charges == TABLES * READ

    def test_filtered_absent_read_charges_only_probes(self):
        path = Path.build(19, True, 0, BLOCK)
        path.get(SPACE + 5)
        assert path.charges == TABLES * PROBE

    def test_the_cache_absorbs_a_repeat(self):
        path = Path.build(19, False, 64, BLOCK)
        key = next(iter(path.tables[0]))
        path.get(key)
        first = path.charges
        path.get(key)
        assert path.charges == first

    def test_the_cache_evicts_at_capacity(self):
        path = Path.build(19, False, 1, BLOCK)
        keys = sorted(path.tables[0])
        path.get(keys[0])
        path.get(keys[-1])
        first = path.charges
        path.get(keys[0])
        assert path.charges > first


class TestSweep:
    def test_the_sweep_covers_twelve_cells(self):
        assert len(sweep()) == 12

    def test_charges_are_positive_everywhere(self):
        assert all(row["charges"] > 0 for row in sweep())

    def test_workloads_draw_from_the_right_pools(self):
        assert all(_held(key) for key in _workload("uniform", 1, 200))
        absent = _workload("absent", 1, 200)
        assert sum(1 for key in absent if not _held(key)) > 120

    def test_render_has_a_row_per_cell(self):
        assert len(render().splitlines()) == 13


class TestClaims:
    @pytest.mark.parametrize(
        "claim",
        [
            the_filter_pays_on_present_keys_too,
            a_hot_cache_makes_the_filter_overhead,
            the_filter_owns_the_absent_workload,
            the_organs_overlap_rather_than_add,
            a_coarse_granule_multiplies_reads,
        ],
    )
    def test_claim_holds(self, claim):
        assert claim() is True

    def test_summary_is_all_true(self):
        told = summarise()
        assert all(value for name, value in told.items() if name != "module")
