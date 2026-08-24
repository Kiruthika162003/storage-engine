from __future__ import annotations

import pytest

from store import tracing as mod
from store.errors import ConfigError
from store.tracing import Span, critical_path, total_by_name


def simple() -> Span:
    return Span(
        name="root",
        start=0,
        end=10,
        children=[
            Span(name="a", start=1, end=4),
            Span(name="b", start=4, end=9),
        ],
    )


class TestSpan:
    def test_a_zero_duration_span_is_refused(self):
        with pytest.raises(ConfigError):
            Span(name="x", start=5, end=5)

    def test_the_duration_is_the_window(self):
        assert Span(name="x", start=3, end=9).duration == 6

    def test_walk_visits_everything(self):
        assert len(list(simple().walk())) == 3


class TestTotals:
    def test_totals_sum_per_name(self):
        totals = total_by_name(simple())
        assert totals == {"root": 10, "a": 3, "b": 5}

    def test_same_named_spans_accumulate(self):
        root = Span(
            name="root",
            start=0,
            end=10,
            children=[
                Span(name="io", start=0, end=4),
                Span(name="io", start=0, end=5),
            ],
        )
        assert total_by_name(root)["io"] == 9

    def test_parallel_sums_exceed_the_wall(self):
        root = mod._request()
        totals = total_by_name(root)
        assert totals["cache_read"] > root.duration


class TestCriticalPath:
    def test_a_leaf_is_its_own_path(self):
        assert critical_path(Span(name="x", start=0, end=5)) == ["x"]

    def test_the_last_ending_child_gates(self):
        assert critical_path(simple()) == ["root", "b"]

    def test_the_path_recurses(self):
        root = Span(
            name="root",
            start=0,
            end=20,
            children=[
                Span(
                    name="middle",
                    start=0,
                    end=18,
                    children=[Span(name="inner", start=10, end=17)],
                ),
                Span(name="early", start=0, end=5),
            ],
        )
        assert critical_path(root) == ["root", "middle", "inner"]

    def test_the_request_path_ends_at_serialize(self):
        assert critical_path(mod._request())[-1] == "serialize"


class TestMeasurements:
    def test_the_flame_graph_lies_under_concurrency(self):
        assert mod.the_flame_graph_crowns_the_wrong_operation()

    def test_the_path_names_the_gate(self):
        assert mod.the_critical_path_names_the_gate()

    def test_off_path_work_moves_nothing(self):
        assert mod.optimising_off_the_path_moves_nothing()

    def test_the_ratio_is_the_warning(self):
        assert mod.the_sum_over_wall_ratio_measures_the_parallelism()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4
