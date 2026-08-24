from __future__ import annotations

import random

import pytest

from store import interval as mod
from store.errors import ConfigError
from store.interval import Meter, Span, build, stab_list, stab_tree


class TestSpan:
    def test_a_backwards_span_is_refused(self):
        with pytest.raises(ConfigError):
            Span(start=5, stop=3, name=0)

    def test_an_empty_span_is_refused(self):
        with pytest.raises(ConfigError):
            Span(start=5, stop=5, name=0)

    def test_the_start_is_covered(self):
        assert Span(start=3, stop=7, name=0).covers(3)

    def test_the_stop_is_not_covered(self):
        assert not Span(start=3, stop=7, name=0).covers(7)


class TestBuild:
    def test_no_spans_builds_nothing(self):
        assert build([]) is None

    def test_a_single_span_terminates_and_answers(self):
        root = build([Span(start=3, stop=7, name=0)])
        assert len(stab_tree(root, 5, Meter())) == 1

    def test_two_identical_spans_terminate(self):
        spans = [Span(start=3, stop=7, name=0), Span(start=3, stop=7, name=1)]
        root = build(spans)
        assert len(stab_tree(root, 5, Meter())) == 2

    def test_touching_spans_terminate(self):
        spans = [Span(start=0, stop=5, name=0), Span(start=5, stop=10, name=1)]
        root = build(spans)
        assert {span.name for span in stab_tree(root, 5, Meter())} == {1}

    def test_a_deep_scattered_set_builds(self):
        spans = list(mod._spans(400, "scattered"))
        assert build(spans) is not None


class TestStab:
    def probe_both(self, spans, point):
        root = build(spans)
        tree = {span.name for span in stab_tree(root, point, Meter())}
        walk = {span.name for span in stab_list(spans, point, Meter())}
        return tree, walk

    def test_a_covered_point_is_found(self):
        tree, walk = self.probe_both([Span(start=0, stop=10, name=0)], 5)
        assert tree == walk == {0}

    def test_an_uncovered_point_is_not(self):
        tree, walk = self.probe_both([Span(start=0, stop=10, name=0)], 50)
        assert tree == walk == set()

    def test_overlapping_spans_all_answer(self):
        spans = [Span(start=0, stop=10, name=0), Span(start=5, stop=15, name=1)]
        tree, walk = self.probe_both(spans, 7)
        assert tree == walk == {0, 1}

    def test_random_agreement_holds(self):
        source = random.Random(7)
        spans = list(mod._spans(150, "scattered", seed=8))
        root = build(spans)
        for _ in range(300):
            point = source.randrange(0, 100000)
            tree = {span.name for span in stab_tree(root, point, Meter())}
            walk = {span.name for span in stab_list(spans, point, Meter())}
            assert tree == walk

    def test_the_meter_counts(self):
        spans = [Span(start=0, stop=10, name=0)]
        meter = Meter()
        stab_list(spans, 5, meter)
        assert meter.comparisons == 1


class TestMeasurements:
    def test_the_tree_agrees_with_the_walk(self):
        assert mod.the_tree_agrees_with_the_walk_on_every_shape()

    def test_scattered_stabs_are_cheap(self):
        assert mod.scattered_spans_stab_in_a_fraction_of_the_comparisons()

    def test_nesting_is_the_bad_day(self):
        assert mod.fully_nested_spans_are_the_trees_bad_day()

    def test_empty_answers_empty(self):
        assert mod.an_empty_tree_answers_empty()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4

    def test_an_unknown_shape_is_refused(self):
        with pytest.raises(ConfigError):
            mod._spans(10, "spiral")
