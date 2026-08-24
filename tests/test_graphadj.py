from __future__ import annotations

import pytest

from store.graphadj import (
    Graph,
    a_two_hop_walk_repeats_a_fifth_of_its_steps,
    neighbours_cost_one_scan_each_way,
    one_celebrity_taxes_every_walk_through_them,
    summarise,
    the_reverse_question_without_its_index_reads_the_world,
)


class TestGraph:
    def test_a_link_writes_both_directions(self):
        graph = Graph()
        graph.link(1, 2)
        assert len(graph.edges) == 2

    def test_out_finds_the_forward_edge(self):
        graph = Graph()
        graph.link(1, 2)
        graph.link(1, 3)
        assert sorted(graph.out(1)) == [2, 3]

    def test_into_finds_the_reverse_edge(self):
        graph = Graph()
        graph.link(1, 2)
        graph.link(3, 2)
        assert sorted(graph.into(2)) == [1, 3]

    def test_into_without_index_agrees_with_into(self):
        graph = Graph()
        graph.link(1, 2)
        graph.link(3, 2)
        graph.link(2, 1)
        assert sorted(graph.into_without_index(2)) == sorted(graph.into(2))

    def test_two_hop_excludes_the_walker(self):
        graph = Graph()
        graph.link(1, 2)
        graph.link(2, 1)
        graph.link(2, 3)
        assert graph.two_hop(1) == {3}

    def test_scans_and_touched_are_counted(self):
        graph = Graph()
        graph.link(1, 2)
        graph.out(1)
        assert graph.scans == 1 and graph.touched == 1

    def test_reset_meters_zeroes_the_counters(self):
        graph = Graph()
        graph.link(1, 2)
        graph.out(1)
        graph.reset_meters()
        assert graph.scans == 0 and graph.touched == 0

    def test_an_isolated_vertex_has_no_neighbours(self):
        graph = Graph()
        graph.link(1, 2)
        assert graph.out(9) == [] and graph.into(9) == []


class TestClaims:
    @pytest.mark.parametrize(
        "claim",
        [
            neighbours_cost_one_scan_each_way,
            the_reverse_question_without_its_index_reads_the_world,
            a_two_hop_walk_repeats_a_fifth_of_its_steps,
            one_celebrity_taxes_every_walk_through_them,
        ],
    )
    def test_claim_holds(self, claim):
        assert claim() is True

    def test_summary_is_all_true(self):
        told = summarise()
        assert all(value for name, value in told.items() if name != "module")
