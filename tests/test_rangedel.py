from __future__ import annotations

import pytest

from store import rangedel as mod
from store.errors import ConfigError
from store.rangedel import Ranged, Span, Spans


class TestSpan:
    def test_a_backwards_range_is_refused(self):
        with pytest.raises(ConfigError):
            Span(start=b"b", stop=b"a", sequence=1)

    def test_an_empty_range_is_refused(self):
        with pytest.raises(ConfigError):
            Span(start=b"a", stop=b"a", sequence=1)

    def test_the_start_is_covered(self):
        assert Span(start=b"a", stop=b"c", sequence=1).covers(b"a")

    def test_the_stop_is_not_covered(self):
        assert not Span(start=b"a", stop=b"c", sequence=1).covers(b"c")

    def test_the_middle_is_covered(self):
        assert Span(start=b"a", stop=b"c", sequence=1).covers(b"b")


class TestSpans:
    def test_an_added_span_covers(self):
        made = Spans()
        made.add(b"a", b"c", 1)
        assert made.covering(b"b") is not None

    def test_an_uncovered_key_finds_nothing(self):
        made = Spans()
        made.add(b"a", b"c", 1)
        assert made.covering(b"z") is None

    def test_disjoint_spans_both_cover(self):
        made = Spans()
        made.add(b"a", b"c", 1)
        made.add(b"x", b"z", 2)
        assert made.covering(b"b") and made.covering(b"y") and len(made) == 2

    def test_overlapping_spans_merge(self):
        made = Spans()
        made.add(b"a", b"m", 1)
        made.add(b"h", b"z", 2)
        assert len(made) == 1

    def test_a_merge_takes_the_widest_bounds(self):
        made = Spans()
        made.add(b"d", b"h", 1)
        made.add(b"a", b"z", 2)
        span = made.covering(b"e")
        assert span.start == b"a" and span.stop == b"z"

    def test_a_merge_keeps_the_newest_sequence(self):
        made = Spans()
        made.add(b"a", b"m", 5)
        made.add(b"h", b"z", 2)
        assert made.covering(b"i").sequence == 5

    def test_the_held_spans_stay_sorted(self):
        made = Spans()
        made.add(b"t", b"v", 1)
        made.add(b"a", b"c", 2)
        made.add(b"j", b"l", 3)
        starts = [span.start for span in made.held]
        assert starts == sorted(starts)

    def test_coverage_counts_checks(self):
        made = Spans()
        made.add(b"a", b"c", 1)
        made.covering(b"b")
        made.covering(b"z")
        assert made.checks == 2


class TestRanged:
    def test_a_put_reads_back(self):
        made = Ranged()
        made.put(b"k", b"v")
        assert made.get(b"k") == b"v"

    def test_a_point_delete_hides_the_key(self):
        made = Ranged()
        made.put(b"k", b"v")
        made.delete(b"k")
        assert made.get(b"k") is None

    def test_a_range_delete_hides_the_covered(self):
        made = Ranged()
        made.put(b"a1", b"v")
        made.put(b"b1", b"v")
        made.delete_range(b"a", b"b")
        assert made.get(b"a1") is None and made.get(b"b1") == b"v"

    def test_a_range_delete_is_one_write(self):
        made = Ranged()
        for at in range(100):
            made.put(f"a{at:03d}".encode(), b"v")
        made.delete_range(b"a", b"b")
        assert made.range_writes == 1

    def test_a_later_write_survives_the_range(self):
        made = Ranged()
        made.delete_range(b"a", b"z")
        made.put(b"m", b"new")
        assert made.get(b"m") == b"new"

    def test_an_earlier_write_does_not(self):
        made = Ranged()
        made.put(b"m", b"old")
        made.delete_range(b"a", b"z")
        assert made.get(b"m") is None

    def test_keys_applies_the_spans(self):
        made = Ranged()
        made.put(b"a1", b"v")
        made.put(b"x1", b"v")
        made.delete_range(b"a", b"b")
        assert made.keys() == [b"x1"]

    def test_a_missing_key_is_absent(self):
        assert Ranged().get(b"k") is None


class TestMeasurements:
    def test_one_write_per_drop(self):
        assert mod.dropping_a_tenant_is_one_write_instead_of_five_hundred()

    def test_reads_pay_the_check(self):
        assert mod.the_read_side_pays_one_span_check_per_read()

    def test_overlaps_merge(self):
        assert mod.overlapping_ranges_merge_into_one_span()

    def test_later_writes_survive(self):
        assert mod.a_write_after_the_delete_survives_it()

    def test_disjoint_stays_disjoint(self):
        assert mod.disjoint_ranges_stay_separate()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5

    def test_the_cost_table_has_two_rows(self):
        rows = mod.compare_the_drop_costs(100)
        assert [row["shape"] for row in rows] == ["point", "range"]

    def test_the_point_shape_writes_per_row(self):
        rows = mod.compare_the_drop_costs(100)
        assert rows[0]["writes"] == rows[0]["rows"]

    def test_the_range_shape_writes_once(self):
        rows = mod.compare_the_drop_costs(100)
        assert rows[1]["writes"] == 1

    def test_the_tenanted_store_is_cached(self):
        assert mod._tenanted(2, 10, 1) is mod._tenanted(2, 10, 1)
