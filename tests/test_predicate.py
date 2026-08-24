from __future__ import annotations

import pytest

from store import predicate as mod
from store.errors import ConfigError
from store.predicate import (
    EVERYTHING,
    NOTHING,
    Both,
    Compare,
    Either,
    Range,
    Scanner,
)
from store.zonemap import Mapped


class TestRange:
    def test_an_ordinary_range_is_not_empty(self):
        assert not Range(low=1, high=5).empty

    def test_an_inverted_range_is_empty(self):
        assert Range(low=5, high=1).empty

    def test_meet_intersects(self):
        made = Range(low=0, high=10).meet(Range(low=5, high=20))
        assert (made.low, made.high) == (5, 10)

    def test_meet_of_disjoint_is_empty(self):
        assert Range(low=0, high=4).meet(Range(low=6, high=9)).empty

    def test_join_hulls(self):
        made = Range(low=0, high=2).join(Range(low=8, high=9))
        assert (made.low, made.high) == (0, 9)

    def test_join_with_empty_is_identity(self):
        made = Range(low=3, high=7).join(NOTHING)
        assert (made.low, made.high) == (3, 7)

    def test_the_constants_behave(self):
        assert NOTHING.empty and not EVERYTHING.empty


class TestCompare:
    def test_an_unknown_op_is_refused(self):
        with pytest.raises(ConfigError):
            Compare(op="~", value=1)

    def test_less_than_holds_below(self):
        assert Compare(op="<", value=5).holds(4)
        assert not Compare(op="<", value=5).holds(5)

    def test_at_most_holds_at(self):
        assert Compare(op="<=", value=5).holds(5)

    def test_greater_than_holds_above(self):
        assert Compare(op=">", value=5).holds(6)

    def test_at_least_holds_at(self):
        assert Compare(op=">=", value=5).holds(5)

    def test_equality_holds_only_at(self):
        assert Compare(op="==", value=5).holds(5)
        assert not Compare(op="==", value=5).holds(6)

    def test_inequality_holds_elsewhere(self):
        assert Compare(op="!=", value=5).holds(6)

    def test_the_possible_range_matches_the_semantics(self):
        assert Compare(op="<", value=5).possible().high == 4
        assert Compare(op=">=", value=5).possible().low == 5
        made = Compare(op="==", value=5).possible()
        assert (made.low, made.high) == (5, 5)

    def test_inequality_admits_everything(self):
        assert Compare(op="!=", value=5).possible() == EVERYTHING


class TestCombinators:
    def test_both_needs_both(self):
        made = Both(left=Compare(op=">", value=2), right=Compare(op="<", value=8))
        assert made.holds(5) and not made.holds(1) and not made.holds(9)

    def test_either_needs_one(self):
        made = Either(left=Compare(op="<", value=2), right=Compare(op=">", value=8))
        assert made.holds(1) and made.holds(9) and not made.holds(5)

    def test_both_intersects_ranges(self):
        made = Both(left=Compare(op=">=", value=3), right=Compare(op="<=", value=7))
        assert (made.possible().low, made.possible().high) == (3, 7)

    def test_either_hulls_ranges(self):
        made = Either(left=Compare(op="==", value=1), right=Compare(op="==", value=9))
        assert (made.possible().low, made.possible().high) == (1, 9)

    def test_nesting_works(self):
        made = Both(
            left=Compare(op=">=", value=0),
            right=Either(left=Compare(op="<", value=5), right=Compare(op="==", value=9)),
        )
        assert made.holds(3) and made.holds(9) and not made.holds(7)


class TestScanner:
    def build(self):
        return Scanner(mapped=Mapped.build(list(range(1000)), block_size=100))

    def test_plain_and_pushed_agree(self):
        predicate = Both(left=Compare(op=">=", value=250), right=Compare(op="<", value=350))
        assert self.build().scan_plain(predicate) == self.build().scan_pushed(predicate)

    def test_pushed_skips_blocks(self):
        predicate = Both(left=Compare(op=">=", value=250), right=Compare(op="<", value=350))
        scanner = self.build()
        scanner.scan_pushed(predicate)
        assert scanner.mapped.blocks_skipped >= 8

    def test_pushed_evaluates_less(self):
        predicate = Compare(op="==", value=500)
        plain = self.build()
        plain.scan_plain(predicate)
        pushed = self.build()
        pushed.scan_pushed(predicate)
        assert pushed.evaluated < plain.evaluated

    def test_an_empty_predicate_reads_nothing(self):
        predicate = Both(left=Compare(op=">", value=10), right=Compare(op="<", value=5))
        scanner = self.build()
        assert scanner.scan_pushed(predicate) == [] and scanner.mapped.blocks_read == 0


class TestMeasurements:
    def test_pushdown_changes_no_answer(self):
        assert mod.pushdown_never_changes_an_answer()

    def test_narrow_ands_prune(self):
        assert mod.a_narrow_and_prunes_almost_everything()

    def test_contradictions_are_free(self):
        assert mod.a_contradiction_reads_nothing_at_all()

    def test_the_or_hull_costs(self):
        assert mod.the_or_hull_is_the_price_of_simplicity()

    def test_not_equals_degrades_safely(self):
        assert mod.not_equals_pushes_nothing_and_still_answers_right()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5
