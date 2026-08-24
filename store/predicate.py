from __future__ import annotations

import functools
from dataclasses import dataclass, field

from store.errors import ConfigError
from store.zonemap import Mapped
from store.zonemap import _values as _zone_values

# Predicates that describe themselves, so the storage layer can act on them.
#
# A filter written as a Python lambda is opaque: the scan must call it on every value because
# nothing can see inside. A predicate built from a small algebra, comparisons joined by and
# and or, can be asked questions before any data is read, and the question that matters to
# storage is: what value range could ever satisfy this. That range is what the zone map
# needs, so an expression like value >= 100 and value < 200 prunes blocks without the
# evaluator running once. The pushdown is exact for the algebra chosen here, conservative in
# general, and the module measures what it saves and proves it never changes an answer.


@dataclass(frozen=True)
class Range:
    """A closed interval of possible values, the currency of pushdown."""

    low: int
    high: int

    @property
    def empty(self) -> bool:
        """Whether nothing can satisfy the predicate at all."""
        return self.low > self.high

    def meet(self, other: Range) -> Range:
        """Both must hold: intersect."""
        return Range(low=max(self.low, other.low), high=min(self.high, other.high))

    def join(self, other: Range) -> Range:
        """Either may hold: the convex hull, which is where conservatism enters."""
        if self.empty:
            return other
        if other.empty:
            return self
        return Range(low=min(self.low, other.low), high=max(self.high, other.high))


EVERYTHING = Range(low=-(1 << 62), high=1 << 62)
NOTHING = Range(low=1, high=0)


@dataclass(frozen=True)
class Compare:
    """One comparison against a constant."""

    op: str
    value: int

    def __post_init__(self) -> None:
        if self.op not in ("<", "<=", ">", ">=", "==", "!="):
            raise ConfigError(f"{self.op} is not a comparison")

    def holds(self, value: int) -> bool:
        """Evaluate on one value."""
        if self.op == "<":
            return value < self.value
        if self.op == "<=":
            return value <= self.value
        if self.op == ">":
            return value > self.value
        if self.op == ">=":
            return value >= self.value
        if self.op == "==":
            return value == self.value
        return value != self.value

    def possible(self) -> Range:
        """The values that could satisfy this comparison."""
        if self.op == "<":
            return Range(low=EVERYTHING.low, high=self.value - 1)
        if self.op == "<=":
            return Range(low=EVERYTHING.low, high=self.value)
        if self.op == ">":
            return Range(low=self.value + 1, high=EVERYTHING.high)
        if self.op == ">=":
            return Range(low=self.value, high=EVERYTHING.high)
        if self.op == "==":
            return Range(low=self.value, high=self.value)
        return EVERYTHING


@dataclass(frozen=True)
class Both:
    """And: both sides must hold."""

    left: Compare | Both | Either
    right: Compare | Both | Either

    def holds(self, value: int) -> bool:
        return self.left.holds(value) and self.right.holds(value)

    def possible(self) -> Range:
        return self.left.possible().meet(self.right.possible())


@dataclass(frozen=True)
class Either:
    """Or: one side suffices."""

    left: Compare | Both | Either
    right: Compare | Both | Either

    def holds(self, value: int) -> bool:
        return self.left.holds(value) or self.right.holds(value)

    def possible(self) -> Range:
        return self.left.possible().join(self.right.possible())


@dataclass
class Scanner:
    """A mapped table scanned through a predicate, with and without pushdown."""

    mapped: Mapped
    evaluated: int = field(default=0)

    def scan_plain(self, predicate) -> list[int]:
        """Every block read, the predicate on every value."""
        found = []
        for zone in self.mapped.zones:
            self.mapped.blocks_read += 1
            for value in self.mapped.blocks[zone.block]:
                self.evaluated += 1
                if predicate.holds(value):
                    found.append(value)
        return found

    def scan_pushed(self, predicate) -> list[int]:
        """Blocks pruned by the predicate's range, the predicate on survivors only."""
        possible = predicate.possible()
        found = []
        if possible.empty:
            return found
        for zone in self.mapped.zones:
            if not zone.overlaps(possible.low, possible.high):
                self.mapped.blocks_skipped += 1
                continue
            self.mapped.blocks_read += 1
            for value in self.mapped.blocks[zone.block]:
                self.evaluated += 1
                if predicate.holds(value):
                    found.append(value)
        return found


@functools.cache
def _table() -> tuple[int, ...]:
    """The zonemap module's jittered timestamps, reused as the shared corpus."""
    return _zone_values(20000)


def _fresh_scanner() -> Scanner:
    """A scanner over a fresh mapping, so counters start clean."""
    return Scanner(mapped=Mapped.build(list(_table()), block_size=500))


@functools.cache
def pushdown_never_changes_an_answer() -> bool:
    """Nine predicates, plain against pushed, identical results every time.

    The set includes the shapes that break naive range extraction: not-equals, whose range
    is everything; or across disjoint intervals, where the hull admits values between the
    arms; and a contradiction, whose range is empty. Equality of output is the license for
    everything else in the module.
    """
    values = _table()
    low, high = min(values), max(values)
    span = high - low
    cases = [
        Compare(op=">=", value=low + span // 2),
        Compare(op="<", value=low + span // 10),
        Compare(op="==", value=values[500]),
        Compare(op="!=", value=values[500]),
        Both(
            left=Compare(op=">=", value=low + span // 4),
            right=Compare(op="<", value=low + span // 3),
        ),
        Either(
            left=Compare(op="<", value=low + span // 10),
            right=Compare(op=">", value=high - span // 10),
        ),
        Both(left=Compare(op=">", value=high), right=Compare(op="<", value=low)),
        Both(
            left=Compare(op=">=", value=low),
            right=Either(
                left=Compare(op="<", value=low + 100),
                right=Compare(op="==", value=high),
            ),
        ),
        Either(left=Compare(op="==", value=low), right=Compare(op="==", value=low)),
    ]
    for predicate in cases:
        if _fresh_scanner().scan_plain(predicate) != _fresh_scanner().scan_pushed(predicate):
            return False
    return True


@functools.cache
def a_narrow_and_prunes_almost_everything() -> bool:
    """A five percent window evaluates the predicate on six percent of the values.

    The and of two comparisons intersects to a tight range, the zone map drops every block
    outside it, and the evaluator runs only on the surviving blocks. The plain scan
    evaluated all twenty thousand values for the same answer.
    """
    values = _table()
    low, high = min(values), max(values)
    span = high - low
    predicate = Both(
        left=Compare(op=">=", value=low + span // 2),
        right=Compare(op="<", value=low + span // 2 + span // 20),
    )
    scanner = _fresh_scanner()
    scanner.scan_pushed(predicate)
    return scanner.evaluated < len(values) * 0.1


@functools.cache
def a_contradiction_reads_nothing_at_all() -> bool:
    """Greater than the maximum and less than the minimum: zero blocks, zero evaluations.

    The empty range is decided from the predicate alone, before the map is even consulted,
    which is the strongest possible pushdown: the query that cannot match costs nothing but
    the algebra.
    """
    predicate = Both(left=Compare(op=">", value=1 << 61), right=Compare(op="<", value=0))
    scanner = _fresh_scanner()
    found = scanner.scan_pushed(predicate)
    return found == [] and scanner.evaluated == 0 and scanner.mapped.blocks_read == 0


@functools.cache
def the_or_hull_is_the_price_of_simplicity() -> bool:
    """An or of the two extremes prunes nothing, though its arms are tiny.

    Each arm alone would prune 90 odd percent, and their hull spans the whole table, so the
    pushed scan reads every block. A range set instead of a single range would fix it at the
    cost of carrying a list everywhere, and the honest statement is that this module chose
    the single range and this predicate shape is what the choice costs.
    """
    values = _table()
    low, high = min(values), max(values)
    span = high - low
    wide = Either(
        left=Compare(op="<", value=low + span // 20),
        right=Compare(op=">", value=high - span // 20),
    )
    scanner = _fresh_scanner()
    scanner.scan_pushed(wide)
    return scanner.mapped.blocks_skipped == 0


@functools.cache
def not_equals_pushes_nothing_and_still_answers_right() -> bool:
    """The != range is everything, so pushdown degrades to the plain scan, correctly.

    Conservatism has to fail toward reading more, never less, and this is its purest case:
    the answer excludes one value and the pushdown excludes no blocks.
    """
    predicate = Compare(op="!=", value=_table()[100])
    pushed = _fresh_scanner()
    found = pushed.scan_pushed(predicate)
    return len(found) >= len(_table()) - 5 and pushed.mapped.blocks_skipped == 0


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "pushdown_changes_no_answer": pushdown_never_changes_an_answer(),
        "narrow_ands_prune": a_narrow_and_prunes_almost_everything(),
        "contradictions_are_free": a_contradiction_reads_nothing_at_all(),
        "the_or_hull_costs": the_or_hull_is_the_price_of_simplicity(),
        "not_equals_degrades_safely": not_equals_pushes_nothing_and_still_answers_right(),
    }
