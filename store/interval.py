from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

from store.errors import ConfigError

# An interval tree for the range tombstones, measured against the list it replaces.
#
# The rangedel module keeps its spans disjoint, which makes one binary search enough. Spans
# that must be kept as written, snapshots seeing different subsets, sequence numbers
# deciding visibility per reader, cannot be merged, and a reader's coverage question
# becomes stabbing: which of n possibly overlapping intervals contain this key. The linear
# walk answers in n; the interval tree, a balanced structure keyed on midpoints with each
# node holding the intervals crossing it, answers in log n plus the matches. The
# measurements check the tree against the walk on adversarial layouts, then count the
# comparisons that separate them.


@dataclass(frozen=True)
class Span:
    """One interval, closed on the left, open on the right."""

    start: int
    stop: int
    name: int

    def __post_init__(self) -> None:
        if self.start >= self.stop:
            raise ConfigError(f"{self.start}..{self.stop} is not a span")

    def covers(self, point: int) -> bool:
        return self.start <= point < self.stop


@dataclass
class Node:
    """One tree node: a centre, the spans crossing it, subtrees left and right of it."""

    centre: int
    crossing: list[Span] = field(default_factory=list)
    left: Node | None = field(default=None)
    right: Node | None = field(default=None)


def build(spans: list[Span]) -> Node | None:
    """A balanced interval tree from any span list, overlaps welcome.

    The centre is the lower median of the endpoints. The first draft took the upper, and a
    single span [a, b) recursed on itself forever: centre landed on b, which a half open
    span does not cover, so the span went into the left subtree whole, met the same centre,
    and went left again. The termination argument for this construction is that the centre
    is covered by, or strictly separates, at least one span, and the upper median broke it
    at the smallest possible input. A guard remains for the argument being wrong again:
    any subtree that fails to shrink is absorbed into the crossing list, which costs stab
    time and never correctness, because stab checks coverage per span anyway.
    """
    if not spans:
        return None
    points = sorted({span.start for span in spans} | {span.stop for span in spans})
    centre = points[(len(points) - 1) // 2]
    crossing = [span for span in spans if span.covers(centre)]
    lefts = [span for span in spans if span.stop <= centre]
    rights = [span for span in spans if span.start > centre]
    if len(lefts) == len(spans) or len(rights) == len(spans):
        return Node(centre=centre, crossing=list(spans))
    return Node(
        centre=centre,
        crossing=crossing,
        left=build(lefts),
        right=build(rights),
    )


@dataclass
class Meter:
    """Comparison counting shared by both implementations."""

    comparisons: int = field(default=0)


def stab_tree(node: Node | None, point: int, meter: Meter) -> list[Span]:
    """Every span covering the point, via the tree."""
    found: list[Span] = []
    while node is not None:
        for span in node.crossing:
            meter.comparisons += 1
            if span.covers(point):
                found.append(span)
        meter.comparisons += 1
        node = node.left if point < node.centre else node.right
    return found


def stab_list(spans: list[Span], point: int, meter: Meter) -> list[Span]:
    """Every span covering the point, via the walk."""
    found = []
    for span in spans:
        meter.comparisons += 1
        if span.covers(point):
            found.append(span)
    return found


@functools.cache
def _spans(count: int, kind: str, seed: int = 179) -> tuple[Span, ...]:
    """Span populations with different overlap shapes."""
    source = random.Random(seed)
    made = []
    for name in range(count):
        if kind == "scattered":
            start = source.randrange(0, 100000)
            made.append(Span(start=start, stop=start + source.randrange(10, 200), name=name))
        elif kind == "nested":
            inset = name * 40
            made.append(Span(start=inset, stop=100000 - inset, name=name))
        elif kind == "same_point":
            made.append(Span(start=50000 - name, stop=50000 + name + 1, name=name))
        else:
            raise ConfigError(f"{kind} is not a span shape")
    return tuple(made)


@functools.cache
def the_tree_agrees_with_the_walk_on_every_shape() -> bool:
    """Three adversarial layouts, five hundred stabs each, identical answer sets.

    Scattered spans, fully nested spans, and spans all sharing one point: the shapes that
    break interval tree implementations differently. Nested spans overload one subtree,
    shared-point spans overload one node's crossing list, and the tree must return exactly
    the walk's set at every probe regardless.
    """
    source = random.Random(181)
    for kind in ("scattered", "nested", "same_point"):
        spans = list(_spans(400, kind))
        root = build(spans)
        for _ in range(500):
            point = source.randrange(0, 100000)
            tree_found = {span.name for span in stab_tree(root, point, Meter())}
            list_found = {span.name for span in stab_list(spans, point, Meter())}
            if tree_found != list_found:
                return False
    return True


@functools.cache
def scattered_spans_stab_in_a_fraction_of_the_comparisons() -> bool:
    """On scattered spans the tree answers in under a tenth of the walk's comparisons.

    Four hundred spans, five hundred stabs: the walk compares every span every time, two
    hundred thousand comparisons, and the tree compares the path plus the crossing lists it
    meets, under a tenth of that. This is the case range tombstones actually produce, many
    narrow spans across a wide keyspace.
    """
    source = random.Random(191)
    spans = list(_spans(400, "scattered"))
    root = build(spans)
    tree_meter, list_meter = Meter(), Meter()
    for _ in range(500):
        point = source.randrange(0, 100000)
        stab_tree(root, point, tree_meter)
        stab_list(spans, point, list_meter)
    return tree_meter.comparisons < list_meter.comparisons / 10


@functools.cache
def fully_nested_spans_are_the_trees_bad_day() -> bool:
    """When every span crosses the centre, the tree degenerates toward the walk.

    Four hundred nested spans all contain the midpoint, so the root's crossing list holds
    everything and a stab near the centre compares them all, the walk in a tree costume.
    The layout is not exotic, deletes of a shrinking prefix produce it, and the honest
    statement is that the tree's win is the scattered case, not a law.
    """
    source = random.Random(193)
    spans = list(_spans(400, "nested"))
    root = build(spans)
    tree_meter, list_meter = Meter(), Meter()
    for _ in range(200):
        point = 50000 + source.randrange(-100, 100)
        stab_tree(root, point, tree_meter)
        stab_list(spans, point, list_meter)
    return tree_meter.comparisons > list_meter.comparisons * 0.5


@functools.cache
def an_empty_tree_answers_empty() -> bool:
    """No spans, no matches, no comparisons on the crossing lists."""
    meter = Meter()
    return build([]) is None and stab_tree(None, 5, meter) == []


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "the_tree_agrees_with_the_walk": the_tree_agrees_with_the_walk_on_every_shape(),
        "scattered_stabs_are_cheap": scattered_spans_stab_in_a_fraction_of_the_comparisons(),
        "nesting_is_the_bad_day": fully_nested_spans_are_the_trees_bad_day(),
        "empty_answers_empty": an_empty_tree_answers_empty(),
    }
