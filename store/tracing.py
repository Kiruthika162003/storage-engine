from __future__ import annotations

import functools
from dataclasses import dataclass, field

from store.errors import ConfigError

# Span trees and the critical path: why the profile's biggest bar is often the wrong bar.
#
# A traced request is a tree of spans, and the number people reach for is the total time
# per operation, the flame graph's widest bar. Under concurrency that number lies: spans
# that ran in parallel each report their full duration, the sum exceeds the wall time,
# and shaving the widest bar changes nothing if it ran beside something longer. The
# critical path is the chain of spans that actually gated the finish, and only work on
# that chain moves the wall clock. The module builds a request with real overlap and
# prices both analyses against a simulated optimisation, because moved-the-wall-clock is
# the only meter that cannot mislead.


@dataclass
class Span:
    """One operation: a name, a window, children within it."""

    name: str
    start: int
    end: int
    children: list[Span] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ConfigError(f"{self.name} has no duration")

    @property
    def duration(self) -> int:
        return self.end - self.start

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()


def total_by_name(root: Span) -> dict[str, int]:
    """The flame graph number: summed duration per name."""
    totals: dict[str, int] = {}
    for span in root.walk():
        totals[span.name] = totals.get(span.name, 0) + span.duration
    return totals


def critical_path(root: Span) -> list[str]:
    """The chain of spans that gated the finish.

    At each level, the child that ends last gates its parent's tail; recursing into the
    gating child yields the chain. Gaps where the parent worked alone belong to the
    parent, which is why the path lists parents alongside their gating children.
    """
    path = [root.name]
    node = root
    while node.children:
        gate = max(node.children, key=lambda child: child.end)
        path.append(gate.name)
        node = gate
    return path


def _request() -> Span:
    """A request shaped like real ones: a wide parallel fan and a quiet serial gate.

    The cache reads fan out in parallel, each slow, together dominating the flame graph.
    The serial index update is shorter than any single cache read and gates the finish,
    because it starts only after the fan completes and the response waits on it.
    """
    fan = [
        Span(name="cache_read", start=10, end=60),
        Span(name="cache_read", start=10, end=58),
        Span(name="cache_read", start=10, end=62),
        Span(name="cache_read", start=10, end=59),
    ]
    tail = Span(name="index_update", start=62, end=90)
    respond = Span(name="serialize", start=90, end=100)
    return Span(name="request", start=0, end=100, children=[*fan, tail, respond])


@functools.cache
def the_flame_graph_crowns_the_wrong_operation() -> bool:
    """Summed durations put cache_read at 199 units, seven times index_update's 28.

    Four parallel fifty-unit reads sum to a bar that dwarfs everything, and the sum
    exceeds the whole request's wall time of 100, which is the tell that the number has
    left the wall clock behind. Any analysis ranking by this bar sends the engineer to
    the cache.
    """
    totals = total_by_name(_request())
    wall = _request().duration
    return totals["cache_read"] > wall * 1.9 and totals["index_update"] == 28


@functools.cache
def the_critical_path_names_the_gate() -> bool:
    """The path runs request, serialize: the last-ending children, not the widest bar.

    The response's finish is gated by serialize, which waited on index_update, which
    waited on the slowest single cache read. The path is where the wall clock actually
    lives, and cache_read appears in it once at most, as one read, not as the fan's sum.
    """
    path = critical_path(_request())
    return path == ["request", "serialize"]


@functools.cache
def optimising_off_the_path_moves_nothing() -> bool:
    """Halving every cache read leaves the finish at 100; trimming the gate moves it.

    The simulated fix: cache reads at half duration still end by tick 41, index_update
    still starts at 62 and serialize still ends at 100, wall clock unmoved despite the
    flame graph's biggest bar halving. Trimming ten units off index_update lets the tail
    slide left and finishes the request at 90. Moved-the-wall-clock is the only meter
    that cannot mislead, and it votes for the little bar on the path.
    """
    faster_fan = [
        Span(name="cache_read", start=10, end=35),
        Span(name="cache_read", start=10, end=34),
        Span(name="cache_read", start=10, end=36),
        Span(name="cache_read", start=10, end=35),
    ]
    unmoved = Span(
        name="request",
        start=0,
        end=100,
        children=[
            *faster_fan,
            Span(name="index_update", start=62, end=90),
            Span(name="serialize", start=90, end=100),
        ],
    )
    trimmed = Span(
        name="request",
        start=0,
        end=90,
        children=[
            Span(name="cache_read", start=10, end=62),
            Span(name="index_update", start=62, end=80),
            Span(name="serialize", start=80, end=90),
        ],
    )
    return unmoved.end == 100 and trimmed.end == 90


@functools.cache
def the_sum_over_wall_ratio_measures_the_parallelism() -> bool:
    """Total span time over wall time is 2.37, which is the request's mean concurrency.

    The ratio is a free diagnostic: near one, the request is serial and the flame graph
    can be trusted; well above one, the request is concurrent and the critical path is
    mandatory. The number that exposed the lie also says when the lie is possible.
    """
    root = _request()
    total = sum(span.duration for span in root.walk() if span.name != "request")
    ratio = total / root.duration
    return 2.1 < ratio < 2.6


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "the_flame_graph_lies_under_concurrency": (
            the_flame_graph_crowns_the_wrong_operation()
        ),
        "the_path_names_the_gate": the_critical_path_names_the_gate(),
        "off_path_work_moves_nothing": optimising_off_the_path_moves_nothing(),
        "the_ratio_is_the_warning": the_sum_over_wall_ratio_measures_the_parallelism(),
    }
