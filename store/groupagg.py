from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

from store.errors import ConfigError

# Group-by in one pass: the join module's memory argument, run again for aggregation.
#
# Sum the orders per customer. The hash aggregate keeps a running total per group, any input
# order, memory proportional to the group count. The stream aggregate demands input sorted
# by group and holds exactly one accumulator, emitting each group as the key changes. The
# same separation as the joins, with one addition worth measuring: the stream aggregate
# emits groups incrementally, so its first result arrives after one group's rows, while the
# hash aggregate emits nothing until it has seen the last row, which is the difference
# between a pipeline and a barrier wearing aggregation clothes.


@dataclass
class Meter:
    """Held groups and emission latency in rows."""

    held_groups: int = field(default=0)
    first_emit_after: int = field(default=-1)


def hash_aggregate(rows: list[tuple[int, int]], meter: Meter) -> list[tuple[int, int]]:
    """Running totals per group, emitted at the end, sorted for comparison."""
    totals: dict[int, int] = {}
    for at, (group, amount) in enumerate(rows):
        totals[group] = totals.get(group, 0) + amount
        del at
    meter.held_groups = len(totals)
    meter.first_emit_after = len(rows)
    return sorted(totals.items())


def stream_aggregate(rows: list[tuple[int, int]], meter: Meter) -> list[tuple[int, int]]:
    """One accumulator over sorted input, groups emitted as they close."""
    if [group for group, _ in rows] != sorted(group for group, _ in rows):
        raise ConfigError("the stream aggregate needs input sorted by group")
    found: list[tuple[int, int]] = []
    current: int | None = None
    total = 0
    for at, (group, amount) in enumerate(rows):
        if group != current:
            if current is not None:
                found.append((current, total))
                if meter.first_emit_after < 0:
                    meter.first_emit_after = at
            current = group
            total = 0
        total += amount
    if current is not None:
        found.append((current, total))
        if meter.first_emit_after < 0:
            meter.first_emit_after = len(rows)
    meter.held_groups = 1 if rows else 0
    return found


@functools.cache
def _orders(count: int = 30000, groups: int = 1500, seed: int = 251):
    """Order amounts by customer, sorted by customer, as the store scans them."""
    source = random.Random(seed)
    made = sorted(
        (source.randrange(groups), source.randrange(1, 500)) for _ in range(count)
    )
    return tuple(made)


@functools.cache
def both_aggregates_agree() -> bool:
    """Thirty thousand rows, fifteen hundred groups, identical totals.

    The differential license once more, including the empty input, one group, and one row
    per group, the shapes where the emit-on-change logic slips a group or doubles one.
    """
    rows = list(_orders())
    if hash_aggregate(rows, Meter()) != stream_aggregate(rows, Meter()):
        return False
    for edge in ([], [(1, 5)], [(1, 5), (2, 6), (3, 7)]):
        if hash_aggregate(list(edge), Meter()) != stream_aggregate(list(edge), Meter()):
            return False
    return True


@functools.cache
def the_stream_holds_one_group_and_the_hash_holds_all() -> bool:
    """1,500 accumulators against one.

    The joins module's window argument in aggregate form, and the same fine print applies:
    the window is per group, and grouping is by the sort key, so this only prices as shown
    when the store's sort order and the query's group key agree, which is a schema design
    decision wearing a query plan.
    """
    rows = list(_orders())
    hash_meter, stream_meter = Meter(), Meter()
    hash_aggregate(rows, hash_meter)
    stream_aggregate(rows, stream_meter)
    return hash_meter.held_groups == 1500 and stream_meter.held_groups == 1


@functools.cache
def the_stream_emits_early_and_the_hash_emits_at_the_end() -> bool:
    """First result after 22 rows against after all 30,000.

    The stream aggregate closes its first group as soon as the key changes, so a consumer
    starts receiving results while the scan is still running. The hash aggregate is a
    barrier: nothing until everything. In a pipeline of operators the difference compounds,
    because a barrier anywhere is a barrier everywhere downstream.
    """
    rows = list(_orders())
    hash_meter, stream_meter = Meter(), Meter()
    hash_aggregate(rows, hash_meter)
    stream_aggregate(rows, stream_meter)
    return stream_meter.first_emit_after < 100 and hash_meter.first_emit_after == 30000


@functools.cache
def unsorted_input_is_refused_not_misgrouped() -> bool:
    """Shuffled input raises, because the alternative is splitting groups silently.

    A stream aggregate over unsorted input does not crash; it emits the same group several
    times with partial totals, and downstream code that assumes one row per group adds them
    or takes the last one, both quietly wrong. The refusal converts that into a doorstep
    error, the merge join's discipline repeated because the failure shape is repeated.
    """
    rows = list(_orders(2000, 100))
    shuffled = list(rows)
    random.Random(1).shuffle(shuffled)
    try:
        stream_aggregate(shuffled, Meter())
    except ConfigError:
        return True
    return False


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "the_aggregates_agree": both_aggregates_agree(),
        "one_group_against_all": the_stream_holds_one_group_and_the_hash_holds_all(),
        "pipelines_against_barriers": the_stream_emits_early_and_the_hash_emits_at_the_end(),
        "unsorted_is_refused": unsorted_input_is_refused_not_misgrouped(),
    }
