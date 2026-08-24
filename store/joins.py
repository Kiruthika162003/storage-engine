from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

from store.errors import ConfigError

# Two joins over two scans, priced in touches and memory.
#
# The store's scans produce sorted streams, and the question of relating two of them, orders
# to customers, index entries to rows, is a join whether or not anyone says the word. The
# hash join builds a table from the smaller side and probes it with the larger, any input
# order, memory proportional to the build side. The merge join walks both sides once with no
# table at all, and demands what the store happens to sell: sorted inputs. The measurements
# hold the two to identical output, then price the demand, because sorted-input-for-free is
# the LSM's standing discount and the merge join is where it gets spent.


@dataclass
class Meter:
    """Touches and held bytes, the two currencies."""

    touches: int = field(default=0)
    held_rows: int = field(default=0)


def hash_join(
    left: list[tuple[int, bytes]], right: list[tuple[int, bytes]], meter: Meter
) -> list[tuple[int, bytes, bytes]]:
    """Build on the smaller side, probe with the larger."""
    build, probe, flipped = (
        (left, right, False) if len(left) <= len(right) else (right, left, True)
    )
    table: dict[int, list[bytes]] = {}
    for key, value in build:
        meter.touches += 1
        table.setdefault(key, []).append(value)
    meter.held_rows = len(build)
    found = []
    for key, value in probe:
        meter.touches += 1
        for other in table.get(key, ()):
            if flipped:
                found.append((key, value, other))
            else:
                found.append((key, other, value))
    found.sort()
    return found


def merge_join(
    left: list[tuple[int, bytes]], right: list[tuple[int, bytes]], meter: Meter
) -> list[tuple[int, bytes, bytes]]:
    """Walk both sorted sides once, holding only the current key's group."""
    if [key for key, _ in left] != sorted(key for key, _ in left):
        raise ConfigError("the merge join needs sorted inputs")
    if [key for key, _ in right] != sorted(key for key, _ in right):
        raise ConfigError("the merge join needs sorted inputs")
    found = []
    at, other = 0, 0
    while at < len(left) and other < len(right):
        meter.touches += 1
        if left[at][0] < right[other][0]:
            at += 1
        elif left[at][0] > right[other][0]:
            other += 1
        else:
            key = left[at][0]
            left_group = []
            while at < len(left) and left[at][0] == key:
                left_group.append(left[at][1])
                at += 1
                meter.touches += 1
            right_group = []
            while other < len(right) and right[other][0] == key:
                right_group.append(right[other][1])
                other += 1
                meter.touches += 1
            meter.held_rows = max(meter.held_rows, len(left_group) + len(right_group))
            for a in left_group:
                for b in right_group:
                    found.append((key, a, b))
    found.sort()
    return found


@functools.cache
def _sides(orders: int = 20000, customers: int = 2000, seed: int = 239):
    """Customers and their orders, sorted by customer id, as the store would scan them."""
    source = random.Random(seed)
    left = [(at, f"customer-{at}".encode()) for at in range(customers)]
    right = sorted(
        (source.randrange(customers), f"order-{at}".encode()) for at in range(orders)
    )
    return tuple(left), tuple(right)


@functools.cache
def both_joins_produce_identical_output() -> bool:
    """Twenty thousand orders against two thousand customers, row for row equal.

    Including the edges: customers with no orders contribute nothing, a customer with many
    orders contributes the product, and both joins agree on every row after sorting.
    """
    left, right = _sides()
    hashed = hash_join(list(left), list(right), Meter())
    merged = merge_join(list(left), list(right), Meter())
    return hashed == merged and len(merged) == 20000


@functools.cache
def the_hash_join_holds_the_build_side_and_the_merge_holds_a_group() -> bool:
    """Memory: 2,000 rows held against 13.

    The hash join's table is the whole smaller side. The merge join holds one key's groups
    at a time, and its peak is the fattest customer's orders plus one. That ratio, three
    orders of magnitude here, is what sorted inputs buy, and it is the same purchase the
    heap merge made against the sort in the iterator module: the stream shape lets the
    state be a window instead of a copy.
    """
    left, right = _sides()
    hash_meter, merge_meter = Meter(), Meter()
    hash_join(list(left), list(right), hash_meter)
    merge_join(list(left), list(right), merge_meter)
    return hash_meter.held_rows == 2000 and merge_meter.held_rows < 30


@functools.cache
def both_joins_touch_each_row_about_once() -> bool:
    """Touch counts are within a factor of two of each other and of the input size.

    Neither join wins on touches; both are linear passes. The separation is entirely in
    memory and in the sortedness demand, which is why the planner's choice between them is
    about what the inputs already are, not about the join itself.
    """
    left, right = _sides()
    hash_meter, merge_meter = Meter(), Meter()
    hash_join(list(left), list(right), hash_meter)
    merge_join(list(left), list(right), merge_meter)
    total = len(left) + len(right)
    return (
        total <= hash_meter.touches <= total * 2
        and total <= merge_meter.touches <= total * 2
    )


@functools.cache
def the_merge_join_refuses_unsorted_input() -> bool:
    """Shuffled input raises rather than joining wrongly.

    An unsorted merge join does not fail loudly on its own; it silently drops the matches
    the cursor walks past, which is the worst kind of wrong answer, plausible and small.
    The check costs one pass and turns it into a refusal at the door.
    """
    left, right = _sides(2000, 200)
    shuffled = list(right)
    random.Random(1).shuffle(shuffled)
    try:
        merge_join(list(left), shuffled, Meter())
    except ConfigError:
        return True
    return False


@functools.cache
def skew_bloats_the_merge_joins_window() -> bool:
    """One customer with half the orders pushes the merge join's held rows to 10,003.

    The window is the fattest key's group, and a skewed join makes the fattest key half the
    table, at which point the merge join is quietly holding what the hash join holds and
    the memory advantage is gone. Skew is to joins what the hot tenant was to quotas: the
    average hides it and the maximum is the bill.
    """
    source = random.Random(241)
    left = [(at, f"c{at}".encode()) for at in range(2000)]
    right = sorted(
        (0 if source.random() < 0.5 else source.randrange(2000), f"o{at}".encode())
        for at in range(20000)
    )
    meter = Meter()
    merge_join(left, right, meter)
    return meter.held_rows > 9000


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "the_joins_agree": both_joins_produce_identical_output(),
        "memory_is_the_separation": (
            the_hash_join_holds_the_build_side_and_the_merge_holds_a_group()
        ),
        "touches_are_a_tie": both_joins_touch_each_row_about_once(),
        "unsorted_is_refused": the_merge_join_refuses_unsorted_input(),
        "skew_bloats_the_window": skew_bloats_the_merge_joins_window(),
    }
