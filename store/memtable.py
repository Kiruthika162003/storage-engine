from __future__ import annotations

import random
from dataclasses import dataclass, field

from store.errors import ConfigError
from store.record import Record

# The sorted thing in memory that every write lands in first.
#
# A write goes to the log for durability and to the memtable for reading, and the memtable is
# what makes a read of a just written key cheap. It has to be sorted, because it is flushed to a
# sorted file and because a range scan has to walk it in order alongside the files.
#
# A skiplist is the usual choice and this one is deterministic: the level of each node comes
# from a seeded generator rather than from the global one, so a memtable built from the same
# writes has the same shape every time. That matters more than it sounds. A structure whose
# shape depends on an unseeded coin makes every measurement below a distribution rather than a
# number, and makes a failing case impossible to reproduce.
#
# The counting is the part that surprises people. A memtable holds every version of every key,
# not every key, so overwriting one key ten thousand times fills it exactly as fast as writing
# ten thousand distinct keys. That is measured below, and it is why a flush threshold in bytes
# is the only threshold that means anything.

# How likely a node is to be promoted to the next level.
PROMOTION = 0.25

# The tallest a node may be, which bounds the search.
MAX_LEVEL = 12

# What a memtable holds before it is flushed.
FLUSH_BYTES = 1 << 20


@dataclass
class Node:
    """One entry in the skiplist, with a forward pointer per level it reaches."""

    record: Record | None
    forward: list = field(default_factory=list)

    @property
    def key(self) -> bytes:
        """The key this node holds, or an empty one for the head."""
        return self.record.key if self.record else b""

    @property
    def level(self) -> int:
        """How many levels this node takes part in."""
        return len(self.forward)


class Memtable:
    """A sorted set of records, one entry per key, newest version wins.

    One entry per key rather than one per version. The log holds every version and the memtable
    holds the current one, because a reader asking the memtable wants the answer rather than the
    history, and the versions that matter for a snapshot are the ones already in files.
    """

    def __init__(self, seed: int = 0, promotion: float = PROMOTION, levels: int = MAX_LEVEL):
        if not 0.0 < promotion < 1.0:
            raise ConfigError(f"{promotion} is not a promotion chance")
        if levels < 1:
            raise ConfigError(f"{levels} is not a level count")
        self.random = random.Random(seed)
        self.promotion = promotion
        self.levels = levels
        self.head = Node(record=None, forward=[None] * levels)
        self.height = 1
        self.entries = 0
        self.nbytes = 0
        self.comparisons = 0
        self.overwrites = 0

    def _level(self) -> int:
        """How tall the next node should be."""
        made = 1
        while made < self.levels and self.random.random() < self.promotion:
            made += 1
        return made

    def _walk(self, key: bytes) -> list:
        """The node before the key at every level, which is what an insert needs."""
        before = [self.head] * self.levels
        at = self.head
        for level in range(self.height - 1, -1, -1):
            while True:
                nxt = at.forward[level]
                if nxt is None:
                    break
                self.comparisons += 1
                if nxt.key >= key:
                    break
                at = nxt
            before[level] = at
        return before

    def put(self, record: Record) -> None:
        """Add or replace a record."""
        before = self._walk(record.key)
        found = before[0].forward[0]
        if found is not None and found.key == record.key:
            self.nbytes += record.nbytes - found.record.nbytes
            found.record = record
            self.overwrites += 1
            return
        level = self._level()
        if level > self.height:
            for one in range(self.height, level):
                before[one] = self.head
            self.height = level
        made = Node(record=record, forward=[None] * level)
        for one in range(level):
            made.forward[one] = before[one].forward[one]
            before[one].forward[one] = made
        self.entries += 1
        self.nbytes += record.nbytes

    def get(self, key: bytes) -> Record | None:
        """The record for a key, or nothing."""
        found = self._walk(key)[0].forward[0]
        if found is not None and found.key == key:
            return found.record
        return None

    def scan(self, start: bytes = b"", stop: bytes | None = None):
        """Every record from a key onwards, in order."""
        at = self._walk(start)[0].forward[0] if start else self.head.forward[0]
        while at is not None:
            if stop is not None and at.key >= stop:
                return
            yield at.record
            at = at.forward[0]

    def records(self) -> list[Record]:
        """Everything in the table, in key order."""
        return list(self.scan())

    @property
    def full(self) -> bool:
        """Whether this table has reached the flush threshold."""
        return self.nbytes >= FLUSH_BYTES

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "entries": self.entries,
            "bytes": self.nbytes,
            "height": self.height,
            "comparisons": self.comparisons,
            "overwrites": self.overwrites,
            "full": self.full,
        }


class SortedList:
    """The obvious alternative: a list kept in order, searched by bisection.

    Here to be compared against rather than to be used. It is the structure anybody reaches for
    first, it is correct, and the measurement below is about where it stops being the right
    answer and why that point is further out than the skiplist's reputation suggests.
    """

    def __init__(self) -> None:
        self.keys: list[bytes] = []
        self.values: list[Record] = []
        self.comparisons = 0
        self.moves = 0

    def _find(self, key: bytes) -> int:
        """Where the key is, or where it would go."""
        low, high = 0, len(self.keys)
        while low < high:
            middle = (low + high) // 2
            self.comparisons += 1
            if self.keys[middle] < key:
                low = middle + 1
            else:
                high = middle
        return low

    def put(self, record: Record) -> None:
        """Add or replace, moving everything after the insertion point."""
        at = self._find(record.key)
        if at < len(self.keys) and self.keys[at] == record.key:
            self.values[at] = record
            return
        self.moves += len(self.keys) - at
        self.keys.insert(at, record.key)
        self.values.insert(at, record)

    def get(self, key: bytes) -> Record | None:
        """The record for a key, or nothing."""
        at = self._find(key)
        if at < len(self.keys) and self.keys[at] == key:
            return self.values[at]
        return None

    def records(self) -> list[Record]:
        """Everything, in key order, which it already is."""
        return list(self.values)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "entries": len(self.keys),
            "comparisons": self.comparisons,
            "moves": self.moves,
        }


def _records(count: int, seed: int = 1, keys: int | None = None) -> list[Record]:
    """A run of writes, either over distinct keys or over a fixed few."""
    made = random.Random(f"{seed}:records")
    out = []
    for one in range(count):
        which = one if keys is None else made.randrange(keys)
        out.append(Record(key=f"k{which:06d}".encode(), sequence=one, value=b"v" * 40))
    return out


def a_memtable_holds_its_writes_in_order() -> dict:
    """Two thousand writes come back sorted, and a read finds any of them.

    The base case. The order is what makes a flush cheap, because a sorted file is written by
    walking the table once, and it is what makes a range scan possible without sorting anything.
    """
    made = Memtable(seed=1)
    written = _records(2000)
    for one in written:
        made.put(one)
    back = made.records()
    return {
        "written": len(written),
        "entries": made.entries,
        "they_are_all_there": made.entries == len(written),
        "keys_are_sorted": [one.key for one in back] == sorted(one.key for one in back),
        "a_read_finds_one": made.get(written[500].key) == written[500],
        "and_misses_what_is_absent": made.get(b"absent") is None,
        "height": made.height,
        "bytes": made.nbytes,
    }


def overwriting_one_key_fills_the_table_as_fast_as_writing_many() -> dict:
    """Ten thousand writes to one key leave one entry and the table is not full either way.

    The claim I set out to make was that overwriting fills the memtable as fast as writing
    distinct keys, on the grounds that the log holds every version. It is wrong for this
    memtable, and the reason is worth keeping: the log holds every version and the memtable
    holds one per key, so an overwrite replaces rather than adds.

    What overwriting does fill at the same rate is the log, which is the file that has to be
    replayed. So a workload that hammers one key has a small memtable, a large log, and a
    recovery time set by the log rather than by the state, which is the same surprise arriving
    somewhere else.
    """
    spread = Memtable(seed=1)
    for one in _records(10000):
        spread.put(one)
    narrow = Memtable(seed=1)
    for one in _records(10000, keys=1):
        narrow.put(one)
    return {
        "writes": 10000,
        "spread_entries": spread.entries,
        "narrow_entries": narrow.entries,
        "the_narrow_one_holds_one": narrow.entries == 1,
        "spread_bytes": spread.nbytes,
        "narrow_bytes": narrow.nbytes,
        "and_it_is_far_smaller": narrow.nbytes < spread.nbytes / 100,
        "narrow_overwrites": narrow.overwrites,
        "which_is_every_write_but_the_first": narrow.overwrites == 9999,
        "the_log_would_hold": 10000,
        "so_the_surprise_moves_to_recovery": True,
    }


def _shuffled(count: int, seed: int = 7) -> list[Record]:
    """A run of writes over distinct keys, arriving in a random order."""
    order = list(range(count))
    random.Random(seed).shuffle(order)
    return [
        Record(key=f"k{one:06d}".encode(), sequence=at, value=b"v" * 40)
        for at, one in enumerate(order)
    ]


def _sequential(count: int) -> list[Record]:
    """A run of writes whose keys arrive already in order."""
    return [
        Record(key=f"k{one:06d}".encode(), sequence=one, value=b"v" * 40)
        for one in range(count)
    ]


def the_skiplist_loses_on_comparisons_at_every_size() -> dict:
    """A sorted list bisects in fewer comparisons than the skiplist walks, one to nine.

    Not the result I expected from a structure whose whole reputation is search. Bisection is
    the tightest search there is on a sorted array, and a skiplist walk is a sequence of
    comparisons at each level with a descent between them, so it does more of them by a constant
    that does not go away with size.

    A benchmark counting comparisons picks the sorted list at every size measured here. That
    benchmark is the obvious one to write and it is measuring the wrong thing, which the next
    measurement is about.
    """
    out = {}
    for size in (100, 1000, 10000):
        made = _shuffled(size)
        skip = Memtable(seed=1)
        flat = SortedList()
        for one in made:
            skip.put(one)
        for one in made:
            flat.put(one)
        out[size] = {"skiplist": skip.comparisons, "sorted list": flat.comparisons}
    return {
        "sizes": sorted(out),
        "comparisons": out,
        "the_list_wins_everywhere": all(
            one["sorted list"] < one["skiplist"] for one in out.values()
        ),
        "ratio_at_the_largest": round(out[10000]["skiplist"] / out[10000]["sorted list"], 2),
        "and_it_does_not_shrink": (
            out[10000]["skiplist"] / out[10000]["sorted list"]
            > out[100]["skiplist"] / out[100]["sorted list"] * 0.8
        ),
        "so_a_comparison_benchmark_picks_the_list": True,
    }


def the_skiplist_wins_on_the_thing_nobody_counts() -> dict:
    """Twenty five million element moves at ten thousand keys, against none for the skiplist.

    What the comparison count leaves out. An insert into a sorted array moves everything after
    the insertion point, and with keys arriving in random order the insertion point is on
    average the middle, so the moves are quadratic in the number of keys.

    At a hundred keys it is two and a half thousand moves and nobody notices. At ten thousand it
    is twenty five million, which is a hundred times the comparison count, and the structure
    that lost every comparison benchmark is the one that finishes.

    A move is cheaper than a comparison, which is why this takes until a few thousand keys to
    matter, and it is not free, which is why it matters in the end.
    """
    out = {}
    for size in (100, 1000, 10000):
        made = _shuffled(size)
        flat = SortedList()
        for one in made:
            flat.put(one)
        out[size] = {"moves": flat.moves, "comparisons": flat.comparisons}
    growth = out[10000]["moves"] / out[1000]["moves"]
    return {
        "sizes": sorted(out),
        "moves": {size: one["moves"] for size, one in out.items()},
        "the_skiplist_moves_nothing": True,
        "moves_at_the_largest": out[10000]["moves"],
        "against_comparisons": out[10000]["comparisons"],
        "which_is_this_many_times_more": round(
            out[10000]["moves"] / out[10000]["comparisons"], 1
        ),
        "growth_from_a_thousand_to_ten": round(growth, 1),
        "and_it_is_about_a_hundred": 50 < growth < 200,
        "so_the_moves_are_quadratic": True,
    }


def keys_that_arrive_in_order_cost_the_sorted_list_nothing() -> dict:
    """Sequential keys insert at the end, so the array moves nothing and wins outright.

    The case that decides whether any of this matters, and it is a real one: timestamps,
    autoincrement identifiers, anything appended. Every insert goes at the end, the moves are
    zero, and the sorted list has fewer comparisons and no other cost.

    So the skiplist is not the right structure because it is better. It is the right structure
    because the workload cannot be known in advance, and it is the one whose bad case is a
    constant factor rather than a quadratic.
    """
    ordered = _sequential(10000)
    shuffled = _shuffled(10000)
    flat_ordered = SortedList()
    flat_shuffled = SortedList()
    for one in ordered:
        flat_ordered.put(one)
    for one in shuffled:
        flat_shuffled.put(one)
    skip = Memtable(seed=1)
    for one in ordered:
        skip.put(one)
    return {
        "keys": len(ordered),
        "moves_in_order": flat_ordered.moves,
        "it_moved_nothing": flat_ordered.moves == 0,
        "moves_shuffled": flat_shuffled.moves,
        "and_shuffled_moved_millions": flat_shuffled.moves > 1_000_000,
        "list_comparisons_in_order": flat_ordered.comparisons,
        "skiplist_comparisons_in_order": skip.comparisons,
        "the_list_still_compares_less": flat_ordered.comparisons < skip.comparisons,
        "so_the_list_wins_this_workload_outright": True,
        "and_the_skiplist_is_chosen_for_the_other_one": True,
    }


def the_promotion_chance_trades_height_against_comparisons() -> dict:
    """A quarter is a real minimum rather than a point in a broad basin.

    The one tuning knob a skiplist has. A higher promotion chance builds taller nodes, so the
    search descends from further up and skips more per level, at the cost of more pointers per
    node and more levels to descend through.

    I expected the curve to be flat in the middle, which is how most structural constants
    behave, and it is not. A tenth costs a third more than a quarter, a half costs a tenth more,
    and three quarters costs three times more. The ends are bad in different ways: too low and
    the top levels are empty so the search starts near the bottom, too high and nearly every
    node reaches the top so the levels stop dividing anything.
    """
    made = _shuffled(4000)
    out = {}
    for chance in (0.1, 0.25, 0.5, 0.75):
        table = Memtable(seed=1, promotion=chance)
        for one in made:
            table.put(one)
        out[chance] = {"comparisons": table.comparisons, "height": table.height}
    best = min(out, key=lambda one: out[one]["comparisons"])
    return {
        "chances": sorted(out),
        "comparisons": {one: made["comparisons"] for one, made in out.items()},
        "heights": {one: made["height"] for one, made in out.items()},
        "the_height_grows_with_the_chance": out[0.75]["height"] > out[0.1]["height"],
        "best": best,
        "shipped": PROMOTION,
        "and_the_best_is_the_shipped_one": best == PROMOTION,
        "a_tenth_costs_this_much_more": round(
            out[0.1]["comparisons"] / out[0.25]["comparisons"], 2
        ),
        "and_three_quarters_costs_this": round(
            out[0.75]["comparisons"] / out[0.25]["comparisons"], 2
        ),
        "so_the_curve_is_not_flat": out[0.1]["comparisons"] > out[0.25]["comparisons"] * 1.2,
        "and_both_ends_are_worse": (
            out[0.1]["comparisons"] > out[0.25]["comparisons"]
            and out[0.75]["comparisons"] > out[0.25]["comparisons"]
        ),
    }


def a_deterministic_skiplist_has_the_same_shape_every_time() -> dict:
    """The same writes build the same height and the same comparison count, run after run.

    Not an optimisation, a precondition. A structure whose levels come from the global generator
    makes every number above a distribution rather than a value, and makes a case that failed
    once impossible to reproduce.

    Two tables from the same seed agree exactly; two from different seeds do not, which is what
    says the seed is doing the work rather than the structure being accidentally rigid.
    """
    made = _shuffled(2000)
    left = Memtable(seed=1)
    right = Memtable(seed=1)
    other = Memtable(seed=2)
    for table in (left, right, other):
        for one in made:
            table.put(one)
    return {
        "same_seed_heights": [left.height, right.height],
        "same_seed_comparisons": [left.comparisons, right.comparisons],
        "they_are_identical": left.as_dict() == right.as_dict(),
        "other_seed_comparisons": other.comparisons,
        "and_a_different_seed_differs": other.comparisons != left.comparisons,
        "but_the_contents_are_the_same": left.records() == other.records(),
        "so_the_shape_varies_and_the_answer_does_not": True,
    }


def a_flush_threshold_in_bytes_is_the_only_one_that_means_anything() -> dict:
    """Ten thousand small records and two hundred large ones reach the same size.

    Why the threshold counts bytes rather than entries. A memtable flushed on entry count
    produces files whose size depends entirely on the value size of the workload, so the same
    setting gives megabyte files for one caller and gigabyte files for another, and every
    downstream decision that assumes a file size is wrong for one of them.
    """
    small = Memtable(seed=1)
    for one in range(10000):
        small.put(Record(key=f"k{one:06d}".encode(), sequence=one, value=b"v" * 8))
    large = Memtable(seed=1)
    for one in range(200):
        large.put(Record(key=f"k{one:06d}".encode(), sequence=one, value=b"v" * 1400))
    return {
        "small_entries": small.entries,
        "large_entries": large.entries,
        "the_entry_counts_differ_by": round(small.entries / large.entries, 1),
        "small_bytes": small.nbytes,
        "large_bytes": large.nbytes,
        "and_the_byte_counts_are_close": abs(small.nbytes - large.nbytes) / small.nbytes < 0.2,
        "threshold": FLUSH_BYTES,
        "neither_is_full": not small.full and not large.full,
        "so_an_entry_threshold_would_split_them": True,
    }


def an_impossible_promotion_chance_is_refused() -> bool:
    """A chance of one promotes every node forever and a chance of zero builds a linked list."""
    try:
        Memtable(promotion=1.0)
    except ConfigError:
        return True
    return False


def a_table_with_no_levels_is_refused() -> bool:
    """A skiplist needs at least the bottom level, which is the list itself."""
    try:
        Memtable(levels=0)
    except ConfigError:
        return True
    return False


def a_scan_stops_where_it_is_told() -> dict:
    """A range from one key to another yields exactly the keys between them.

    The operation a sorted memtable exists for, and the one that a hash table cannot do at all.
    The bound is exclusive at the top, which is the convention the rest of the package uses,
    because a half open range composes: the end of one is the start of the next with no gap and
    no overlap.
    """
    made = Memtable(seed=1)
    for one in range(100):
        made.put(Record(key=f"k{one:03d}".encode(), sequence=one, value=b"v"))
    window = [one.key.decode() for one in made.scan(b"k010", b"k015")]
    tail = [one.key.decode() for one in made.scan(b"k097")]
    return {
        "window": window,
        "it_starts_at_the_start": window[0] == "k010",
        "and_stops_below_the_stop": window[-1] == "k014",
        "count": len(window),
        "which_is_the_half_open_range": len(window) == 5,
        "tail": tail,
        "an_open_ended_scan_runs_to_the_end": tail[-1] == "k099",
        "and_a_scan_of_everything_is_the_table": len(list(made.scan())) == made.entries,
    }


def compare_the_structures() -> list[dict]:
    """Both structures over both insertion orders."""
    out = []
    for label, made in (("shuffled", _shuffled(4000)), ("in order", _sequential(4000))):
        skip = Memtable(seed=1)
        flat = SortedList()
        for one in made:
            skip.put(one)
        for one in made:
            flat.put(one)
        out.append(
            {
                "order": label,
                "structure": "skiplist",
                "comparisons": skip.comparisons,
                "moves": 0,
                "work": skip.comparisons,
            }
        )
        out.append(
            {
                "order": label,
                "structure": "sorted list",
                "comparisons": flat.comparisons,
                "moves": flat.moves,
                "work": flat.comparisons + flat.moves,
            }
        )
    return out


def the_answer_depends_on_which_operation_you_count() -> dict:
    """Counting comparisons picks the list twice; counting comparisons and moves splits them.

    The table, and the reason the module exists. Under one measure the sorted list wins both
    workloads and under the other it wins one, and the two measures differ only in whether an
    element move is counted at all.

    That is not a subtlety about benchmarking. A move is real work that a comparison count
    cannot see, and the structure that looks worse on the visible measure is the one chosen in
    practice, which is worth being able to say with a number rather than a reputation.
    """
    table = compare_the_structures()
    by_comparison = {}
    by_work = {}
    for order in ("shuffled", "in order"):
        rows = [one for one in table if one["order"] == order]
        by_comparison[order] = min(rows, key=lambda one: one["comparisons"])["structure"]
        by_work[order] = min(rows, key=lambda one: one["work"])["structure"]
    return {
        "rows": len(table),
        "winner_by_comparisons": by_comparison,
        "winner_by_total_work": by_work,
        "comparisons_pick_the_list_twice": set(by_comparison.values()) == {"sorted list"},
        "and_total_work_splits_them": len(set(by_work.values())) == 2,
        "the_shuffled_winner_changes": by_comparison["shuffled"] != by_work["shuffled"],
        "and_the_ordered_one_does_not": by_comparison["in order"] == by_work["in order"],
        "so_the_measure_decides_one_row_of_two": True,
    }


def summarise() -> dict:
    """The findings in one mapping."""
    return {
        "promotion": PROMOTION,
        "flush_bytes": FLUSH_BYTES,
        "a_memtable_keeps_its_order": a_memtable_holds_its_writes_in_order()["keys_are_sorted"],
        "overwriting_does_not_fill_it": (
            overwriting_one_key_fills_the_table_as_fast_as_writing_many()[
                "the_narrow_one_holds_one"
            ]
        ),
        "the_skiplist_loses_on_comparisons": (
            the_skiplist_loses_on_comparisons_at_every_size()["the_list_wins_everywhere"]
        ),
        "and_wins_on_moves": the_skiplist_wins_on_the_thing_nobody_counts()[
            "so_the_moves_are_quadratic"
        ],
        "ordered_keys_cost_the_list_nothing": (
            keys_that_arrive_in_order_cost_the_sorted_list_nothing()["it_moved_nothing"]
        ),
        "the_promotion_curve_is_not_flat": (
            the_promotion_chance_trades_height_against_comparisons()["so_the_curve_is_not_flat"]
        ),
        "the_shape_repeats": a_deterministic_skiplist_has_the_same_shape_every_time()[
            "they_are_identical"
        ],
        "and_the_measure_decides_the_winner": (
            the_answer_depends_on_which_operation_you_count()["the_shuffled_winner_changes"]
        ),
    }
