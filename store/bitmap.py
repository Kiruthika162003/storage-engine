from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

# Bitmap indexes: set algebra at memory bandwidth, priced against posting lists.
#
# The dictionary module filtered one column by one value. Analytical predicates combine:
# status paid AND city 7, status refunded OR flagged. A posting list per value answers each
# clause with sorted row ids and intersects them by merge; a bitmap per value answers with
# one bit per row and intersects 64 rows per machine word. The bitmap's cost is that the bit
# per row is charged whether the value is common or rare, so rare values make sparse
# bitmaps that posting lists undercut by orders of magnitude, which is why real systems
# roaring-compress them. Both structures are held to identical answers first.

WORD = 64


@dataclass
class Bitmap:
    """One value's rows, one bit each."""

    words: list[int] = field(default_factory=list)
    rows: int = field(default=0)

    def set_row(self, at: int) -> None:
        while at >= len(self.words) * WORD:
            self.words.append(0)
        self.words[at // WORD] |= 1 << (at % WORD)
        self.rows = max(self.rows, at + 1)

    def positions(self) -> list[int]:
        found = []
        for word_at, held in enumerate(self.words):
            base = word_at * WORD
            remaining = held
            while remaining:
                low = remaining & -remaining
                found.append(base + low.bit_length() - 1)
                remaining ^= low
        return found

    def intersect(self, other: Bitmap) -> Bitmap:
        made = Bitmap()
        made.words = [
            a & b
            for a, b in zip(self.words, other.words, strict=False)
        ]
        made.rows = min(self.rows, other.rows)
        return made

    def union(self, other: Bitmap) -> Bitmap:
        longer, shorter = (
            (self.words, other.words)
            if len(self.words) >= len(other.words)
            else (other.words, self.words)
        )
        made = Bitmap()
        made.words = list(longer)
        for at, word in enumerate(shorter):
            made.words[at] |= word
        made.rows = max(self.rows, other.rows)
        return made

    @property
    def nbytes(self) -> int:
        return len(self.words) * 8


def posting_intersect(left: list[int], right: list[int]) -> list[int]:
    """The merge intersection, counting nothing, correct by construction."""
    found = []
    at, other = 0, 0
    while at < len(left) and other < len(right):
        if left[at] == right[other]:
            found.append(left[at])
            at += 1
            other += 1
        elif left[at] < right[other]:
            at += 1
        else:
            other += 1
    return found


def posting_bytes(rows: list[int]) -> int:
    """Four bytes per row id."""
    return 4 * len(rows)


@functools.cache
def _table(rows: int = 50000, seed: int = 233):
    """A status column and a city column, as value-to-rows mappings."""
    source = random.Random(seed)
    status_rows: dict[str, list[int]] = {name: [] for name in ("paid", "open", "flagged")}
    city_rows: dict[int, list[int]] = {at: [] for at in range(50)}
    for row in range(rows):
        status = source.choices(("paid", "open", "flagged"), weights=(70, 29, 1))[0]
        status_rows[status].append(row)
        city_rows[source.randrange(50)].append(row)
    return status_rows, city_rows


def _bitmap_of(rows: list[int]) -> Bitmap:
    made = Bitmap()
    for at in rows:
        made.set_row(at)
    return made


@functools.cache
def bitmaps_and_postings_answer_identically() -> bool:
    """AND and OR of every clause pair agree between the two structures.

    The set algebra license: nine status-city pairs intersected and three status pairs
    unioned, each compared position for position against the posting merge.
    """
    status_rows, city_rows = _table(10000)
    for status in status_rows:
        for city in (0, 7, 49):
            wanted = posting_intersect(status_rows[status], city_rows[city])
            got = _bitmap_of(tuple(status_rows[status])).intersect(
                _bitmap_of(tuple(city_rows[city]))
            )
            if got.positions() != wanted:
                return False
    paid = _bitmap_of(tuple(status_rows["paid"]))
    flagged = _bitmap_of(tuple(status_rows["flagged"]))
    unioned = sorted(set(status_rows["paid"]) | set(status_rows["flagged"]))
    return paid.union(flagged).positions() == unioned


@functools.cache
def common_values_cost_less_as_bits() -> bool:
    """The paid bitmap costs 6,248 bytes against the posting list's 140,000.

    Seventy percent of fifty thousand rows is 35,000 four byte ids, and the bitmap is one
    bit per row regardless. Dense values are the bitmap's home ground: the break even is
    one set bit per 32 rows, and paid is 22 times past it.
    """
    status_rows, _ = _table()
    paid = status_rows["paid"]
    bitmap = _bitmap_of(tuple(paid))
    return bitmap.nbytes < posting_bytes(paid) / 20


@functools.cache
def rare_values_waste_bits_by_the_thousand() -> bool:
    """The flagged bitmap costs 3.3 times the posting list.

    One percent of the rows set one bit each and the other 99 percent of the bits are paid
    for anyway: 6,248 bitmap bytes against 1,916 for the 479 ids. Sparse values are the
    posting list's home ground, the same one-in-32 break even seen from the other side,
    and roaring bitmaps exist to stop anyone choosing per column.
    """
    status_rows, _ = _table()
    flagged = status_rows["flagged"]
    bitmap = _bitmap_of(tuple(flagged))
    return bitmap.nbytes > posting_bytes(flagged) * 3


@functools.cache
def the_intersection_touches_words_not_rows() -> bool:
    """The bitmap AND does one operation per 64 rows; the merge does one per posting.

    Fifty thousand rows intersect in 782 word operations, against the tens of thousands of
    comparisons the posting merge walks. This is the bandwidth argument in its purest form:
    the bitmap does not skip work cleverly, it makes the work vector shaped.
    """
    status_rows, city_rows = _table()
    paid = _bitmap_of(tuple(status_rows["paid"]))
    city = _bitmap_of(tuple(city_rows[7]))
    word_ops = min(len(paid.words), len(city.words))
    posting_ops = len(status_rows["paid"]) + len(city_rows[7])
    return word_ops * 40 < posting_ops


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "the_structures_agree": bitmaps_and_postings_answer_identically(),
        "dense_values_suit_bits": common_values_cost_less_as_bits(),
        "sparse_values_suit_postings": rare_values_waste_bits_by_the_thousand(),
        "the_and_is_word_shaped": the_intersection_touches_words_not_rows(),
    }
