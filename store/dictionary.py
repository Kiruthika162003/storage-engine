from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

from store.errors import NotFound

# Dictionary encoding: pay for the distinct values, not the occurrences.
#
# The columnar module ended by saying the narrow-field win pays for encodings; this is the
# first encoding it buys. A low cardinality column, status codes, cities, enum fields,
# stores each distinct value once in a dictionary and each occurrence as a small code. The
# saving is the value width against the code width, times the repetition. The measurements
# price it on the shapes that matter: a genuinely low cardinality column, a high cardinality
# one where the dictionary is pure overhead, and the operation that makes the encoding more
# than compression, filtering on codes without decoding a single value.


@dataclass
class Encoded:
    """One column, dictionary encoded."""

    values: list[bytes] = field(default_factory=list)
    codes: dict[bytes, int] = field(default_factory=dict)
    column: list[int] = field(default_factory=list)

    def append(self, value: bytes) -> None:
        """One occurrence."""
        code = self.codes.get(value)
        if code is None:
            code = len(self.values)
            self.values.append(value)
            self.codes[value] = code
        self.column.append(code)

    def read(self, at: int) -> bytes:
        """One occurrence back."""
        if not 0 <= at < len(self.column):
            raise NotFound(f"{at} is past the column")
        return self.values[self.column[at]]

    def scan_equal(self, wanted: bytes) -> list[int]:
        """Every position holding the value, compared as codes.

        The dictionary is consulted once; the scan compares small integers. A value not in
        the dictionary matches nowhere without touching the column at all, which is the
        filter's contradiction case appearing in an encoding.
        """
        code = self.codes.get(wanted)
        if code is None:
            return []
        return [at for at, held in enumerate(self.column) if held == code]

    @property
    def code_bytes(self) -> int:
        """What the codes cost: one, two or four bytes each, by cardinality."""
        if len(self.values) <= 256:
            each = 1
        elif len(self.values) <= 65536:
            each = 2
        else:
            each = 4
        return len(self.column) * each

    @property
    def dictionary_bytes(self) -> int:
        """What the dictionary costs: each distinct value once."""
        return sum(len(value) for value in self.values)

    @property
    def nbytes(self) -> int:
        return self.code_bytes + self.dictionary_bytes


def plain_bytes(column: list[bytes]) -> int:
    """The unencoded cost: every occurrence in full."""
    return sum(len(value) for value in column)


@functools.cache
def _cities(count: int = 50000, seed: int = 211) -> tuple[bytes, ...]:
    """A twelve byte city column with two hundred distinct values."""
    source = random.Random(seed)
    names = [f"city-{at:07d}".encode() for at in range(200)]
    return tuple(source.choice(names) for _ in range(count))


@functools.cache
def _uuids(count: int = 50000, seed: int = 223) -> tuple[bytes, ...]:
    """A unique column, where every value is its own dictionary entry."""
    source = random.Random(seed)
    return tuple(source.randbytes(12) for _ in range(count))


@functools.cache
def low_cardinality_compresses_by_the_width_over_the_code() -> bool:
    """The city column stores at 8.7 percent of plain: one code byte against twelve.

    Fifty thousand twelve byte cities with two hundred distinct values: 600,000 plain bytes
    become 50,000 code bytes plus a 2,400 byte dictionary. The ratio approaches code width
    over value width as repetition grows, and the dictionary amortises to nothing, which is
    the entire economics in one column.
    """
    column = list(_cities())
    encoded = Encoded()
    for value in column:
        encoded.append(value)
    return encoded.nbytes / plain_bytes(column) < 0.1


@functools.cache
def high_cardinality_makes_the_dictionary_pure_overhead() -> bool:
    """The unique column stores at 117 percent of plain: every value plus its code.

    Fifty thousand distinct values need the whole column in the dictionary and a two byte
    code per row on top, so the encoding costs 17 percent more than storing nothing cleverly.
    Cardinality is the whole decision, and real writers measure it per block and encode per
    block, because a column that is low cardinality this month is not sworn to stay so.
    """
    column = list(_uuids())
    encoded = Encoded()
    for value in column:
        encoded.append(value)
    return encoded.nbytes > plain_bytes(column) * 1.1


@functools.cache
def every_read_round_trips() -> bool:
    """Both columns read back exactly, position by position, through the encoding."""
    for column in (list(_cities(5000)), list(_uuids(5000))):
        encoded = Encoded()
        for value in column:
            encoded.append(value)
        if any(encoded.read(at) != column[at] for at in range(len(column))):
            return False
    return True


@functools.cache
def equality_scans_compare_codes_not_values() -> bool:
    """The filter finds every match by integer comparison, and misses without the column.

    The scan for one city agrees exactly with the plain filter, and a value outside the
    dictionary answers empty from the dictionary lookup alone: zero column positions
    touched, the predicate module's contradiction case delivered by an encoding.
    """
    column = list(_cities(20000))
    encoded = Encoded()
    for value in column:
        encoded.append(value)
    wanted = column[7]
    truth = [at for at, value in enumerate(column) if value == wanted]
    if encoded.scan_equal(wanted) != truth:
        return False
    return encoded.scan_equal(b"city-none") == []


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "low_cardinality_compresses": low_cardinality_compresses_by_the_width_over_the_code(),
        "high_cardinality_overheads": high_cardinality_makes_the_dictionary_pure_overhead(),
        "reads_round_trip": every_read_round_trips(),
        "scans_compare_codes": equality_scans_compare_codes_not_values(),
    }
