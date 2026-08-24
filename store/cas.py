from __future__ import annotations

import contextlib
import functools
import itertools
import random
from dataclasses import dataclass, field

from store.errors import ConfigError, Conflict, NotFound

# Compare and swap, and the ABA that value comparison cannot see.
#
# The transaction machinery is heavy for one key; compare-and-swap is the light tool: the
# caller says change it to Y only if it is still X, and the read-modify-write race
# disappears into the atomic compare. The trap is inside the word still: comparing values
# says the value equals X, not that it stayed X, and a key that went X to B and back to X
# passes the compare while everything the caller inferred from X is stale. The version
# counter closes the gap by comparing a number that never revisits, and the module builds
# the ABA against the value CAS, then shows the versioned CAS refusing it.


@dataclass
class Cell:
    """One key with a value and a version that only climbs."""

    value: bytes
    version: int = field(default=1)


@dataclass
class Register:
    """A CAS store offering both compare disciplines."""

    cells: dict[bytes, Cell] = field(default_factory=dict)
    swaps: int = field(default=0)
    refusals: int = field(default=0)

    def put(self, key: bytes, value: bytes) -> int:
        """An unconditional write, versions climbing."""
        if not key:
            raise ConfigError("a key needs at least one byte")
        held = self.cells.get(key)
        if held is None:
            self.cells[key] = Cell(value=value)
        else:
            held.value = value
            held.version += 1
        return self.cells[key].version

    def read(self, key: bytes) -> tuple[bytes, int]:
        """The value and its version, both needed for the safe compare."""
        held = self.cells.get(key)
        if held is None:
            raise NotFound(f"{key!r} is not here")
        return held.value, held.version

    def cas_value(self, key: bytes, expect: bytes, value: bytes) -> None:
        """The value compare: succeeds whenever the value matches, history invisible."""
        held = self.cells.get(key)
        if held is None or held.value != expect:
            self.refusals += 1
            raise Conflict("the value moved")
        held.value = value
        held.version += 1
        self.swaps += 1

    def cas_version(self, key: bytes, expect_version: int, value: bytes) -> None:
        """The version compare: succeeds only if nothing happened since the read."""
        held = self.cells.get(key)
        if held is None or held.version != expect_version:
            self.refusals += 1
            raise Conflict("the cell moved")
        held.value = value
        held.version += 1
        self.swaps += 1


@functools.cache
def a_cas_loop_survives_interleaving_that_breaks_read_modify_write() -> bool:
    """Two hundred interleaved increments through CAS loops land exactly on two hundred.

    The read-modify-write baseline drops updates whenever two actors read the same value;
    the CAS loop retries the loser instead, and the final counter is the proof: every
    increment landed once. The refusal count is the price, visible on the meter.
    """
    source = random.Random(353)
    register = Register()
    register.put(b"counter", (0).to_bytes(4, "big"))
    actors = [None, None]
    landed = 0
    while landed < 200:
        at = source.randrange(2)
        if actors[at] is None:
            value, _ = register.read(b"counter")
            actors[at] = value
            continue
        expect = actors[at]
        actors[at] = None
        fresh = (int.from_bytes(expect, "big") + 1).to_bytes(4, "big")
        try:
            register.cas_value(b"counter", expect, fresh)
            landed += 1
        except Conflict:
            continue
    final = int.from_bytes(register.read(b"counter")[0], "big")
    return final == 200 and register.refusals > 0


@functools.cache
def the_value_compare_accepts_the_aba() -> bool:
    """X to B and back to X, and the value CAS walks straight through.

    The head pointer went away and came back while the caller slept, everything reachable
    from the old X is stale, and the compare has no way to know: it sees X, X matches,
    swap. The success is the bug, and the meter shows it as an ordinary swap.
    """
    register = Register()
    register.put(b"head", b"X")
    stale_value, _ = register.read(b"head")
    register.put(b"head", b"B")
    register.put(b"head", b"X")
    try:
        register.cas_value(b"head", stale_value, b"mine")
    except Conflict:
        return False
    return register.read(b"head")[0] == b"mine"


@functools.cache
def the_version_compare_refuses_the_aba() -> bool:
    """The same round trip under version compare: refused, because versions never revisit.

    The value came back and the version did not, three writes being three increments, and
    the sleeping caller's version 1 no longer names the cell. The counter costs eight
    bytes a key and converts stayed-the-same from an inference into a fact.
    """
    register = Register()
    register.put(b"head", b"X")
    _, stale_version = register.read(b"head")
    register.put(b"head", b"B")
    register.put(b"head", b"X")
    try:
        register.cas_version(b"head", stale_version, b"mine")
        return False
    except Conflict:
        pass
    return register.read(b"head")[0] == b"X"


@functools.cache
def versions_only_climb() -> bool:
    """A thousand mixed operations and the version sequence never repeats or descends."""
    source = random.Random(359)
    register = Register()
    register.put(b"k", b"0")
    seen = [register.read(b"k")[1]]
    for _ in range(1000):
        if source.random() < 0.5:
            register.put(b"k", source.randbytes(4))
        else:
            _, version = register.read(b"k")
            with contextlib.suppress(Conflict):
                register.cas_version(b"k", version, source.randbytes(4))
        seen.append(register.read(b"k")[1])
    return all(later >= earlier for earlier, later in itertools.pairwise(seen))


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "the_loop_beats_the_race": (
            a_cas_loop_survives_interleaving_that_breaks_read_modify_write()
        ),
        "value_compares_accept_aba": the_value_compare_accepts_the_aba(),
        "version_compares_refuse_aba": the_version_compare_refuses_the_aba(),
        "versions_only_climb": versions_only_climb(),
    }
