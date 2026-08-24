from __future__ import annotations

import functools
import random

from store.errors import ConfigError

# Newest first: making descending time the cheap direction.
#
# A feed, a log viewer, an inbox: the query is the latest N, and a store that sorts ascending
# answers it by walking to the end, which for an iterator that only goes forward means reading
# everything. The fix is in the key, not the engine: encode the timestamp complemented, so the
# largest time makes the smallest key, and the latest N is a forward scan of N records from
# the front. The trick costs nothing and its cost is real anyway: the oldest-first query
# becomes the expensive one, because the encoding can only donate the cheap direction once.

WIDTH = 8
CEILING = (1 << 64) - 1


def ascending(moment: int) -> bytes:
    """Time as bytes that sort oldest first."""
    if not 0 <= moment <= CEILING:
        raise ConfigError(f"{moment} does not fit in 64 bits")
    return moment.to_bytes(WIDTH, "big")


def descending(moment: int) -> bytes:
    """Time as bytes that sort newest first: the complement."""
    if not 0 <= moment <= CEILING:
        raise ConfigError(f"{moment} does not fit in 64 bits")
    return (CEILING - moment).to_bytes(WIDTH, "big")


def read_ascending(raw: bytes) -> int:
    """The moment back from the ascending form."""
    if len(raw) != WIDTH:
        raise ConfigError(f"{len(raw)} bytes is not a time key")
    return int.from_bytes(raw, "big")


def read_descending(raw: bytes) -> int:
    """The moment back from the complemented form."""
    if len(raw) != WIDTH:
        raise ConfigError(f"{len(raw)} bytes is not a time key")
    return CEILING - int.from_bytes(raw, "big")


def latest(keys: list[bytes], count: int) -> list[bytes]:
    """The first N of a sorted key list, which is the whole query plan."""
    return sorted(keys)[:count]


@functools.cache
def _moments(count: int = 5000, seed: int = 107) -> tuple[int, ...]:
    """Event times, increasing with jitter."""
    source = random.Random(seed)
    made = []
    clock = 1_700_000_000_000
    for _ in range(count):
        clock += source.randrange(1, 2000)
        made.append(clock)
    return tuple(made)


@functools.cache
def the_complement_reverses_the_sort_exactly() -> bool:
    """Sorting complemented keys gives exactly the reverse of sorting plain ones.

    Not roughly, exactly: the two orders are element for element mirrors across five
    thousand jittered timestamps, because complementation is strictly monotone decreasing
    and byte comparison respects it at every position.
    """
    moments = list(_moments())
    plain = sorted(ascending(moment) for moment in moments)
    flipped = sorted(descending(moment) for moment in moments)
    forward = [read_ascending(key) for key in plain]
    backward = [read_descending(key) for key in flipped]
    return backward == list(reversed(forward))


@functools.cache
def the_latest_n_is_a_prefix_read() -> bool:
    """The ten newest events are the first ten complemented keys, no tail walk.

    Under the ascending encoding the same query is the last ten, which a forward iterator
    reaches by passing 4,990 records. The record counts are the argument: ten against five
    thousand for the identical answer.
    """
    moments = list(_moments())
    flipped = [descending(moment) for moment in moments]
    front = latest(flipped, 10)
    found = [read_descending(key) for key in front]
    return found == sorted(moments, reverse=True)[:10]


@functools.cache
def the_round_trip_is_exact_at_the_edges() -> bool:
    """Zero, the ceiling, and a spray of random moments all survive both encodings.

    The edges are where a complement encoding breaks: an off by one turns the ceiling into
    an overflow or zero into a negative, so both ends are pinned along with the middle.
    """
    source = random.Random(3)
    probes = [0, 1, CEILING - 1, CEILING] + [source.randrange(CEILING) for _ in range(500)]
    for moment in probes:
        if read_ascending(ascending(moment)) != moment:
            return False
        if read_descending(descending(moment)) != moment:
            return False
    return True


@functools.cache
def the_donated_direction_cannot_be_taken_back() -> bool:
    """Under the complement, oldest first becomes the full walk that newest first was.

    The encoding moves the cost, it does not remove it. The oldest ten under descending keys
    are the last ten, 4,990 records deep, exactly the position the newest ten occupied under
    ascending. A table that needs both directions cheap needs two key layouts, which is a
    secondary index by another name.
    """
    moments = list(_moments())
    flipped = sorted(descending(moment) for moment in moments)
    oldest_position = len(flipped) - 10
    oldest = [read_descending(key) for key in flipped[oldest_position:]]
    return oldest == sorted(moments)[:10][::-1]


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "the_complement_mirrors": the_complement_reverses_the_sort_exactly(),
        "latest_n_is_a_prefix": the_latest_n_is_a_prefix_read(),
        "edges_round_trip": the_round_trip_is_exact_at_the_edges(),
        "the_donation_is_final": the_donated_direction_cannot_be_taken_back(),
    }
