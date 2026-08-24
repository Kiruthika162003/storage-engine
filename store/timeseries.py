from __future__ import annotations

import functools
import itertools
import random

from store.errors import BadFormat, ConfigError
from store.varint import decode as varint_decode
from store.varint import encode as varint_encode

# Delta of delta: the encoding that knows clocks tick evenly.
#
# The varint module compressed sequence numbers by storing gaps. Timestamps go one further:
# their gaps are themselves nearly constant, ten seconds between samples, every sample, so
# the gap of the gaps is nearly always zero, and an encoding that stores the second
# difference spends almost nothing on the common case. The zigzag mapping folds the signed
# second difference into the unsigned varint the package already has: zero stays zero,
# minus one becomes one, one becomes two, so small magnitudes of either sign stay one byte.
# The measurements price three streams: a metronome, a realistic jittery scrape, and the
# irregular events where the scheme falls back to plain deltas.


def zigzag(value: int) -> int:
    """A signed integer folded to unsigned, small magnitudes staying small."""
    return (value << 1) ^ (value >> 63) if value >= 0 else ((-value) << 1) - 1


def unzigzag(value: int) -> int:
    """The fold undone."""
    return (value >> 1) if value % 2 == 0 else -((value + 1) >> 1)


def encode(moments: list[int]) -> bytes:
    """First value, first delta, then delta of deltas, all varints, signed ones zigzagged."""
    if not moments:
        raise ConfigError("an empty series encodes nothing")
    if any(later < earlier for earlier, later in itertools.pairwise(moments)):
        raise ConfigError("a time series does not run backwards")
    made = bytearray()
    made.extend(varint_encode(moments[0]))
    if len(moments) == 1:
        return bytes(made)
    first_delta = moments[1] - moments[0]
    made.extend(varint_encode(first_delta))
    previous_delta = first_delta
    for earlier, later in itertools.pairwise(moments[1:]):
        delta = later - earlier
        made.extend(varint_encode(zigzag(delta - previous_delta)))
        previous_delta = delta
    return bytes(made)


def decode(raw: bytes) -> list[int]:
    """The series back."""
    if not raw:
        raise BadFormat("an empty buffer holds no series")
    moments = []
    first, at = varint_decode(raw)
    moments.append(first)
    if at >= len(raw):
        return moments
    delta, at = varint_decode(raw, at)
    moments.append(first + delta)
    while at < len(raw):
        wobble, at = varint_decode(raw, at)
        delta += unzigzag(wobble)
        moments.append(moments[-1] + delta)
    return moments


def flat_bytes(moments: list[int]) -> int:
    """The fixed width reference."""
    return 8 * len(moments)


@functools.cache
def _metronome(count: int = 20000, step: int = 10000) -> tuple[int, ...]:
    """The perfect scraper: exactly one sample per interval."""
    return tuple(1_700_000_000_000 + at * step for at in range(count))


@functools.cache
def _scrape(count: int = 20000, step: int = 10000, seed: int = 227) -> tuple[int, ...]:
    """The real scraper: the interval, plus scheduling jitter of a few milliseconds."""
    source = random.Random(seed)
    made = []
    clock = 1_700_000_000_000
    for _ in range(count):
        made.append(clock + source.randrange(-4, 5))
        clock += step
    return tuple(sorted(made))


@functools.cache
def _events(count: int = 20000, seed: int = 229) -> tuple[int, ...]:
    """Irregular arrivals, exponential gaps: the shape logs have."""
    source = random.Random(seed)
    made = []
    clock = 1_700_000_000_000
    for _ in range(count):
        clock += int(source.expovariate(1 / 8000)) + 1
        made.append(clock)
    return tuple(made)


@functools.cache
def the_metronome_buys_nothing_over_the_jittery_scrape_here() -> bool:
    """Perfect regularity and millisecond jitter cost identically: 20,006 bytes each.

    The claim was going to celebrate the metronome at a fraction of a byte per sample, and
    the measurement said one byte per sample, exactly the jittery scrape's cost, because a
    zero varint is a byte and this module stops at bytes. The celebrated sub-byte numbers
    of the Gorilla paper live in the layer underneath, bit packing that spends one bit on a
    zero second difference and run length on streaks of them. The second difference alone
    gets any regular-ish stream to one byte per sample, eight times under flat, and not a
    bit further; regularity beyond jitter pays only when the encoding can spend less than a
    byte, which is a statement about the container, not the mathematics.
    """
    metronome_raw = encode(list(_metronome()))
    scrape_raw = encode(list(_scrape()))
    per_sample = len(metronome_raw) / 20000
    return len(metronome_raw) == len(scrape_raw) and abs(per_sample - 1.0) < 0.01


@functools.cache
def jitter_costs_one_byte_per_sample() -> bool:
    """The realistic scrape stores at about one byte per sample, eight times under flat.

    Milliseconds of jitter make the second difference a small signed number, which zigzag
    keeps in one varint byte. This is the number that matters, because no production
    scraper is a metronome, and the encoding's value survives contact with real jitter at
    the cost of the run of zeros.
    """
    moments = list(_scrape())
    raw = encode(moments)
    per_sample = len(raw) / len(moments)
    return per_sample < 1.5 and decode(raw) == moments


@functools.cache
def irregular_events_fall_back_to_delta_cost() -> bool:
    """The event stream stores at about two bytes per sample: delta encoding in disguise.

    With no regular interval the second difference is as large as the first, and the scheme
    quietly becomes the varint module's delta encoding plus zigzag overhead. Still several
    times under flat, and the honest reading is that delta of delta buys nothing here, the
    plain delta was doing all the work.
    """
    moments = list(_events())
    raw = encode(moments)
    per_sample = len(raw) / len(moments)
    return 1.5 < per_sample < 3.5 and decode(raw) == moments


@functools.cache
def the_round_trip_is_exact_on_every_shape() -> bool:
    """All three streams, plus the edges: singletons, pairs, and repeated timestamps."""
    for moments in (
        list(_metronome(1000)),
        list(_scrape(1000)),
        list(_events(1000)),
        [5],
        [5, 5],
        [5, 5, 5, 100, 100],
    ):
        if decode(encode(moments)) != moments:
            return False
    return True


@functools.cache
def backwards_time_is_refused() -> bool:
    """A series that runs backwards raises at encode, where the mistake is.

    Timestamps run backwards in real systems, clock steps and NTP corrections, and the
    encoding could represent them, zigzag handles negative deltas. Refusing is a choice:
    the store's contract sorts by time, and a backwards timestamp is a corruption to
    surface, not a wobble to absorb silently.
    """
    try:
        encode([100, 90])
    except ConfigError:
        return True
    return False


def compare_the_shapes() -> list[dict]:
    """One row per stream shape."""
    rows = []
    for name, moments in (
        ("metronome", list(_metronome())),
        ("scrape", list(_scrape())),
        ("events", list(_events())),
    ):
        raw = encode(moments)
        rows.append(
            {
                "shape": name,
                "samples": len(moments),
                "encoded_bytes": len(raw),
                "flat_bytes": flat_bytes(moments),
                "bytes_per_sample": round(len(raw) / len(moments), 3),
            }
        )
    return rows


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "regularity_needs_bit_packing": (
            the_metronome_buys_nothing_over_the_jittery_scrape_here()
        ),
        "jitter_costs_a_byte": jitter_costs_one_byte_per_sample(),
        "events_fall_back_to_deltas": irregular_events_fall_back_to_delta_cost(),
        "round_trips_hold": the_round_trip_is_exact_on_every_shape(),
        "backwards_time_is_refused": backwards_time_is_refused(),
    }
