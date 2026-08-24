from __future__ import annotations

import functools
from dataclasses import dataclass, field

from store.errors import ConfigError

# Hybrid logical clocks: timestamps that respect causality when the wall clock lies.
#
# The store stamps versions with a counter, which orders perfectly and means nothing to a
# human; operators want wall time in their ids. The wall clock cannot be trusted with
# ordering: NTP steps it backward, and an id scheme that trusts it mints a later id below
# an earlier one, breaking every structure this package built on monotonic sequence. The
# hybrid clock keeps the wall time when it can and a logical counter when it must: the
# timestamp is the pair (wall, logical), advanced to max(wall now, last seen) with the
# counter breaking ties, so ids never go backward, causality is preserved across message
# exchange, and the wall component stays within the clock error of true time. The module
# steps the clock backward on purpose and shows each scheme's behaviour.


@dataclass(frozen=True)
class Stamp:
    """One hybrid timestamp."""

    wall: int
    logical: int

    def key(self) -> tuple[int, int]:
        return (self.wall, self.logical)


@dataclass
class HybridClock:
    """The clock, fed wall readings that may misbehave."""

    last: Stamp = field(default_factory=lambda: Stamp(wall=0, logical=0))

    def now(self, wall_reading: int) -> Stamp:
        """A fresh stamp, never below the last."""
        if wall_reading < 0:
            raise ConfigError("wall readings are not negative")
        if wall_reading > self.last.wall:
            made = Stamp(wall=wall_reading, logical=0)
        else:
            made = Stamp(wall=self.last.wall, logical=self.last.logical + 1)
        self.last = made
        return made

    def observe(self, other: Stamp, wall_reading: int) -> Stamp:
        """A stamp after seeing another clock's stamp: causality folded in."""
        top_wall = max(wall_reading, self.last.wall, other.wall)
        if top_wall == wall_reading and top_wall > self.last.wall and top_wall > other.wall:
            made = Stamp(wall=top_wall, logical=0)
        else:
            logical = 0
            if top_wall == self.last.wall:
                logical = max(logical, self.last.logical + 1)
            if top_wall == other.wall:
                logical = max(logical, other.logical + 1)
            made = Stamp(wall=top_wall, logical=logical)
        self.last = made
        return made


def wall_only_ids(readings: list[int]) -> list[int]:
    """The naive scheme: the id is the reading."""
    return list(readings)


STEPPED_READINGS = (1000, 1005, 1010, 940, 941, 950, 1011, 1012)


@functools.cache
def wall_ids_go_backward_under_a_step() -> bool:
    """The NTP step makes the naive scheme mint id 940 after id 1010.

    Every structure keyed on these ids now believes the later write is older: the memtable
    resolves the wrong winner, the changefeed replays out of order, the timekey module's
    feed shows the newest post in fourth place. One clock correction, and ordering breaks
    everywhere ordering was assumed.
    """
    ids = wall_only_ids(list(STEPPED_READINGS))
    return ids != sorted(ids)


@functools.cache
def hybrid_stamps_never_go_backward() -> bool:
    """The same stepped readings through the hybrid clock: strictly increasing stamps.

    Through the backward step the wall component holds at its high water mark and the
    logical counter climbs, so order survives the correction; when the wall catches back
    up past the mark, the logical resets to zero and the stamp is wall time again. The
    lie is bounded and temporary, the ordering is neither.
    """
    clock = HybridClock()
    stamps = [clock.now(reading) for reading in STEPPED_READINGS]
    keys = [stamp.key() for stamp in stamps]
    return keys == sorted(keys) and len(set(keys)) == len(keys)


@functools.cache
def causality_survives_an_exchange_between_skewed_clocks() -> bool:
    """A message from a fast clock to a slow one: the reply stamps above the request.

    The receiver's wall reads 900 while the sender's stamp says 1000, and observe folds
    the sender's stamp in, so the reply's stamp exceeds the request's despite the
    receiver's clock being behind. Happened-before survives skew, which is the property
    replication and the changefeed would need the moment two machines exist.
    """
    sender = HybridClock()
    receiver = HybridClock()
    request = sender.now(1000)
    reply = receiver.observe(request, 900)
    return reply.key() > request.key() and reply.wall == 1000


@functools.cache
def the_wall_component_tracks_true_time_when_clocks_behave() -> bool:
    """With honest readings the logical part stays zero and the stamp is the wall time.

    The hybrid scheme costs nothing when nothing is wrong: operators read real timestamps,
    the logical counter is dormant, and the whole mechanism only spends its second
    component during the corrections it exists for.
    """
    clock = HybridClock()
    stamps = [clock.now(reading) for reading in (10, 20, 30, 45)]
    return all(stamp.logical == 0 for stamp in stamps) and stamps[-1].wall == 45


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "wall_ids_go_backward": wall_ids_go_backward_under_a_step(),
        "hybrid_stamps_do_not": hybrid_stamps_never_go_backward(),
        "causality_survives_skew": causality_survives_an_exchange_between_skewed_clocks(),
        "honest_clocks_cost_nothing": the_wall_component_tracks_true_time_when_clocks_behave(),
    }
