from __future__ import annotations

import contextlib
import functools
import random
from dataclasses import dataclass, field

from store.errors import Closed, ConfigError

# Changefeeds: readers tailing the write stream, and the slow one's bill.
#
# A cache invalidator, an index builder, a replica: consumers that want every write, in
# order, from where they left off. The sequence number the store already stamps on every
# record is the cursor, and the feed is a buffer of recent records plus per subscriber
# positions. The design decision with teeth is the buffer bound: an unbounded buffer lets
# one dead subscriber hold every record forever, the mvcc horizon problem in feed clothing,
# and the bound converts the slow subscriber's problem into the slow subscriber's problem,
# a lost position and a forced resync, instead of everyone's memory.

BUFFER_RECORDS = 1000


@dataclass
class Feed:
    """The buffer and the subscribers."""

    buffer_records: int = field(default=BUFFER_RECORDS)
    held: list[tuple[int, bytes, bytes]] = field(default_factory=list)
    floor: int = field(default=0)
    sequence: int = field(default=0)
    positions: dict[str, int] = field(default_factory=dict)
    evicted_past: dict[str, bool] = field(default_factory=dict)
    resyncs: int = field(default=0)

    def __post_init__(self) -> None:
        if self.buffer_records < 1:
            raise ConfigError(f"{self.buffer_records} is not a buffer")

    def publish(self, key: bytes, value: bytes) -> int:
        """One write into the feed."""
        self.sequence += 1
        self.held.append((self.sequence, key, value))
        while len(self.held) > self.buffer_records:
            dropped_sequence, _, _ = self.held.pop(0)
            self.floor = dropped_sequence
        return self.sequence

    def subscribe(self, name: str) -> None:
        """A new subscriber starts at the present."""
        if name in self.positions:
            raise ConfigError(f"{name} is already subscribed")
        self.positions[name] = self.sequence

    def poll(self, name: str, limit: int = 100) -> list[tuple[int, bytes, bytes]]:
        """The subscriber's next batch, or a forced resync if it fell off the buffer."""
        if name not in self.positions:
            raise Closed(f"{name} is not subscribed")
        position = self.positions[name]
        if position < self.floor:
            self.resyncs += 1
            self.positions[name] = self.sequence
            raise Closed(f"{name} fell behind the buffer; resync from a snapshot")
        found = [entry for entry in self.held if entry[0] > position][:limit]
        if found:
            self.positions[name] = found[-1][0]
        return found

    def lag(self, name: str) -> int:
        """How far behind a subscriber runs."""
        return self.sequence - self.positions[name]


@functools.cache
def a_keeping_up_subscriber_sees_every_write_in_order() -> bool:
    """Five thousand writes polled in batches arrive complete, ordered, without gaps.

    The at-least-once floor of any feed: a subscriber that polls faster than the buffer
    turns over misses nothing, and the sequence numbers it receives are consecutive, which
    the test checks rather than assumes because off by one cursor updates deliver either
    duplicates or holes, both silently.
    """
    feed = Feed()
    feed.subscribe("indexer")
    received = []
    source = random.Random(257)
    for at in range(5000):
        feed.publish(f"k{at:05d}".encode(), b"v")
        if source.random() < 0.3:
            received.extend(entry[0] for entry in feed.poll("indexer", limit=500))
    received.extend(entry[0] for entry in feed.poll("indexer", limit=5000))
    return received == list(range(1, 5001))


@functools.cache
def a_slow_subscriber_is_cut_loose_not_carried() -> bool:
    """A subscriber that stops polling is forced to resync once the buffer laps it.

    Two thousand writes into a thousand record buffer: the sleeper's position falls below
    the floor, its next poll raises with the resync instruction, and the buffer stayed at
    its bound throughout. The alternative was the buffer growing without limit on the
    sleeper's behalf, the mvcc horizon lesson with a subscription attached.
    """
    feed = Feed()
    feed.subscribe("sleeper")
    for at in range(2000):
        feed.publish(f"k{at:05d}".encode(), b"v")
    if len(feed.held) != BUFFER_RECORDS:
        return False
    try:
        feed.poll("sleeper")
    except Closed:
        return feed.resyncs == 1
    return False


@functools.cache
def the_resync_position_is_the_present() -> bool:
    """After the forced resync the subscriber follows cleanly from now.

    The resync costs the subscriber a snapshot read outside the feed, and what the feed
    owes it afterwards is a clean tail: the next poll returns only post-resync writes, in
    order, no stragglers from the lapped past.
    """
    feed = Feed()
    feed.subscribe("sleeper")
    for at in range(2000):
        feed.publish(f"old{at:05d}".encode(), b"v")
    with contextlib.suppress(Closed):
        feed.poll("sleeper")
    feed.publish(b"fresh", b"v")
    found = feed.poll("sleeper")
    return [entry[1] for entry in found] == [b"fresh"]


@functools.cache
def lag_is_visible_before_it_is_fatal() -> bool:
    """The lag crosses half the buffer with no resync, which is the alarm's window.

    A subscriber polling every third publish runs a measurable lag, and the distance
    between lag and buffer size is the operator's warning time. A feed without the meter
    delivers its first signal as the resync, which is the alarm ringing after the fire.
    """
    feed = Feed(buffer_records=300)
    feed.subscribe("slowish")
    worst = 0
    for at in range(600):
        feed.publish(f"k{at:05d}".encode(), b"v")
        if at % 25 == 24:
            feed.poll("slowish", limit=17)
            worst = max(worst, feed.lag("slowish"))
    return worst > 150 and feed.resyncs == 0


@functools.cache
def two_subscribers_run_independent_cursors() -> bool:
    """Each subscriber's poll moves only its own position."""
    feed = Feed()
    feed.subscribe("a")
    feed.subscribe("b")
    for at in range(10):
        feed.publish(f"k{at}".encode(), b"v")
    feed.poll("a", limit=10)
    return feed.lag("a") == 0 and feed.lag("b") == 10


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "keeping_up_sees_everything": a_keeping_up_subscriber_sees_every_write_in_order(),
        "the_slow_are_cut_loose": a_slow_subscriber_is_cut_loose_not_carried(),
        "resync_lands_at_the_present": the_resync_position_is_the_present(),
        "lag_warns_before_the_cut": lag_is_visible_before_it_is_fatal(),
        "cursors_are_independent": two_subscribers_run_independent_cursors(),
    }
