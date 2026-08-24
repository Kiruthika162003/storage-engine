from __future__ import annotations

import functools
from dataclasses import dataclass, field

from store.disk import Disk
from store.errors import Closed, ConfigError
from store.record import DELETE, Record
from store.wal import Log

# Group commit: the sync is the cost, so writers share one.
#
# The wal module measured the raw fact: a sync per record is the safest policy and the most
# expensive by count. Group commit is the standard escape. Writers add records to an open
# batch, and when the batch closes, one sync covers all of them. Everyone in the batch waits
# for the same sync, so latency is shared rather than added, and the sync count drops by the
# batch size while the durability guarantee stays exactly per record: nothing is acknowledged
# before the sync that covers it returns.
#
# The knob is when to close the batch, and the two limits, size and operations, express the
# same tension as every buffer in the package: close early and pay more syncs, close late and
# hold everyone's acknowledgement hostage to the slowest arrival.


@dataclass
class Ticket:
    """What a writer holds while its batch is open: settled once the sync lands."""

    sequence: int
    batch: int
    settled: bool = field(default=False)


@dataclass
class Committer:
    """Collects writers into batches and syncs once per batch."""

    log: Log = field(default_factory=lambda: Log(disk=Disk(name="WAL"), policy="never"))
    close_at: int = field(default=8)
    open_batch: list[tuple[Record, Ticket]] = field(default_factory=list)
    batches: int = field(default=0)
    settled: int = field(default=0)
    sequence: int = field(default=0)
    closed: bool = field(default=False)

    def __post_init__(self) -> None:
        if self.close_at < 1:
            raise ConfigError(f"{self.close_at} is not a batch size")

    def submit(self, key: bytes, value: bytes, kind: int = 0) -> Ticket:
        """Join the open batch, receiving a ticket that settles at the sync."""
        if self.closed:
            raise Closed("the committer is closed")
        self.sequence += 1
        record = Record(key=key, sequence=self.sequence, kind=kind, value=value)
        ticket = Ticket(sequence=self.sequence, batch=self.batches)
        self.open_batch.append((record, ticket))
        if len(self.open_batch) >= self.close_at:
            self.flush()
        return ticket

    def delete(self, key: bytes) -> Ticket:
        """A delete is a submission like any other."""
        return self.submit(key, b"", DELETE)

    def flush(self) -> int:
        """Close the batch: one append, one sync, every ticket settles together."""
        if not self.open_batch:
            return 0
        records = [record for record, _ in self.open_batch]
        self.log.append(records)
        self.log.disk.sync()
        for _, ticket in self.open_batch:
            ticket.settled = True
            self.settled += 1
        count = len(self.open_batch)
        self.open_batch = []
        self.batches += 1
        return count

    def close(self) -> None:
        """Flush what is open and stop accepting."""
        self.flush()
        self.closed = True

    @property
    def syncs(self) -> int:
        """How many times the disk was synced."""
        return self.log.disk.syncs

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "close_at": self.close_at,
            "batches": self.batches,
            "settled": self.settled,
            "syncs": self.syncs,
            "pending": len(self.open_batch),
        }


@functools.cache
def _committed(writes: int = 4000, close_at: int = 8) -> Committer:
    """A committer with a stream of writes pushed through it."""
    made = Committer(close_at=close_at)
    for at in range(writes):
        made.submit(f"k{at:06d}".encode(), bytes(8))
    made.close()
    return made


@functools.cache
def the_sync_count_drops_by_the_batch_size() -> bool:
    """Four thousand writes cost 4,000 syncs alone and 500 in batches of eight.

    The division is exact because the stream divides evenly, and the guarantee did not move:
    every ticket settles at a sync that covers its record, so an acknowledged write is a
    durable write in both configurations. What changed is only who shares each sync.
    """
    alone = _committed(4000, 1)
    grouped = _committed(4000, 8)
    return alone.syncs == 4000 and grouped.syncs == 500 and grouped.settled == 4000


@functools.cache
def nothing_settles_before_its_sync() -> bool:
    """A ticket in an open batch is not settled, and settles the moment the batch closes.

    This is the line between group commit and lying. A committer that settles tickets on
    submit has acknowledged a write the disk has not seen, and the crash that catches it
    produces the worst bug class in storage: an acknowledged, vanished write.
    """
    made = Committer(close_at=4)
    ticket = made.submit(b"k", b"v")
    before = ticket.settled
    made.flush()
    return not before and ticket.settled


@functools.cache
def a_batch_settles_together() -> bool:
    """Every ticket in a batch settles at the same sync, first arrival and last alike.

    The first writer into a batch waits longest, which is the latency cost of grouping: its
    wait is everyone else's arrival time plus one sync. The sync it gets is the same one, so
    the trade is pure latency for throughput with durability constant.
    """
    made = Committer(close_at=3)
    tickets = [made.submit(f"k{at}".encode(), b"v") for at in range(3)]
    return all(ticket.settled for ticket in tickets) and made.batches == 1


@functools.cache
def a_closed_committer_refuses_and_flushes_first() -> bool:
    """Close settles the stragglers, then refuses new work.

    The other order, refusing before flushing, strands the open batch unacknowledged forever,
    which is a hang at every caller holding a ticket.
    """
    made = Committer(close_at=100)
    ticket = made.submit(b"k", b"v")
    made.close()
    if not ticket.settled:
        return False
    try:
        made.submit(b"x", b"y")
    except Closed:
        return True
    return False


def compare_the_batch_sizes(writes: int = 4000) -> list[dict]:
    """One row per close threshold, syncs against latency shape."""
    rows = []
    for close_at in (1, 2, 4, 8, 16, 64):
        made = _committed(writes, close_at)
        rows.append(
            {
                "close_at": close_at,
                "syncs": made.syncs,
                "writes_per_sync": round(writes / max(made.syncs, 1), 2),
            }
        )
    return rows


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "syncs_drop_by_the_batch": the_sync_count_drops_by_the_batch_size(),
        "nothing_settles_early": nothing_settles_before_its_sync(),
        "a_batch_settles_together": a_batch_settles_together(),
        "close_flushes_first": a_closed_committer_refuses_and_flushes_first(),
    }
