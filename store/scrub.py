from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

from store.checksum import crc32
from store.errors import BadChecksum, ConfigError

# Bit rot, and why a store reads data nobody asked for.
#
# A checksum only helps when something computes it, and a block that is written once and read
# never is a block whose corruption is discovered the day it is finally needed, which is the
# worst possible day: usually during a recovery, when the store is already down to its last
# copy. Scrubbing is the fix, a background pass that reads every block and verifies it against
# its checksum on a schedule, converting silent decay into scheduled repair.
#
# The model gives each block a small chance of rotting per tick and measures the thing that
# matters: how much rot exists undetected, as a function of how often the scrubber comes
# around. The answer has a clean shape, undetected rot is proportional to the scrub interval,
# and the proportionality is the budget argument for spending read bandwidth on data nobody
# asked for.


@dataclass
class Block:
    """One stored block: its bytes, its checksum from write time, its fate."""

    payload: bytearray
    written_sum: int
    rotted_at: int = field(default=-1)

    def intact(self) -> bool:
        """Whether the bytes still match the write time checksum."""
        return crc32(bytes(self.payload)) == self.written_sum


@dataclass
class Farm:
    """A population of blocks decaying at a rate, and a scrubber walking them."""

    blocks: int
    rot_per_tick: float
    seed: int = field(default=53)
    held: list[Block] = field(default_factory=list)
    now: int = field(default=0)
    detected: int = field(default=0)
    repaired: int = field(default=0)
    scrub_reads: int = field(default=0)
    source: random.Random = field(default=None)

    def __post_init__(self) -> None:
        if self.blocks < 1:
            raise ConfigError(f"{self.blocks} is not a population")
        if not 0.0 <= self.rot_per_tick < 1.0:
            raise ConfigError(f"{self.rot_per_tick} is not a rot probability")
        self.source = random.Random(self.seed)
        for _ in range(self.blocks):
            payload = bytearray(self.source.randbytes(64))
            self.held.append(Block(payload=payload, written_sum=crc32(bytes(payload))))

    def tick(self) -> None:
        """One unit of time: every intact block risks a flipped bit."""
        self.now += 1
        for block in self.held:
            if block.rotted_at < 0 and self.source.random() < self.rot_per_tick:
                block.payload[self.source.randrange(len(block.payload))] ^= 0x04
                block.rotted_at = self.now

    def undetected(self) -> int:
        """Rotten blocks nothing has noticed yet."""
        return sum(1 for block in self.held if block.rotted_at >= 0 and not block.intact())

    def scrub(self) -> int:
        """Read every block, verify, repair what fails.

        Repair here is a rewrite from a good copy, modelled as restoring the payload; the
        interesting quantity is detection, and repair is what detection is for.
        """
        found = 0
        for block in self.held:
            self.scrub_reads += 1
            if not block.intact():
                found += 1
                self.detected += 1
                fresh = bytearray(self.source.randbytes(64))
                block.payload = fresh
                block.written_sum = crc32(bytes(fresh))
                block.rotted_at = -1
                self.repaired += 1
        return found

    def read(self, at: int) -> bytes:
        """A foreground read, which verifies like any read should."""
        block = self.held[at % len(self.held)]
        if not block.intact():
            raise BadChecksum(f"block {at} failed its checksum on read")
        return bytes(block.payload)


def run(farm: Farm, ticks: int, scrub_every: int) -> dict:
    """Advance a farm with a scrubber on an interval, reporting the rot picture."""
    peak_undetected = 0
    for at in range(1, ticks + 1):
        farm.tick()
        if scrub_every and at % scrub_every == 0:
            farm.scrub()
        peak_undetected = max(peak_undetected, farm.undetected())
    return {
        "ticks": ticks,
        "scrub_every": scrub_every,
        "detected": farm.detected,
        "repaired": farm.repaired,
        "undetected_at_end": farm.undetected(),
        "peak_undetected": peak_undetected,
        "scrub_reads": farm.scrub_reads,
    }


@functools.cache
def undetected_rot_scales_with_the_scrub_interval() -> bool:
    """Scrub four times less often and carry about four times the undetected rot.

    The peak undetected count tracks the interval because rot accumulates linearly between
    scrubs and each scrub takes it to zero. The proportionality is the whole budgeting
    argument: the read bandwidth spent scrubbing buys down the window in which a second
    failure meets an undetected first one.
    """
    tight = run(Farm(blocks=2000, rot_per_tick=0.0005, seed=54), 400, 10)
    loose = run(Farm(blocks=2000, rot_per_tick=0.0005, seed=54), 400, 40)
    ratio = loose["peak_undetected"] / max(tight["peak_undetected"], 1)
    return 2.0 < ratio < 8.0


@functools.cache
def an_unscrubbed_farm_accumulates_rot_without_limit() -> bool:
    """No scrubber, four hundred ticks, and a third of the farm is silently bad.

    Every one of those blocks answers its checksum with a lie the moment it is read, so the
    failure is not hidden forever, it is deferred to the read, and the reads that matter most,
    recovery reads, are exactly the ones that come last.
    """
    farm = Farm(blocks=2000, rot_per_tick=0.001, seed=55)
    made = run(farm, 400, 0)
    return made["undetected_at_end"] > 500 and made["detected"] == 0


@functools.cache
def a_foreground_read_refuses_rot_rather_than_serving_it() -> bool:
    """A read of a rotted block raises, which is the checksum doing its one job.

    The alternative, serving the bytes, converts a storage failure into a silent application
    error, and the checksum exists precisely to make that conversion impossible.
    """
    farm = Farm(blocks=10, rot_per_tick=0.0, seed=56)
    farm.held[3].payload[0] ^= 0xFF
    try:
        farm.read(3)
    except BadChecksum:
        return True
    return False


@functools.cache
def a_scrub_repairs_what_it_finds_and_the_farm_recovers() -> bool:
    """After heavy rot, one scrub detects everything and the undetected count is zero.

    Detection without repair would just move the bad news earlier, which has value, and
    repair from a good copy is what makes the scrubber a maintenance tool rather than an
    alarm. The model restores from thin air; a real store restores from a replica, and the
    scrub interval bounds how stale that replica read can be.
    """
    farm = Farm(blocks=1000, rot_per_tick=0.002, seed=57)
    for _ in range(200):
        farm.tick()
    rotted = farm.undetected()
    farm.scrub()
    return rotted > 100 and farm.undetected() == 0 and farm.repaired == rotted


@functools.cache
def scrub_reads_are_the_price_and_they_dwarf_the_finds() -> bool:
    """Two hundred thousand scrub reads to find a few hundred bad blocks.

    Almost every scrub read verifies a healthy block, which is the point, not waste: the
    scrubber's product is confidence about the healthy blocks, and the repairs are a side
    effect. Pricing it as reads per repair misprices it, the same mistake as pricing a filter
    by its false positive count alone.
    """
    farm = Farm(blocks=2000, rot_per_tick=0.0005, seed=58)
    made = run(farm, 400, 10)
    return made["scrub_reads"] > 50000 and made["detected"] < made["scrub_reads"] / 50


def compare_the_intervals(ticks: int = 400) -> list[dict]:
    """One row per scrub interval, undetected rot against read cost."""
    rows = []
    for interval in (5, 10, 20, 40, 80, 0):
        farm = Farm(blocks=2000, rot_per_tick=0.0005, seed=59)
        rows.append(run(farm, ticks, interval))
    return rows


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "rot_scales_with_the_interval": undetected_rot_scales_with_the_scrub_interval(),
        "unscrubbed_rot_accumulates": an_unscrubbed_farm_accumulates_rot_without_limit(),
        "reads_refuse_rot": a_foreground_read_refuses_rot_rather_than_serving_it(),
        "scrubs_repair": a_scrub_repairs_what_it_finds_and_the_farm_recovers(),
        "the_price_is_reads": scrub_reads_are_the_price_and_they_dwarf_the_finds(),
    }
