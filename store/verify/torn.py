from __future__ import annotations

import random
from dataclasses import dataclass, field

from store.manifest import Edit, Manifest, add, replay
from store.record import Record
from store.wal import frame, recover

# Damage done to the bytes themselves, and what the framing turns it into.
#
# The disk model's crash loses clean suffixes. Real media do worse: a sector written during
# power loss can hold half the new bytes, a bit can flip in a cable, a firmware bug can write
# the right data at the wrong offset. The framing cannot prevent any of that. What it can do,
# and what these checks measure, is turn every such event into one of two visible outcomes: a
# clean stop at the damage, or an intact record. The disaster is the third outcome, a record
# that reads back changed, and the claim is that the checksum makes the third outcome
# unreachable at any single point of damage.


@dataclass
class Outcome:
    """What one damaged log recovered to."""

    damage: str
    at: int
    recovered: int
    total: int
    changed: list[bytes] = field(default_factory=list)

    @property
    def stopped_clean(self) -> bool:
        """Whether the damage produced a clean stop rather than a changed record."""
        return not self.changed

    def as_dict(self) -> dict:
        """Flat mapping for reports."""
        return {
            "damage": self.damage,
            "at": self.at,
            "recovered": self.recovered,
            "total": self.total,
            "changed": [repr(one) for one in self.changed],
            "stopped_clean": self.stopped_clean,
        }


def _log(records: int = 200, seed: int = 0) -> tuple[bytes, list[Record]]:
    """A log of framed records and the records themselves."""
    source = random.Random(seed)
    made = []
    raw = bytearray()
    for at in range(records):
        record = Record(
            key=f"k{at:05d}".encode(), sequence=at + 1, value=source.randbytes(12)
        )
        made.append(record)
        raw.extend(frame(record.encode()))
    return bytes(raw), made


def _judge(damage: str, at: int, raw: bytes, originals: list[Record]) -> Outcome:
    """Recover a damaged log and compare what came back to what was written."""
    recovery = recover(raw)
    by_sequence = {record.sequence: record for record in originals}
    changed = [
        record.key
        for record in recovery.records
        if by_sequence.get(record.sequence) != record
    ]
    return Outcome(
        damage=damage,
        at=at,
        recovered=len(recovery.records),
        total=len(originals),
        changed=changed,
    )


def truncate(at: int, records: int = 200, seed: int = 0) -> Outcome:
    """Cut the log at a byte, which is the ordinary crash."""
    raw, originals = _log(records, seed)
    return _judge("truncate", at, raw[:at], originals)


def flip(at: int, records: int = 200, seed: int = 0) -> Outcome:
    """Flip one bit, which is corruption in place."""
    raw, originals = _log(records, seed)
    damaged = bytearray(raw)
    damaged[at % len(damaged)] ^= 0x40
    return _judge("flip", at, bytes(damaged), originals)


def tear(at: int, records: int = 200, seed: int = 0, sector: int = 512) -> Outcome:
    """Zero the tail of one sector, which is the torn write itself."""
    raw, originals = _log(records, seed)
    damaged = bytearray(raw)
    start = (at // sector) * sector + sector // 2
    end = min(start + sector // 2, len(damaged))
    for one in range(start, end):
        damaged[one] = 0
    return _judge("tear", at, bytes(damaged), originals)


def misplace(at: int, records: int = 200, seed: int = 0, offset: int = 64) -> Outcome:
    """Write a stretch of good bytes at the wrong place, the firmware bug."""
    raw, originals = _log(records, seed)
    damaged = bytearray(raw)
    stretch = damaged[at : at + offset]
    target = (at + len(damaged) // 2) % max(len(damaged) - offset, 1)
    damaged[target : target + offset] = stretch
    return _judge("misplace", at, bytes(damaged), originals)


DAMAGES = (truncate, flip, tear, misplace)


def sweep(points: int = 60, records: int = 200) -> dict:
    """Every damage kind at many points, hunting the third outcome."""
    raw, _ = _log(records)
    length = len(raw)
    source = random.Random(99)
    changed = 0
    stopped = 0
    total = 0
    for damage in DAMAGES:
        for _ in range(points):
            outcome = damage(source.randrange(length), records)
            total += 1
            if outcome.stopped_clean:
                stopped += 1
            else:
                changed += 1
    return {
        "damages": len(DAMAGES),
        "points_each": points,
        "total": total,
        "stopped_clean": stopped,
        "changed_records": changed,
        "clean": changed == 0,
    }


def manifest_sweep(points: int = 120) -> dict:
    """The same discipline applied to the manifest's edit log."""
    manifest = Manifest()
    for at in range(1, 60):
        manifest.install(Edit(changes=(add(at, 0, 100),)))
    raw = manifest.disk.read()
    source = random.Random(7)
    bad = 0
    for _ in range(points):
        damaged = bytearray(raw)
        damaged[source.randrange(len(damaged))] ^= 0x10
        found = replay(bytes(damaged))
        for number, file in found.version.files.items():
            if file.level != 0 or file.records != 100 or number >= 60:
                bad += 1
                break
    return {"points": points, "poisoned_versions": bad, "clean": bad == 0}
