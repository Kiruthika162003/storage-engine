from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

from store.errors import ConfigError

# The workloads every evaluation shares, named and fixed.
#
# An evaluation is only comparable to another if they ran the same stream, so the streams live
# here, are seeded, and are cached. The names follow the YCSB convention loosely: a read heavy
# mix, a balanced mix, a scan mix, an insert mix, because those are the shapes real systems
# get sized against.

OPERATIONS = ("get", "put", "delete", "scan")


@dataclass(frozen=True)
class Operation:
    """One step of a workload."""

    kind: str
    key: bytes
    value: bytes = field(default=b"")
    length: int = field(default=0)


@dataclass(frozen=True)
class Mix:
    """A named blend of operations over a keyspace."""

    name: str
    gets: float
    puts: float
    deletes: float
    scans: float
    keys: int
    operations: int
    value_bytes: int = field(default=64)
    hot_share: float = field(default=0.0)
    seed: int = field(default=41)

    def __post_init__(self) -> None:
        total = self.gets + self.puts + self.deletes + self.scans
        if abs(total - 1.0) > 1e-9:
            raise ConfigError(f"{self.name} blends to {total}, not 1")

    def _key(self, source: random.Random) -> bytes:
        """One key, hot or uniform."""
        if self.hot_share and source.random() < 0.9:
            hot = max(int(self.keys * self.hot_share), 1)
            return f"k{source.randrange(hot):08d}".encode()
        return f"k{source.randrange(self.keys):08d}".encode()

    def stream(self) -> list[Operation]:
        """The workload as a list of operations."""
        source = random.Random(self.seed)
        made = []
        for _ in range(self.operations):
            draw = source.random()
            key = self._key(source)
            if draw < self.gets:
                made.append(Operation(kind="get", key=key))
            elif draw < self.gets + self.puts:
                made.append(
                    Operation(kind="put", key=key, value=source.randbytes(self.value_bytes))
                )
            elif draw < self.gets + self.puts + self.deletes:
                made.append(Operation(kind="delete", key=key))
            else:
                made.append(Operation(kind="scan", key=key, length=20))
        return made

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "name": self.name,
            "gets": self.gets,
            "puts": self.puts,
            "deletes": self.deletes,
            "scans": self.scans,
            "keys": self.keys,
            "operations": self.operations,
        }


READ_HEAVY = Mix(
    name="read_heavy", gets=0.95, puts=0.05, deletes=0.0, scans=0.0, keys=5000, operations=20000
)
BALANCED = Mix(
    name="balanced", gets=0.5, puts=0.45, deletes=0.05, scans=0.0, keys=5000, operations=20000
)
INSERT_HEAVY = Mix(
    name="insert_heavy",
    gets=0.05,
    puts=0.95,
    deletes=0.0,
    scans=0.0,
    keys=50000,
    operations=20000,
)
SCAN_HEAVY = Mix(
    name="scan_heavy", gets=0.25, puts=0.25, deletes=0.0, scans=0.5, keys=5000, operations=8000
)
HOT_READS = Mix(
    name="hot_reads",
    gets=0.95,
    puts=0.05,
    deletes=0.0,
    scans=0.0,
    keys=5000,
    operations=20000,
    hot_share=0.05,
)

MIXES = (READ_HEAVY, BALANCED, INSERT_HEAVY, SCAN_HEAVY, HOT_READS)


@functools.cache
def stream(name: str) -> tuple[Operation, ...]:
    """A named mix's stream, cached so every evaluation sees the same one."""
    for mix in MIXES:
        if mix.name == name:
            return tuple(mix.stream())
    raise ConfigError(f"{name} is not a workload")
