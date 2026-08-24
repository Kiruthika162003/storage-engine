from __future__ import annotations

import random
from dataclasses import dataclass, field

from store.btree import Tree
from store.compaction import Levelled, Tiered
from store.engine import Store
from store.record import Record

# Four implementations of the same contract, disagreeing with each other or not.
#
# The model checker compares the store to a dictionary. This compares the store to the other
# storage structures in the package, which is a different kind of evidence: the dictionary is
# simple enough to be right, and the structures are complicated enough that a shared bug in a
# shared idea, say the tombstone rule, could pass the model check in one and fail it in
# another. Running the same stream through all of them and diffing the results catches the
# family of bugs that live in one implementation's reading of the shared idea.

DESIGNS = ("store", "btree", "levelled", "tiered")


@dataclass
class Divergence:
    """One key where two designs disagree."""

    key: bytes
    answers: dict[str, bytes | None] = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Flat mapping for reports."""
        return {"key": repr(self.key), "answers": {k: repr(v) for k, v in self.answers.items()}}


@dataclass
class Fleet:
    """The four designs, driven together."""

    store: Store = field(default_factory=lambda: Store(flush_at=500, fold_at=4))
    tree: Tree = field(default_factory=Tree)
    levelled: Levelled = field(default_factory=Levelled)
    tiered: Tiered = field(default_factory=Tiered)
    levelled_buffer: dict = field(default_factory=dict)
    tiered_buffer: dict = field(default_factory=dict)
    sequence: int = field(default=0)
    touched: set = field(default_factory=set)

    def put(self, key: bytes, value: bytes) -> None:
        """One write into every design."""
        self.sequence += 1
        self.touched.add(key)
        self.store.put(key, value)
        self.tree.put(key, value)
        self.levelled_buffer[key] = Record(key=key, sequence=self.sequence, value=value)
        self.tiered_buffer[key] = Record(key=key, sequence=self.sequence, value=value)
        self._spill()

    def delete(self, key: bytes) -> None:
        """One delete into every design."""
        self.sequence += 1
        self.touched.add(key)
        self.store.delete(key)
        self.tree.remove(key)
        tombstone = Record(key=key, sequence=self.sequence, kind=1, value=b"")
        self.levelled_buffer[key] = tombstone
        self.tiered_buffer[key] = tombstone
        self._spill()

    def _spill(self) -> None:
        """Flush the compaction policies' buffers the way an engine would."""
        if len(self.levelled_buffer) >= 500:
            batch = sorted(self.levelled_buffer.values(), key=lambda one: one.order)
            self.levelled.flush(batch)
            self.levelled_buffer = {}
            batch = sorted(self.tiered_buffer.values(), key=lambda one: one.order)
            self.tiered.flush(batch)
            self.tiered_buffer = {}

    def answers(self, key: bytes) -> dict[str, bytes | None]:
        """Every design's answer for one key."""
        found: dict[str, bytes | None] = {
            "store": self.store.get(key),
            "btree": self.tree.get(key),
        }
        for name, policy, buffer in (
            ("levelled", self.levelled, self.levelled_buffer),
            ("tiered", self.tiered, self.tiered_buffer),
        ):
            if key in buffer:
                record = buffer[key]
                found[name] = None if record.kind == 1 else record.value
            else:
                record = policy.get(key)
                value = record.value if record is not None else None
                found[name] = value
        return found

    def divergences(self) -> list[Divergence]:
        """Every key any design answers differently."""
        found = []
        for key in sorted(self.touched):
            answers = self.answers(key)
            if len(set(answers.values())) > 1:
                found.append(Divergence(key=key, answers=answers))
        return found


def run(writes: int = 4000, keys: int = 800, seed: int = 0) -> dict:
    """One stream through the fleet, diffed at the end."""
    source = random.Random(seed)
    fleet = Fleet()
    for _ in range(writes):
        key = f"k{source.randrange(keys):05d}".encode()
        if source.random() < 0.15:
            fleet.delete(key)
        else:
            fleet.put(key, source.randbytes(10))
    diverged = fleet.divergences()
    return {
        "writes": writes,
        "keys_touched": len(fleet.touched),
        "divergences": len(diverged),
        "clean": not diverged,
        "first": diverged[0].as_dict() if diverged else None,
    }
