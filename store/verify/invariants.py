from __future__ import annotations

from dataclasses import dataclass, field

from store.engine import Store
from store.record import DELETE

# What must be true of a store at rest, checked structurally rather than through reads.
#
# A read can be right by accident: two bugs can cancel, a stale version can happen to hold the
# right bytes, a missing tombstone can hide behind a filter false positive. The invariants here
# look at the structures directly, so a store that answers correctly for the wrong reasons
# still fails.
#
# Each check returns the violations it found rather than a boolean, because the first question
# after a failure is always which key, and a boolean throws that away.


@dataclass
class Violation:
    """One broken invariant, with enough context to start debugging."""

    check: str
    detail: str
    key: bytes = field(default=b"")

    def as_dict(self) -> dict:
        """Flat mapping for reports."""
        return {"check": self.check, "detail": self.detail, "key": repr(self.key)}


def sorted_tables(store: Store) -> list[Violation]:
    """Every file's records are sorted by key then newest first."""
    found = []
    for table in store.tables:
        orders = [record.order for record in table.records]
        if orders != sorted(orders):
            found.append(
                Violation(
                    check="sorted_tables",
                    detail=f"table {table.number} is out of order",
                )
            )
    return found


def unique_keys_per_table(store: Store) -> list[Violation]:
    """No file holds two versions of one key, because a flush deduplicates."""
    found = []
    for table in store.tables:
        keys = [record.key for record in table.records]
        if len(keys) != len(set(keys)):
            seen = set()
            for key in keys:
                if key in seen:
                    found.append(
                        Violation(
                            check="unique_keys_per_table",
                            detail=f"table {table.number} holds {key!r} twice",
                            key=key,
                        )
                    )
                seen.add(key)
    return found


def ranges_match_contents(store: Store) -> list[Violation]:
    """Every file's advertised range is exactly its first and last key."""
    found = []
    for table in store.tables:
        if table.first != table.records[0].key or table.last != table.records[-1].key:
            found.append(
                Violation(
                    check="ranges_match_contents",
                    detail=f"table {table.number} advertises a range it does not hold",
                )
            )
    return found


def filters_hold_every_key(store: Store) -> list[Violation]:
    """No filter says no to a key its file holds, which would be a lost read."""
    found = []
    for table in store.tables:
        for record in table.records:
            if not table.filter.might_contain(record.key):
                found.append(
                    Violation(
                        check="filters_hold_every_key",
                        detail=f"table {table.number} filter denies {record.key!r}",
                        key=record.key,
                    )
                )
    return found


def sequences_are_unique(store: Store) -> list[Violation]:
    """No sequence number appears twice across the whole store."""
    seen: dict[int, str] = {}
    found = []
    for record in store.memtable.records():
        seen[record.sequence] = "memtable"
    for table in store.tables:
        for record in table.records:
            place = f"table {table.number}"
            if record.sequence in seen:
                held = seen[record.sequence]
                found.append(
                    Violation(
                        check="sequences_are_unique",
                        detail=f"sequence {record.sequence} in {held} and {place}",
                        key=record.key,
                    )
                )
            seen[record.sequence] = place
    return found


def sequences_do_not_exceed_the_counter(store: Store) -> list[Violation]:
    """No record claims a sequence the store has not issued yet."""
    found = []
    for table in store.tables:
        for record in table.records:
            if record.sequence > store.sequence:
                found.append(
                    Violation(
                        check="sequences_do_not_exceed_the_counter",
                        detail=f"sequence {record.sequence} above counter {store.sequence}",
                        key=record.key,
                    )
                )
    return found


def manifest_matches_tables(store: Store) -> list[Violation]:
    """The manifest's live set and the engine's table list agree exactly."""
    live = set(store.manifest.version.files)
    held = {table.number for table in store.tables}
    found = []
    for number in live - held:
        found.append(
            Violation(
                check="manifest_matches_tables",
                detail=f"manifest lists file {number} the engine does not hold",
            )
        )
    for number in held - live:
        found.append(
            Violation(
                check="manifest_matches_tables",
                detail=f"engine holds file {number} the manifest does not list",
            )
        )
    return found


def newest_version_wins_everywhere(store: Store) -> list[Violation]:
    """For every key, the engine's answer equals the newest version anywhere in it."""
    newest: dict[bytes, tuple[int, int, bytes]] = {}
    for record in store.memtable.records():
        held = newest.get(record.key)
        if held is None or record.sequence > held[0]:
            newest[record.key] = (record.sequence, record.kind, record.value)
    for table in store.tables:
        for record in table.records:
            held = newest.get(record.key)
            if held is None or record.sequence > held[0]:
                newest[record.key] = (record.sequence, record.kind, record.value)
    found = []
    for key, (_, kind, value) in newest.items():
        wanted = None if kind == DELETE else value
        got = store.get(key)
        if got != wanted:
            found.append(
                Violation(
                    check="newest_version_wins_everywhere",
                    detail=f"engine answers {got!r} and the newest version says {wanted!r}",
                    key=key,
                )
            )
    return found


CHECKS = (
    sorted_tables,
    unique_keys_per_table,
    ranges_match_contents,
    filters_hold_every_key,
    sequences_are_unique,
    sequences_do_not_exceed_the_counter,
    manifest_matches_tables,
    newest_version_wins_everywhere,
)


def check(store: Store) -> list[Violation]:
    """Every invariant, one pass, all violations."""
    found = []
    for one in CHECKS:
        found.extend(one(store))
    return found


def report(store: Store) -> dict:
    """A summary a test can assert on and a person can read."""
    found = check(store)
    return {
        "checks": len(CHECKS),
        "violations": len(found),
        "clean": not found,
        "details": [one.as_dict() for one in found],
    }
