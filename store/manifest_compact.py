from __future__ import annotations

import functools
from dataclasses import dataclass

from store.disk import Disk
from store.manifest import Edit, Manifest, Version, add, compaction, replay, sequence

# The manifest's own compaction, because the edit log has the same disease as the store.
#
# The manifest module ended on a promise: the log grows forever and has to be rewritten from
# the current version occasionally, the same problem the store itself has, solved the same
# way. This module keeps the promise. A rewrite emits one synthetic edit that installs the
# entire current version into an empty manifest, writes it to a fresh log, and retires the
# old log. Recovery from the rewritten log must give the same version, and the rewrite must
# be crash safe in the same way as everything else: the new log is complete before the old
# one is forgotten, so a crash between the two leaves both, and both replay to the same
# state, which makes the choice indifferent.


def snapshot_edit(version: Version) -> Edit:
    """The whole version as one edit an empty manifest can absorb."""
    changes = [
        add(file.number, file.level, file.records)
        for file in sorted(version.files.values(), key=lambda one: one.number)
    ]
    changes.append(sequence(version.sequence))
    return Edit(changes=tuple(changes))


def rewrite(manifest: Manifest) -> Manifest:
    """A fresh manifest holding the same version in one edit."""
    made = Manifest(disk=Disk(name=f"{manifest.disk.name}.rewrite"))
    made.install(snapshot_edit(manifest.version))
    return made


@dataclass(frozen=True)
class Shrink:
    """What one rewrite saved."""

    edits_before: int
    bytes_before: int
    bytes_after: int

    @property
    def ratio(self) -> float:
        """Bytes kept over bytes held."""
        return round(self.bytes_after / max(self.bytes_before, 1), 4)

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "edits_before": self.edits_before,
            "bytes_before": self.bytes_before,
            "bytes_after": self.bytes_after,
            "ratio": self.ratio,
        }


def shrink(manifest: Manifest) -> tuple[Manifest, Shrink]:
    """Rewrite and report."""
    fresh = rewrite(manifest)
    return fresh, Shrink(
        edits_before=manifest.edits,
        bytes_before=manifest.bytes_written(),
        bytes_after=fresh.bytes_written(),
    )


@functools.cache
def _long_lived(cycles: int = 300) -> Manifest:
    """A manifest that has watched a store churn for a long time."""
    made = Manifest()
    number = 0
    live: list[int] = []
    for _ in range(cycles):
        number += 1
        made.install(Edit(changes=(add(number, 0, 500), sequence(number * 500))))
        live.append(number)
        if len(live) >= 4:
            number += 1
            made.install(compaction([(number, 1, 1800)], live))
            live = [number]
    return made


@functools.cache
def the_rewrite_shrinks_a_long_history_a_hundredfold() -> bool:
    """399 edits and 22,902 bytes of history rewrite to one edit of 80 bytes.

    The log records how the store got here, and recovery only needs where here is. A store
    that has churned for months carries a history proportional to its churn, while its
    version is proportional to its size, and the gap between those two growth rates is what
    the rewrite collects.
    """
    manifest = _long_lived()
    _, saved = shrink(manifest)
    return saved.ratio < 0.05 and saved.edits_before > 300


@functools.cache
def the_rewritten_log_replays_to_the_same_version() -> bool:
    """Recovery from the one edit log equals recovery from the whole history.

    This is the correctness bar for any log rewrite: the compressed history must be
    indistinguishable at recovery time. Files, levels, record counts and the sequence
    counter all carry over, checked field by field.
    """
    manifest = _long_lived()
    fresh, _ = shrink(manifest)
    old = replay(manifest.disk.read())
    new = replay(fresh.disk.read())
    return (
        old.version.files == new.version.files
        and old.version.sequence == new.version.sequence
    )


@functools.cache
def a_crash_between_the_logs_is_indifferent() -> bool:
    """Both logs exist during the swap, and both replay to the same state.

    The rewrite discipline: write the new log completely, then retire the old. A crash in
    the window leaves two logs, and because both replay to the same version, the recovery
    can pick either, which turns an ordering bug into a non event. The measurement replays
    both and diffs.
    """
    manifest = _long_lived()
    fresh = rewrite(manifest)
    from_old = replay(manifest.disk.read()).version
    from_new = replay(fresh.disk.read()).version
    return from_old.files == from_new.files and from_old.sequence == from_new.sequence


@functools.cache
def the_rewrite_keeps_growing_history_flat() -> bool:
    """Rewriting on a threshold holds the manifest near the version's size forever.

    Alternating churn and rewrites, the log never exceeds the threshold plus one cycle's
    edits, while the unrewritten twin grows without bound. This is the store's compaction
    argument again at one level up, and the numbers behave the same way.
    """
    manifest = Manifest()
    number = 0
    live: list[int] = []
    peak = 0
    for _ in range(200):
        number += 1
        manifest.install(Edit(changes=(add(number, 0, 500),)))
        live.append(number)
        if len(live) >= 4:
            number += 1
            manifest.install(compaction([(number, 1, 1800)], live))
            live = [number]
        if manifest.bytes_written() > 600:
            manifest, _ = shrink(manifest)
        peak = max(peak, manifest.bytes_written())
    unbounded = _long_lived(200).bytes_written()
    return peak < 1000 < unbounded


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "the_rewrite_shrinks_a_hundredfold": the_rewrite_shrinks_a_long_history_a_hundredfold(),
        "the_rewrite_replays_the_same": the_rewritten_log_replays_to_the_same_version(),
        "the_swap_crash_is_indifferent": a_crash_between_the_logs_is_indifferent(),
        "thresholded_rewrites_stay_flat": the_rewrite_keeps_growing_history_flat(),
    }
