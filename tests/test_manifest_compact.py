from __future__ import annotations

from store import manifest_compact as mod
from store.manifest import Edit, Manifest, add, replay, sequence
from store.manifest_compact import rewrite, shrink, snapshot_edit


def grown(files: int = 10) -> Manifest:
    made = Manifest()
    for at in range(1, files + 1):
        made.install(Edit(changes=(add(at, at % 3, at * 10), sequence(at * 100))))
    return made


class TestSnapshotEdit:
    def test_the_edit_holds_every_file(self):
        manifest = grown(7)
        made = snapshot_edit(manifest.version)
        assert len(made.adds) == 7

    def test_the_edit_carries_the_sequence(self):
        manifest = grown(3)
        fresh = Manifest()
        fresh.install(snapshot_edit(manifest.version))
        assert fresh.version.sequence == manifest.version.sequence

    def test_an_empty_version_snapshots_to_a_sequence_only(self):
        made = snapshot_edit(Manifest().version)
        assert not made.adds and len(made.changes) == 1


class TestRewrite:
    def test_the_rewrite_is_one_edit(self):
        assert rewrite(grown(20)).edits == 1

    def test_the_rewrite_keeps_the_files(self):
        manifest = grown(20)
        assert rewrite(manifest).version.files == manifest.version.files

    def test_the_rewrite_keeps_the_levels(self):
        manifest = grown(9)
        fresh = rewrite(manifest)
        for number, file in manifest.version.files.items():
            assert fresh.version.files[number].level == file.level

    def test_the_rewrite_uses_a_fresh_disk(self):
        manifest = grown(5)
        assert rewrite(manifest).disk is not manifest.disk

    def test_the_rewritten_log_replays(self):
        manifest = grown(12)
        found = replay(rewrite(manifest).disk.read())
        assert found and found.version.files == manifest.version.files


class TestShrink:
    def test_shrink_reports_the_before_and_after(self):
        manifest = mod._long_lived(50)
        _, saved = shrink(manifest)
        assert saved.bytes_before > saved.bytes_after

    def test_the_ratio_is_after_over_before(self):
        manifest = mod._long_lived(50)
        _, saved = shrink(manifest)
        assert saved.ratio == round(saved.bytes_after / saved.bytes_before, 4)

    def test_a_short_history_barely_shrinks(self):
        manifest = grown(2)
        _, saved = shrink(manifest)
        assert saved.ratio > 0.3


class TestMeasurements:
    def test_the_rewrite_shrinks_a_hundredfold(self):
        assert mod.the_rewrite_shrinks_a_long_history_a_hundredfold()

    def test_the_rewrite_replays_the_same(self):
        assert mod.the_rewritten_log_replays_to_the_same_version()

    def test_the_swap_crash_is_indifferent(self):
        assert mod.a_crash_between_the_logs_is_indifferent()

    def test_thresholded_rewrites_stay_flat(self):
        assert mod.the_rewrite_keeps_growing_history_flat()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4

    def test_the_long_lived_manifest_is_cached(self):
        assert mod._long_lived(50) is mod._long_lived(50)
