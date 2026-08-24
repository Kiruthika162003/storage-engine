from __future__ import annotations

import pytest

from store import manifest as mod
from store.disk import Disk
from store.errors import BadFormat, Conflict, MissingFile
from store.manifest import (
    ADD,
    EDIT,
    REMOVE,
    SEQUENCE,
    Edit,
    File,
    Manifest,
    Recovered,
    Version,
    add,
    compaction,
    decode_change,
    from_directory,
    remove,
    replay,
    sequence,
)
from store.wal import frame


def store(changes) -> Edit:
    return Edit(changes=tuple(changes))


class TestChange:
    def test_an_add_round_trips(self):
        made = add(7, 2, 500)
        assert decode_change(made.encode()) == made

    def test_a_remove_round_trips(self):
        made = remove(7)
        assert decode_change(made.encode()) == made

    def test_a_sequence_round_trips(self):
        made = sequence(900)
        assert decode_change(made.encode()) == made

    def test_a_change_is_a_fixed_size(self):
        assert len(add(1, 0, 0).encode()) == EDIT.size

    def test_a_short_change_is_refused(self):
        with pytest.raises(BadFormat):
            decode_change(b"\x00")

    def test_a_long_change_is_refused(self):
        with pytest.raises(BadFormat):
            decode_change(add(1, 0, 0).encode() + b"\x00")

    def test_an_unknown_kind_is_refused(self):
        broken = bytearray(add(1, 0, 0).encode())
        broken[0] = 99
        with pytest.raises(BadFormat):
            decode_change(bytes(broken))

    def test_an_add_names_itself(self):
        assert add(1, 0, 0).as_dict()["kind"] == "add"

    def test_a_remove_names_itself(self):
        assert remove(1).as_dict()["kind"] == "remove"

    def test_a_sequence_names_itself(self):
        assert sequence(1).as_dict()["kind"] == "sequence"

    def test_the_kinds_are_distinct(self):
        assert len({ADD, REMOVE, SEQUENCE}) == 3

    def test_a_large_number_survives(self):
        assert decode_change(add(2**40, 0, 0).encode()).number == 2**40

    def test_a_large_record_count_survives(self):
        assert decode_change(add(1, 0, 2**40).encode()).records == 2**40


class TestEdit:
    def test_an_edit_frames_its_changes(self):
        made = store([add(1, 0, 10), remove(2)])
        assert len(made.encode()) == 8 + 2 * EDIT.size

    def test_the_adds_are_separated(self):
        made = store([add(1, 0, 10), remove(2), add(3, 1, 20)])
        assert len(made.adds) == 2

    def test_the_removes_are_separated(self):
        made = store([add(1, 0, 10), remove(2), remove(3)])
        assert len(made.removes) == 2

    def test_an_empty_edit_frames_to_nothing(self):
        assert len(store([]).encode()) == 8

    def test_as_dict_counts_the_changes(self):
        assert store([add(1, 0, 1), remove(2)]).as_dict()["changes"] == 2

    def test_a_compaction_adds_and_removes(self):
        made = compaction([(5, 1, 100)], [1, 2, 3])
        assert len(made.adds) == 1 and len(made.removes) == 3

    def test_a_compaction_with_no_output_only_removes(self):
        made = compaction([], [1, 2])
        assert not made.adds and len(made.removes) == 2


class TestVersion:
    def test_a_fresh_version_is_empty(self):
        assert Version().files == {}

    def test_an_add_installs_a_file(self):
        made = Version().apply(store([add(1, 0, 100)]))
        assert 1 in made.files

    def test_a_remove_retires_a_file(self):
        made = Version().apply(store([add(1, 0, 100)])).apply(store([remove(1)]))
        assert made.files == {}

    def test_adding_a_live_file_is_refused(self):
        made = Version().apply(store([add(1, 0, 100)]))
        with pytest.raises(Conflict):
            made.apply(store([add(1, 0, 100)]))

    def test_removing_a_missing_file_is_refused(self):
        with pytest.raises(MissingFile):
            Version().apply(store([remove(1)]))

    def test_a_sequence_only_advances(self):
        made = Version().apply(store([sequence(50)])).apply(store([sequence(20)]))
        assert made.sequence == 50

    def test_the_record_count_adds_up(self):
        made = Version().apply(store([add(1, 0, 100), add(2, 1, 200)]))
        assert made.records == 300

    def test_the_levels_group_the_files(self):
        made = Version().apply(store([add(1, 0, 1), add(2, 0, 1), add(3, 1, 1)]))
        assert len(made.levels()[0]) == 2 and len(made.levels()[1]) == 1

    def test_the_levels_are_in_number_order(self):
        made = Version().apply(store([add(3, 0, 1), add(1, 0, 1), add(2, 0, 1)]))
        assert [one.number for one in made.levels()[0]] == [1, 2, 3]

    def test_apply_leaves_the_original_alone(self):
        first = Version().apply(store([add(1, 0, 1)]))
        first.apply(store([add(2, 0, 1)]))
        assert len(first.files) == 1

    def test_a_removed_file_leaves_the_original_alone(self):
        first = Version().apply(store([add(1, 0, 1)]))
        first.apply(store([remove(1)]))
        assert len(first.files) == 1

    def test_a_refused_edit_leaves_the_version_alone(self):
        first = Version().apply(store([add(1, 0, 1)]))
        with pytest.raises(MissingFile):
            first.apply(store([add(2, 0, 1), remove(9)]))
        assert len(first.files) == 1

    def test_as_dict_counts_the_files(self):
        made = Version().apply(store([add(1, 0, 1), add(2, 0, 1)]))
        assert made.as_dict()["files"] == 2

    def test_as_dict_carries_the_sequence(self):
        assert Version().apply(store([sequence(7)])).as_dict()["sequence"] == 7


class TestManifest:
    def test_a_fresh_manifest_has_nothing(self):
        assert Manifest().edits == 0

    def test_an_install_advances_the_version(self):
        made = Manifest()
        made.install(store([add(1, 0, 100)]))
        assert made.version.records == 100

    def test_an_install_writes_bytes(self):
        made = Manifest()
        made.install(store([add(1, 0, 100)]))
        assert made.bytes_written() == 8 + EDIT.size

    def test_an_install_syncs_by_default(self):
        made = Manifest()
        made.install(store([add(1, 0, 100)]))
        assert made.disk.at_risk == 0

    def test_an_install_without_a_sync_leaves_bytes_at_risk(self):
        made = Manifest()
        made.install(store([add(1, 0, 100)]), sync=False)
        assert made.disk.at_risk > 0

    def test_a_refused_install_writes_nothing(self):
        made = Manifest()
        with pytest.raises(MissingFile):
            made.install(store([remove(1)]))
        assert made.bytes_written() == 0

    def test_a_refused_install_does_not_count(self):
        made = Manifest()
        with pytest.raises(MissingFile):
            made.install(store([remove(1)]))
        assert made.edits == 0

    def test_the_edit_count_follows_the_installs(self):
        made = Manifest()
        for at in range(5):
            made.install(store([add(at + 1, 0, 10)]))
        assert made.edits == 5

    def test_a_named_disk_is_accepted(self):
        made = Manifest(disk=Disk(name="OTHER"))
        made.install(store([add(1, 0, 1)]))
        assert made.disk.name == "OTHER"

    def test_as_dict_carries_the_edit_count(self):
        made = Manifest()
        made.install(store([add(1, 0, 1)]))
        assert made.as_dict()["edits"] == 1

    def test_as_dict_carries_the_version(self):
        made = Manifest()
        made.install(store([add(1, 0, 42)]))
        assert made.as_dict()["records"] == 42


class TestReplay:
    def test_an_empty_log_replays_to_nothing(self):
        assert replay(b"").version.files == {}

    def test_an_empty_log_reaches_the_end(self):
        assert replay(b"")

    def test_one_edit_replays(self):
        made = Manifest()
        made.install(store([add(1, 0, 100)]))
        assert replay(made.disk.read()).version.records == 100

    def test_many_edits_replay(self):
        made = mod._built(20)
        assert replay(made.disk.read()).version.files == made.version.files

    def test_the_sequence_replays(self):
        made = Manifest()
        made.install(store([sequence(7000)]))
        assert replay(made.disk.read()).version.sequence == 7000

    def test_a_compaction_replays(self):
        made = Manifest()
        for at in range(1, 5):
            made.install(store([add(at, 0, 1000)]))
        made.install(compaction([(5, 1, 3600)], [1, 2, 3, 4]))
        assert list(replay(made.disk.read()).version.files) == [5]

    def test_a_truncated_log_stops(self):
        made = mod._built(8)
        assert not replay(made.disk.read()[:-4])

    def test_a_truncated_log_reports_the_tail(self):
        made = mod._built(8)
        assert replay(made.disk.read()[:-4]).tail > 0

    def test_a_corrupt_checksum_stops(self):
        made = mod._built(4)
        raw = bytearray(made.disk.read())
        raw[20] ^= 0xFF
        assert replay(bytes(raw)).stopped == "badchecksum"

    def test_a_corrupt_checksum_keeps_the_prefix(self):
        made = mod._built(4)
        clean = replay(made.disk.read()).edits
        raw = bytearray(made.disk.read())
        raw[-20] ^= 0xFF
        assert 0 < replay(bytes(raw)).edits < clean

    def test_a_payload_that_is_not_a_multiple_of_a_change_stops(self):
        assert replay(frame(b"\x01\x02\x03")).stopped == "badformat"

    def test_every_truncation_gives_a_whole_prefix(self):
        made = mod._built(6)
        raw = made.disk.read()
        found = {replay(raw[:cut]).edits for cut in range(len(raw) + 1)}
        assert found == set(range(max(found) + 1))

    def test_every_truncation_replays_without_raising(self):
        made = mod._built(6)
        raw = made.disk.read()
        assert all(replay(raw[:cut]) is not None for cut in range(len(raw) + 1))

    def test_a_recovered_reports_where_it_stopped(self):
        assert replay(b"\x00" * 3).stopped == "tornwrite"

    def test_a_recovered_is_false_when_it_stopped_early(self):
        assert not replay(b"\x00" * 3)

    def test_as_dict_carries_the_stop_reason(self):
        assert replay(b"\x00" * 3).as_dict()["stopped"] == "tornwrite"

    def test_a_recovered_can_be_built_directly(self):
        made = Recovered(version=Version(), edits=0, stopped="end", tail=0)
        assert made


class TestFromDirectory:
    def test_a_listing_counts_every_file(self):
        made = from_directory([File(number=at, level=0, records=10) for at in range(1, 5)])
        assert made.records == 40

    def test_a_listing_counts_the_output_and_the_inputs(self):
        files = [File(number=at, level=0, records=1000) for at in range(1, 5)]
        files.append(File(number=5, level=1, records=3600))
        assert from_directory(files).records == 7600

    def test_a_listing_of_nothing_is_empty(self):
        assert from_directory([]).files == {}

    def test_a_listing_groups_by_level(self):
        files = [File(number=1, level=0, records=1), File(number=2, level=1, records=1)]
        assert set(from_directory(files).levels()) == {0, 1}


class TestMeasurements:
    def test_a_listing_cannot_tell_which_files_count(self):
        assert mod.a_directory_listing_cannot_tell_which_files_count()

    def test_an_edit_is_atomic(self):
        assert mod.an_edit_is_all_or_nothing_because_a_partial_frame_fails_its_checksum()

    def test_a_replay_gives_the_version_that_wrote_it(self):
        assert mod.a_replay_of_a_whole_manifest_gives_the_version_that_wrote_it()

    def test_the_manifest_is_tiny(self):
        assert mod.the_manifest_is_tiny_next_to_what_it_describes()

    def test_an_unsynced_edit_reverts(self):
        assert mod.an_unsynced_edit_is_lost_and_the_store_reverts()

    def test_an_impossible_edit_is_refused(self):
        assert mod.an_edit_that_removes_a_file_that_is_not_live_is_refused()

    def test_a_version_is_immutable(self):
        assert mod.a_version_is_not_changed_by_applying_an_edit_to_it()

    def test_damage_stops_the_replay(self):
        assert mod.a_torn_frame_stops_the_replay_rather_than_being_skipped()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_eight_claims(self):
        assert len(mod.summarise()) == 8

    def test_the_truncation_table_has_seven_rows(self):
        assert len(mod.compare_the_truncations(8)) == 7

    def test_the_truncation_table_starts_empty(self):
        assert mod.compare_the_truncations(8)[0]["edits"] == 0

    def test_the_truncation_table_ends_complete(self):
        assert mod.compare_the_truncations(8)[-1]["stopped"] == "end"

    def test_the_truncation_table_never_goes_backwards(self):
        found = [row["edits"] for row in mod.compare_the_truncations(8)]
        assert found == sorted(found)

    def test_the_built_manifest_is_shared(self):
        assert mod._built(4) is mod._built(4)

    def test_the_built_manifest_installed_edits(self):
        assert mod._built(4).edits > 4

    def test_the_built_manifest_ends_with_a_live_version(self):
        assert mod._built(4).version.files
