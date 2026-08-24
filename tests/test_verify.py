from __future__ import annotations

import random

from store.engine import Store
from store.record import Record
from store.verify import crashfuzz, differential, invariants, model, torn


def worked(seed: int = 6, writes: int = 3000) -> Store:
    source = random.Random(seed)
    store = Store(flush_at=300, fold_at=3)
    for _ in range(writes):
        key = f"k{source.randrange(700):05d}".encode()
        if source.random() < 0.1:
            store.delete(key)
        else:
            store.put(key, source.randbytes(10))
    return store


class TestInvariants:
    def test_a_worked_store_is_clean(self):
        assert invariants.report(worked())["clean"]

    def test_the_report_counts_the_checks(self):
        assert invariants.report(worked())["checks"] == len(invariants.CHECKS)

    def test_a_fresh_store_is_clean(self):
        assert invariants.report(Store())["clean"]

    def test_an_unsorted_table_is_caught(self):
        store = worked()
        store.tables[0].records.reverse()
        assert any(one.check == "sorted_tables" for one in invariants.check(store))

    def test_a_duplicated_key_is_caught(self):
        store = worked()
        store.tables[0].records.append(store.tables[0].records[-1])
        found = {one.check for one in invariants.check(store)}
        assert "unique_keys_per_table" in found

    def test_a_foreign_sequence_is_caught(self):
        store = worked()
        store.tables[0].records[0] = Record(key=b"zzz", sequence=10**9, value=b"x")
        found = {one.check for one in invariants.check(store)}
        assert "sequences_do_not_exceed_the_counter" in found

    def test_a_manifest_mismatch_is_caught(self):
        store = worked()
        store.tables.pop()
        found = {one.check for one in invariants.check(store)}
        assert "manifest_matches_tables" in found

    def test_violations_name_their_key(self):
        store = worked()
        store.tables[0].records.append(store.tables[0].records[-1])
        broken = [one for one in invariants.check(store) if one.key]
        assert broken and broken[0].as_dict()["key"]


class TestModel:
    def test_a_long_program_is_clean(self):
        assert model.run(steps=2500, seed=0)

    def test_every_seed_is_clean(self):
        swept = model.sweep(runs=6, steps=600)
        assert swept["failed"] == 0

    def test_the_outcome_counts_the_steps(self):
        assert model.run(steps=400, seed=2).steps == 400

    def test_the_program_is_kept(self):
        assert len(model.run(steps=300, seed=3).program) == 300

    def test_a_clean_outcome_is_truthy(self):
        made = model.Outcome(steps=10)
        assert made and made.as_dict()["clean"]

    def test_a_disagreement_is_falsy(self):
        made = model.Outcome(steps=10, disagreement="get gave wrong", at_step=4)
        assert not made


class TestDifferential:
    def test_the_fleet_agrees_on_a_mixed_stream(self):
        assert differential.run(3000, 600, 0)["clean"]

    def test_the_fleet_agrees_under_heavy_deletes(self):
        assert differential.run(2000, 150, 4)["clean"]

    def test_the_fleet_agrees_on_a_tiny_key_space(self):
        assert differential.run(2000, 20, 8)["clean"]

    def test_touched_keys_are_counted(self):
        assert differential.run(500, 100, 1)["keys_touched"] <= 100

    def test_a_divergence_names_every_answer(self):
        made = differential.Divergence(key=b"k", answers={"store": b"1", "btree": None})
        assert set(made.as_dict()["answers"]) == {"store", "btree"}


class TestCrashFuzz:
    def test_one_crash_is_clean(self):
        assert crashfuzz.run(1000, 300, 0)

    def test_every_crash_point_is_clean(self):
        swept = crashfuzz.sweep(runs=12, writes=700)
        assert swept["failed"] == 0

    def test_the_double_crash_is_clean(self):
        assert crashfuzz.double_crash(600, 3)

    def test_the_run_counts_survivors(self):
        made = crashfuzz.run(800, 200, 5)
        assert made.survived == made.acknowledged

    def test_a_lost_write_would_be_reported(self):
        made = crashfuzz.Run(
            writes=1, crashed_at=0, acknowledged=1, survived=0, lost_acknowledged=[b"k"]
        )
        assert not made and made.as_dict()["lost"] == ["b'k'"]


class TestTorn:
    def test_a_truncation_stops_clean(self):
        assert torn.truncate(100).stopped_clean

    def test_a_bit_flip_stops_clean(self):
        assert torn.flip(500).stopped_clean

    def test_a_torn_sector_stops_clean(self):
        assert torn.tear(700).stopped_clean

    def test_a_misplaced_write_stops_clean(self):
        assert torn.misplace(300).stopped_clean

    def test_the_sweep_finds_no_changed_record(self):
        assert torn.sweep(points=15)["clean"]

    def test_the_manifest_sweep_finds_no_poisoned_version(self):
        assert torn.manifest_sweep(40)["clean"]

    def test_truncation_at_zero_recovers_nothing(self):
        assert torn.truncate(0).recovered == 0

    def test_truncation_at_the_end_recovers_everything(self):
        raw, _ = torn._log(50)
        assert torn.truncate(len(raw), 50).recovered == 50

    def test_damage_never_invents_records(self):
        outcome = torn.flip(40)
        assert outcome.recovered <= outcome.total
