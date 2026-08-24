from __future__ import annotations

from store import simulation as mod
from store.simulation import Diary, a_week


class TestDiary:
    def test_a_fresh_diary_is_clean(self):
        assert Diary().clean()

    def test_a_mismatch_dirties_it(self):
        assert not Diary(read_mismatches=1).clean()

    def test_a_break_dirties_it(self):
        assert not Diary(invariant_breaks=1).clean()

    def test_as_dict_carries_the_verdict(self):
        assert Diary().as_dict()["clean"] is True


class TestWeek:
    def test_the_week_is_clean(self):
        diary = a_week(0)
        assert diary.clean()

    def test_every_event_fires(self):
        diary = a_week(0)
        assert diary.crashes == 3 and diary.checkpoints == 1 and diary.restores_checked == 1

    def test_the_volumes_are_substantial(self):
        diary = a_week(0)
        assert diary.writes > 8000 and diary.reads > 4000

    def test_a_second_seed_is_also_clean(self):
        assert a_week(1).clean()


class TestMeasurements:
    def test_the_week_runs_clean_across_seeds(self):
        assert mod.the_week_runs_clean_across_seeds()

    def test_the_summary_holds(self):
        assert all(mod.summarise().values())
