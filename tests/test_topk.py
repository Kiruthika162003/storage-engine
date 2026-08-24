from __future__ import annotations

import pytest

from store import topk as mod
from store.errors import ConfigError
from store.topk import Summary


class TestSummary:
    def test_a_zero_capacity_is_refused(self):
        with pytest.raises(ConfigError):
            Summary(capacity=0)

    def test_a_noted_key_is_tracked(self):
        made = Summary(capacity=2)
        made.note(b"a")
        assert made.candidates() == {b"a": 1}

    def test_repeats_increment(self):
        made = Summary(capacity=2)
        made.note(b"a")
        made.note(b"a")
        assert made.candidates()[b"a"] == 2

    def test_free_slots_admit_new_keys(self):
        made = Summary(capacity=2)
        made.note(b"a")
        made.note(b"b")
        assert set(made.candidates()) == {b"a", b"b"}

    def test_a_full_table_charges_the_toll(self):
        made = Summary(capacity=2)
        made.note(b"a")
        made.note(b"b")
        made.note(b"c")
        assert made.decrements == 1

    def test_the_toll_can_evict(self):
        made = Summary(capacity=2)
        made.note(b"a")
        made.note(b"b")
        made.note(b"c")
        assert len(made.candidates()) <= 2

    def test_a_dominant_key_survives_any_interleaving(self):
        made = Summary(capacity=2)
        for at in range(100):
            made.note(b"hot")
            made.note(f"cold{at}".encode())
        assert b"hot" in made.candidates()

    def test_seen_counts_everything(self):
        made = Summary(capacity=2)
        for at in range(10):
            made.note(f"k{at}".encode())
        assert made.seen == 10

    def test_the_bound_is_seen_over_capacity_plus_one(self):
        made = Summary(capacity=4)
        for at in range(100):
            made.note(f"k{at % 10}".encode())
        assert made.bound == 20.0

    def test_certainly_above_uses_the_tracked_counts(self):
        made = Summary(capacity=4)
        for _ in range(50):
            made.note(b"hot")
        assert made.certainly_above(10) == {b"hot"}

    def test_certainly_above_a_huge_threshold_is_empty(self):
        made = Summary(capacity=4)
        made.note(b"a")
        assert made.certainly_above(10**6) == set()


class TestMeasurements:
    def test_heavy_keys_are_present(self):
        assert mod.every_truly_heavy_key_is_present()

    def test_counts_undercount_boundedly(self):
        assert mod.tracked_counts_undercount_within_the_bound()

    def test_uniform_streams_certify_nothing(self):
        assert mod.a_uniform_stream_yields_no_certainties()

    def test_the_toll_is_rare(self):
        assert mod.the_toll_is_rare_on_skewed_streams()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4

    def test_the_stream_is_cached(self):
        assert mod._skewed_stream(100) is mod._skewed_stream(100)
