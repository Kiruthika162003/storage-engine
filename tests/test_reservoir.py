from __future__ import annotations

import pytest

from store import reservoir as mod
from store.errors import ConfigError
from store.reservoir import Reservoir, every_mth, first_k


class TestReservoir:
    def test_a_zero_capacity_is_refused(self):
        with pytest.raises(ConfigError):
            Reservoir(capacity=0)

    def test_the_first_arrivals_fill_the_reservoir(self):
        made = Reservoir(capacity=3)
        for value in (1, 2, 3):
            made.offer(value)
        assert sorted(made.held) == [1, 2, 3]

    def test_the_size_never_exceeds_capacity(self):
        made = Reservoir(capacity=3)
        for value in range(100):
            made.offer(value)
        assert len(made.held) == 3

    def test_seen_counts_every_offer(self):
        made = Reservoir(capacity=3)
        for value in range(100):
            made.offer(value)
        assert made.seen == 100

    def test_late_arrivals_can_enter(self):
        made = Reservoir(capacity=10, seed=1)
        for value in range(1000):
            made.offer(value)
        assert any(value >= 500 for value in made.held)

    def test_early_arrivals_can_survive(self):
        made = Reservoir(capacity=10, seed=1)
        for value in range(1000):
            made.offer(value)
        assert any(value < 500 for value in made.held)

    def test_the_same_seed_samples_the_same(self):
        left, right = Reservoir(seed=9), Reservoir(seed=9)
        for value in range(5000):
            left.offer(value)
            right.offer(value)
        assert left.held == right.held

    def test_a_stream_shorter_than_capacity_is_kept_whole(self):
        made = Reservoir(capacity=100)
        for value in range(30):
            made.offer(value)
        assert sorted(made.held) == list(range(30))


class TestShortcuts:
    def test_first_k_takes_the_front(self):
        assert first_k(range(100), capacity=5) == [0, 1, 2, 3, 4]

    def test_first_k_of_a_short_stream_takes_everything(self):
        assert first_k(range(3), capacity=5) == [0, 1, 2]

    def test_every_mth_strides(self):
        assert every_mth(range(100), capacity=10) == list(range(0, 100, 10))

    def test_every_mth_respects_capacity(self):
        assert len(every_mth(range(1000), capacity=10)) == 10


class TestMeasurements:
    def test_the_reservoir_is_uniform(self):
        assert mod.the_reservoir_is_uniform_over_positions()

    def test_first_k_reports_the_past(self):
        assert mod.first_k_reports_the_past_on_a_drifting_stream()

    def test_every_mth_aliases(self):
        assert mod.every_mth_aliases_on_a_periodic_stream()

    def test_the_size_is_fixed(self):
        assert mod.the_reservoir_holds_its_size_forever()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4

    def test_the_streams_are_cached(self):
        assert mod._drifting_stream(100) is mod._drifting_stream(100)
