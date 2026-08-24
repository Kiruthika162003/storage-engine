from __future__ import annotations

import pytest

from store import groupagg as mod
from store.errors import ConfigError
from store.groupagg import Meter, hash_aggregate, stream_aggregate


class TestHashAggregate:
    def test_one_group_sums(self):
        assert hash_aggregate([(1, 10), (1, 5)], Meter()) == [(1, 15)]

    def test_groups_come_back_sorted(self):
        assert hash_aggregate([(2, 1), (1, 1)], Meter()) == [(1, 1), (2, 1)]

    def test_empty_input_aggregates_to_nothing(self):
        assert hash_aggregate([], Meter()) == []

    def test_the_meter_counts_the_groups(self):
        meter = Meter()
        hash_aggregate([(1, 1), (2, 1), (2, 1)], meter)
        assert meter.held_groups == 2

    def test_the_hash_emits_only_at_the_end(self):
        meter = Meter()
        hash_aggregate([(1, 1), (2, 1)], meter)
        assert meter.first_emit_after == 2


class TestStreamAggregate:
    def test_one_group_sums(self):
        assert stream_aggregate([(1, 10), (1, 5)], Meter()) == [(1, 15)]

    def test_group_changes_close_groups(self):
        assert stream_aggregate([(1, 1), (2, 2), (3, 3)], Meter()) == [
            (1, 1),
            (2, 2),
            (3, 3),
        ]

    def test_the_last_group_is_emitted(self):
        made = stream_aggregate([(1, 1), (2, 2)], Meter())
        assert made[-1] == (2, 2)

    def test_empty_input_aggregates_to_nothing(self):
        assert stream_aggregate([], Meter()) == []

    def test_unsorted_input_is_refused(self):
        with pytest.raises(ConfigError):
            stream_aggregate([(2, 1), (1, 1)], Meter())

    def test_one_accumulator_is_held(self):
        meter = Meter()
        stream_aggregate([(1, 1), (2, 1), (3, 1)], meter)
        assert meter.held_groups == 1

    def test_the_first_emit_is_at_the_first_change(self):
        meter = Meter()
        stream_aggregate([(1, 1), (1, 1), (2, 1)], meter)
        assert meter.first_emit_after == 2


class TestAgreement:
    def test_the_generated_orders_agree(self):
        rows = list(mod._orders(3000, 200))
        assert hash_aggregate(rows, Meter()) == stream_aggregate(rows, Meter())


class TestMeasurements:
    def test_the_aggregates_agree(self):
        assert mod.both_aggregates_agree()

    def test_one_group_against_all(self):
        assert mod.the_stream_holds_one_group_and_the_hash_holds_all()

    def test_pipelines_against_barriers(self):
        assert mod.the_stream_emits_early_and_the_hash_emits_at_the_end()

    def test_unsorted_is_refused(self):
        assert mod.unsorted_input_is_refused_not_misgrouped()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4

    def test_the_orders_are_cached(self):
        assert mod._orders(100, 10) is mod._orders(100, 10)
