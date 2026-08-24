from __future__ import annotations

import pytest

from store.multiget import (
    BLOCK,
    KEYS_PER_TABLE,
    TABLES,
    Table,
    Tier,
    _wanted,
    a_narrow_batch_shares_blocks,
    a_scattered_batch_shares_almost_nothing,
    absent_keys_cost_nothing_here_and_everything_in_life,
    duplicates_pay_again_alone,
    summarise,
    the_batch_is_one_trip,
)


class TestTable:
    def test_block_of_finds_a_present_key(self):
        table = Table(keys=list(range(0, 100, 2)))
        assert table.block_of(40) == 20 // BLOCK

    def test_block_of_rejects_an_absent_key(self):
        table = Table(keys=list(range(0, 100, 2)))
        assert table.block_of(41) == -1

    def test_read_counts_a_fetch(self):
        table = Table(keys=list(range(64)))
        assert table.read(3, remember=False)
        assert table.fetches == 1

    def test_remembered_block_absorbs_the_second_read(self):
        table = Table(keys=list(range(64)))
        table.read(3, remember=True)
        table.read(4, remember=True)
        assert table.fetches == 1

    def test_forget_drops_the_remembered_block(self):
        table = Table(keys=list(range(64)))
        table.read(3, remember=True)
        table.forget()
        table.read(4, remember=True)
        assert table.fetches == 2

    def test_a_miss_fetches_nothing(self):
        table = Table(keys=list(range(0, 64, 2)))
        assert not table.read(33, remember=False)
        assert table.fetches == 0


class TestTier:
    def test_build_is_deterministic(self):
        assert Tier.build(11).tables[0].keys == Tier.build(11).tables[0].keys

    def test_tables_partition_their_keys(self):
        tier = Tier.build(11)
        seen = set()
        for table in tier.tables:
            keys = set(table.keys)
            assert not (keys & seen)
            seen |= keys
        assert len(seen) == TABLES * KEYS_PER_TABLE

    def test_singles_and_batched_agree_on_found(self):
        for seed in range(5):
            wanted = _wanted(seed, 64, 900)
            assert Tier.build(11).singles(wanted)["found"] == (
                Tier.build(11).batched(wanted)["found"]
            )

    def test_batched_never_fetches_more(self):
        for seed in range(5):
            wanted = _wanted(seed, 64, 900)
            alone = Tier.build(11).singles(wanted)["fetches"]
            together = Tier.build(11).batched(wanted)["fetches"]
            assert together <= alone

    def test_the_batch_reports_one_trip(self):
        tier = Tier.build(11)
        assert tier.batched([1, 2, 3])["trips"] == 1


class TestClaims:
    @pytest.mark.parametrize(
        "claim",
        [
            the_batch_is_one_trip,
            a_scattered_batch_shares_almost_nothing,
            a_narrow_batch_shares_blocks,
            duplicates_pay_again_alone,
            absent_keys_cost_nothing_here_and_everything_in_life,
        ],
    )
    def test_claim_holds(self, claim):
        assert claim() is True

    def test_summary_is_all_true(self):
        told = summarise()
        assert all(value for name, value in told.items() if name != "module")
