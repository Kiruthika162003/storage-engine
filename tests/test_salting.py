from __future__ import annotations

import random

import pytest

from store.salting import (
    SHARDS,
    Cluster,
    bump_salted,
    eight_salts_flatten_the_storm,
    one_hot_key_melts_one_shard,
    read_salted,
    salted,
    salts_beyond_shards_reconcentrate,
    summarise,
    the_read_pays_one_key_per_salt_and_stays_correct,
)


class TestCluster:
    def test_a_bump_lands_on_one_shard(self):
        cluster = Cluster()
        cluster.bump(b"k")
        assert sum(cluster.writes) == 1

    def test_placement_is_stable(self):
        cluster = Cluster()
        assert cluster._place(b"k") == cluster._place(b"k")

    def test_read_returns_the_count(self):
        cluster = Cluster()
        for _ in range(3):
            cluster.bump(b"k")
        assert cluster.read(b"k") == 3

    def test_an_unwritten_key_reads_zero(self):
        assert Cluster().read(b"ghost") == 0

    def test_hottest_share_of_a_single_key_is_total(self):
        cluster = Cluster()
        for _ in range(5):
            cluster.bump(b"k")
        assert cluster.hottest_write_share() == 1.0


class TestSalting:
    def test_salted_keys_differ_by_suffix(self):
        assert salted(b"k", 0) != salted(b"k", 1)
        assert salted(b"k", 3) == b"k#3"

    def test_salted_bumps_sum_to_the_truth(self):
        cluster = Cluster()
        source = random.Random(1)
        for _ in range(100):
            bump_salted(cluster, b"k", 4, source)
        assert read_salted(cluster, b"k", 4) == 100

    def test_salted_copies_spread_over_shards(self):
        cluster = Cluster()
        shards = {cluster._place(salted(b"k", salt)) for salt in range(SHARDS)}
        assert len(shards) > 1

    def test_reading_fewer_salts_undercounts(self):
        cluster = Cluster()
        source = random.Random(1)
        for _ in range(100):
            bump_salted(cluster, b"k", 4, source)
        assert read_salted(cluster, b"k", 2) < 100


class TestClaims:
    @pytest.mark.parametrize(
        "claim",
        [
            one_hot_key_melts_one_shard,
            eight_salts_flatten_the_storm,
            the_read_pays_one_key_per_salt_and_stays_correct,
            salts_beyond_shards_reconcentrate,
        ],
    )
    def test_claim_holds(self, claim):
        assert claim() is True

    def test_summary_is_all_true(self):
        told = summarise()
        assert all(value for name, value in told.items() if name != "module")
