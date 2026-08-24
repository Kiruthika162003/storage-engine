from __future__ import annotations

import pytest

from store import cuckoo as mod
from store.cuckoo import BUCKET_SLOTS, Cuckoo, _alternate, _fingerprint, _index
from store.errors import ConfigError, TooLarge


def filled(buckets: int, count: int) -> tuple[Cuckoo, list[bytes]]:
    made = Cuckoo(buckets=buckets)
    keys = mod._keys(count)
    for key in keys:
        made.add(key)
    return made, keys


class TestShape:
    def test_a_zero_bucket_count_is_refused(self):
        with pytest.raises(ConfigError):
            Cuckoo(buckets=0)

    def test_a_non_power_of_two_is_refused(self):
        with pytest.raises(ConfigError):
            Cuckoo(buckets=4000)

    def test_the_slot_count_is_buckets_times_four(self):
        assert Cuckoo(buckets=64).slots == 64 * BUCKET_SLOTS

    def test_a_fresh_filter_is_empty(self):
        assert Cuckoo(buckets=64).occupancy == 0.0

    def test_the_bytes_follow_the_fingerprint_width(self):
        made = Cuckoo(buckets=64)
        assert made.nbytes == (made.slots * mod.FINGERPRINT_BITS + 7) // 8


class TestMembership:
    def test_an_added_key_answers_yes(self):
        made = Cuckoo(buckets=64)
        made.add(b"hello")
        assert made.might_contain(b"hello")

    def test_an_absent_key_usually_answers_no(self):
        made, _ = filled(1024, 2000)
        absent = mod._keys(2000, "absent")
        lies = sum(1 for key in absent if made.might_contain(key))
        assert lies < 100

    def test_every_inserted_key_answers_yes(self):
        made, keys = filled(1024, 3000)
        assert all(made.might_contain(key) for key in keys)

    def test_membership_survives_evictions(self):
        made, keys = filled(1024, 3900)
        assert made.kicks > 0 and all(made.might_contain(key) for key in keys)


class TestRemoval:
    def test_a_removed_key_answers_no(self):
        made = Cuckoo(buckets=64)
        made.add(b"hello")
        assert made.remove(b"hello") and not made.might_contain(b"hello")

    def test_removing_an_absent_key_reports_it(self):
        assert not Cuckoo(buckets=64).remove(b"never")

    def test_removal_leaves_the_others(self):
        made, keys = filled(1024, 2000)
        for key in keys[:1000]:
            made.remove(key)
        assert all(made.might_contain(key) for key in keys[1000:])

    def test_the_key_count_tracks_removals(self):
        made, keys = filled(1024, 2000)
        for key in keys[:500]:
            made.remove(key)
        assert made.keys == 1500

    def test_a_reinsert_after_removal_works(self):
        made = Cuckoo(buckets=64)
        made.add(b"k")
        made.remove(b"k")
        made.add(b"k")
        assert made.might_contain(b"k")


class TestRefusal:
    def test_an_overfull_filter_refuses(self):
        made = Cuckoo(buckets=16)
        with pytest.raises(TooLarge):
            for key in mod._keys(made.slots + 50):
                made.add(key)

    def test_the_refusal_is_counted(self):
        made = Cuckoo(buckets=16)
        try:
            for key in mod._keys(made.slots + 50):
                made.add(key)
        except TooLarge:
            pass
        assert made.refused == 1

    def test_the_filter_accepts_most_of_its_capacity(self):
        made = Cuckoo(buckets=256)
        landed = 0
        try:
            for key in mod._keys(made.slots + 50):
                made.add(key)
                landed += 1
        except TooLarge:
            pass
        assert landed > made.slots * 0.9


class TestHashing:
    def test_fingerprints_are_never_zero(self):
        assert all(_fingerprint(f"k{at}".encode()) != 0 for at in range(2000))

    def test_the_index_is_in_range(self):
        assert all(0 <= _index(f"k{at}".encode(), 128) < 128 for at in range(500))

    def test_the_alternate_differs_from_the_index_usually(self):
        differing = sum(
            1
            for at in range(500)
            if _alternate(_index(f"k{at}".encode(), 128), _fingerprint(f"k{at}".encode()), 128)
            != _index(f"k{at}".encode(), 128)
        )
        assert differing > 450

    def test_the_involution_at_a_power_of_two(self):
        for at in range(500):
            tag = _fingerprint(f"k{at}".encode())
            start = _index(f"k{at}".encode(), 256)
            assert _alternate(_alternate(start, tag, 256), tag, 256) == start


class TestMeasurements:
    def test_no_false_negatives(self):
        assert mod.no_false_negatives_below_the_refusal_point()

    def test_deletion_works(self):
        assert mod.deletion_works_and_bloom_has_nothing_to_compare()

    def test_refusal_near_95(self):
        assert mod.inserts_start_failing_near_ninety_five_percent()

    def test_the_rate_is_the_width(self):
        assert mod.the_false_positive_rate_matches_the_fingerprint_width()

    def test_the_involution_needs_a_power_of_two(self):
        assert mod.the_involution_holds_only_at_powers_of_two()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5

    def test_the_bloom_table_has_two_rows(self):
        rows = mod.compare_with_bloom(4000, 10000)
        assert [row["filter"] for row in rows] == ["cuckoo", "bloom"]

    def test_only_the_cuckoo_row_deletes(self):
        rows = {row["filter"]: row for row in mod.compare_with_bloom(4000, 10000)}
        assert rows["cuckoo"]["deletes"] and not rows["bloom"]["deletes"]
