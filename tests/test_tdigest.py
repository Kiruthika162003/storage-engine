from __future__ import annotations

import pytest

from store.tdigest import (
    COMPRESSION,
    Digest,
    _latencies,
    absorb,
    four_hundred_centroids_hold_a_hundred_thousand,
    rank_error,
    summarise,
    the_average_of_two_p99s_is_nobodys_p99,
    the_middle_error_sits_inside_the_design_bound,
    the_tail_is_sharper_than_the_middle,
)


class TestDigest:
    def test_an_empty_digest_answers_zero(self):
        assert Digest().quantile(0.5) == 0.0

    def test_a_single_value_is_every_quantile(self):
        digest = Digest()
        digest.add(42.0)
        assert digest.quantile(0.1) == 42.0
        assert digest.quantile(0.9) == 42.0

    def test_total_counts_every_add(self):
        digest = Digest()
        for value in range(500):
            digest.add(float(value))
        assert digest.total == 500

    def test_the_digest_stays_small(self):
        digest = Digest()
        for value in _latencies(3):
            digest.add(value)
        assert digest.size() < COMPRESSION * 8

    def test_quantiles_are_monotone(self):
        digest = Digest()
        for value in _latencies(3):
            digest.add(value)
        answers = [digest.quantile(q / 20) for q in range(1, 20)]
        assert answers == sorted(answers)

    def test_the_median_of_a_uniform_ramp(self):
        digest = Digest()
        for value in range(10000):
            digest.add(float(value))
        assert abs(digest.quantile(0.5) - 5000) < 200

    def test_centroid_weights_cover_the_total(self):
        digest = Digest()
        for value in _latencies(3):
            digest.add(value)
        digest._merge()
        assert sum(c.weight for c in digest.centroids) == digest.total


class TestAbsorb:
    def test_absorb_preserves_the_total(self):
        one, two = Digest(), Digest()
        for value in range(100):
            one.add(float(value))
            two.add(float(value + 100))
        assert absorb(one, two).total == 200

    def test_absorb_spans_both_ranges(self):
        one, two = Digest(), Digest()
        for value in range(100):
            one.add(float(value))
            two.add(float(value + 1000))
        joint = absorb(one, two)
        assert joint.quantile(0.25) < 100
        assert joint.quantile(0.75) > 1000

    def test_absorb_does_not_disturb_its_inputs(self):
        one, two = Digest(), Digest()
        for value in range(100):
            one.add(float(value))
            two.add(float(value))
        before = one.quantile(0.5)
        absorb(one, two)
        assert one.quantile(0.5) == before


class TestRankError:
    def test_a_perfect_answer_has_no_error(self):
        ordered = [float(v) for v in range(1000)]
        assert rank_error(ordered, 0.5, 500.0) == 0.0

    def test_a_wrong_answer_measures_its_distance(self):
        ordered = [float(v) for v in range(1000)]
        assert rank_error(ordered, 0.5, 600.0) == 0.1


class TestClaims:
    @pytest.mark.parametrize(
        "claim",
        [
            four_hundred_centroids_hold_a_hundred_thousand,
            the_tail_is_sharper_than_the_middle,
            the_middle_error_sits_inside_the_design_bound,
            the_average_of_two_p99s_is_nobodys_p99,
        ],
    )
    def test_claim_holds(self, claim):
        assert claim() is True

    def test_summary_is_all_true(self):
        told = summarise()
        assert all(value for name, value in told.items() if name != "module")
