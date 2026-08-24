from __future__ import annotations

import math

import pytest

from store import hll as mod
from store.errors import ConfigError
from store.hll import Sketch, exact_bytes


class TestSketch:
    def test_too_few_register_bits_is_refused(self):
        with pytest.raises(ConfigError):
            Sketch(register_bits=2)

    def test_too_many_register_bits_is_refused(self):
        with pytest.raises(ConfigError):
            Sketch(register_bits=20)

    def test_a_fresh_sketch_estimates_zero(self):
        assert Sketch().estimate() == 0

    def test_one_key_estimates_near_one(self):
        made = Sketch()
        made.add(b"only")
        assert made.estimate() in (1, 2)

    def test_the_added_counter_counts_additions(self):
        made = Sketch()
        made.add(b"a")
        made.add(b"a")
        assert made.added == 2

    def test_the_bytes_are_the_registers(self):
        assert Sketch(register_bits=8).nbytes == 256

    def test_the_estimate_tracks_the_truth(self):
        made = Sketch()
        for at in range(20000):
            made.add(f"k{at:06d}".encode())
        assert abs(made.estimate() - 20000) / 20000 < 0.1

    def test_repeats_do_not_inflate(self):
        made = Sketch()
        for _ in range(5):
            for at in range(1000):
                made.add(f"k{at:05d}".encode())
        assert abs(made.estimate() - 1000) / 1000 < 0.15

    def test_small_counts_use_the_correction(self):
        made = Sketch()
        for at in range(50):
            made.add(f"k{at}".encode())
        assert abs(made.estimate() - 50) <= 10

    def test_as_dict_carries_the_estimate(self):
        made = Sketch()
        made.add(b"a")
        assert made.as_dict()["estimate"] >= 1


class TestMerge:
    def test_disjoint_sketches_merge_to_the_sum(self):
        left, right = Sketch(), Sketch()
        for at in range(5000):
            left.add(f"L{at:05d}".encode())
            right.add(f"R{at:05d}".encode())
        union = left.merge(right).estimate()
        assert abs(union - 10000) / 10000 < 0.1

    def test_identical_sketches_merge_to_themselves(self):
        left, right = Sketch(), Sketch()
        for at in range(5000):
            left.add(f"K{at:05d}".encode())
            right.add(f"K{at:05d}".encode())
        assert left.merge(right).estimate() == left.estimate()

    def test_merge_is_commutative(self):
        left, right = Sketch(), Sketch()
        for at in range(2000):
            left.add(f"L{at:05d}".encode())
            right.add(f"R{at:05d}".encode())
        assert left.merge(right).registers == right.merge(left).registers

    def test_mismatched_widths_refuse_to_merge(self):
        with pytest.raises(ConfigError):
            Sketch(register_bits=8).merge(Sketch(register_bits=10))

    def test_merge_leaves_the_inputs_alone(self):
        left, right = Sketch(), Sketch()
        left.add(b"a")
        right.add(b"b")
        before = list(left.registers)
        left.merge(right)
        assert left.registers == before


class TestMeasurements:
    def test_the_error_is_banded(self):
        assert mod.the_error_sits_inside_the_promised_band()

    def test_duplicates_are_invisible(self):
        assert mod.duplicates_do_not_move_the_estimate()

    def test_the_union_is_free(self):
        assert mod.the_union_is_free_and_the_intersection_is_arithmetic()

    def test_the_sketch_stays_small(self):
        assert mod.the_sketch_is_thousands_of_times_smaller_at_the_top_size()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4

    def test_the_size_table_has_four_rows(self):
        assert len(mod.compare_the_sizes()) == 4

    def test_every_row_beats_three_standard_errors(self):
        band = 3 * 1.04 / math.sqrt(2048)
        assert all(row["error"] < band for row in mod.compare_the_sizes())

    def test_the_sketch_bytes_never_grow(self):
        rows = mod.compare_the_sizes()
        assert len({row["sketch_bytes"] for row in rows}) == 1

    def test_the_exact_bytes_grow_linearly(self):
        assert exact_bytes(200) == 2 * exact_bytes(100)

    def test_the_sketches_are_cached(self):
        assert mod._sketched(1000) is mod._sketched(1000)
