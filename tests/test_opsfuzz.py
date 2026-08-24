from __future__ import annotations

from store.verify import opsfuzz as mod


class TestFuzzers:
    def test_the_ranged_fuzzer_is_clean(self):
        assert mod.fuzz_ranged(2000, 0) == ""

    def test_the_shelf_fuzzer_is_clean(self):
        assert mod.fuzz_shelf(2000, 1) == ""

    def test_the_batch_fuzzer_is_clean(self):
        assert mod.fuzz_batches(200, 2) == ""

    def test_more_seeds_stay_clean(self):
        for seed in range(3, 6):
            assert mod.fuzz_ranged(1000, seed) == ""
            assert mod.fuzz_shelf(1000, seed) == ""

    def test_every_fuzzer_runs_clean(self):
        assert mod.every_fuzzer_runs_clean_across_seeds()

    def test_the_summary_holds(self):
        assert all(mod.summarise().values())
