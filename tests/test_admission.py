from __future__ import annotations

import pytest

from store import admission as mod
from store.admission import Admitting, Frequency, drive
from store.cache import Recent
from store.errors import ConfigError


class TestFrequency:
    def test_a_too_narrow_sketch_is_refused(self):
        with pytest.raises(ConfigError):
            Frequency(width=4)

    def test_a_fresh_sketch_estimates_zero(self):
        assert Frequency().estimate(7) == 0

    def test_a_touch_raises_the_estimate(self):
        made = Frequency()
        made.touch(7)
        assert made.estimate(7) >= 1

    def test_estimates_never_undershoot(self):
        made = Frequency(width=64)
        for _ in range(50):
            made.touch(3)
        assert made.estimate(3) >= 50

    def test_fading_halves(self):
        made = Frequency()
        for _ in range(10):
            made.touch(3)
        made.fade()
        assert made.estimate(3) == 5

    def test_the_window_fades_automatically(self):
        made = Frequency(width=64)
        for _ in range(mod.FADE_EVERY):
            made.touch(1)
        assert made.estimate(1) < mod.FADE_EVERY

    def test_the_bytes_follow_the_width(self):
        assert Frequency(width=256).nbytes == 4 * 256


class TestAdmitting:
    def test_an_unfull_cache_admits_everyone(self):
        made = Admitting(capacity=4)
        for at in range(4):
            made.put(at, b"v")
        assert made.admitted == 4 and made.turned_away == 0

    def test_a_cold_newcomer_is_turned_away(self):
        made = Admitting(capacity=2)
        for _ in range(5):
            made.get(0)
            made.get(1)
        made.put(0, b"v")
        made.put(1, b"v")
        made.put(99, b"v")
        assert made.turned_away == 1

    def test_a_hot_newcomer_gets_in(self):
        made = Admitting(capacity=2)
        made.put(0, b"v")
        made.put(1, b"v")
        for _ in range(10):
            made.get(99)
        made.put(99, b"v")
        assert made.cache.get(99) is not None

    def test_gets_feed_the_sketch(self):
        made = Admitting(capacity=2)
        made.get(7)
        assert made.sketch.estimate(7) >= 1

    def test_the_rate_is_the_caches(self):
        made = Admitting(capacity=2)
        made.put(1, b"v")
        made.get(1)
        assert made.rate == made.cache.stats.rate


class TestDrive:
    def test_drive_fills_on_misses(self):
        cache = Recent(capacity=8)
        drive(cache, [1, 2, 3])
        assert len(cache.held) == 3

    def test_drive_returns_the_rate(self):
        assert drive(Recent(capacity=8), [1, 1, 1]) > 0.5

    def test_drive_handles_the_gated_kind(self):
        assert drive(Admitting(capacity=8), [1, 1, 1]) > 0.5


class TestMeasurements:
    def test_the_gate_doubles_polluted_hits(self):
        assert mod.admission_doubles_the_hit_rate_on_a_polluted_stream()

    def test_the_gate_is_invisible_when_clean(self):
        assert mod.the_gate_is_invisible_on_a_clean_working_set()

    def test_estimates_bound_from_above(self):
        assert mod.the_sketch_never_underestimates()

    def test_fading_forgets(self):
        assert mod.fading_lets_yesterdays_hot_key_lose()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4

    def test_the_stream_table_has_two_rows(self):
        rows = mod.compare_the_streams()
        assert [row["stream"] for row in rows] == ["polluted", "clean"]

    def test_the_gate_wins_only_where_it_should(self):
        rows = {row["stream"]: row for row in mod.compare_the_streams()}
        assert rows["polluted"]["gated"] > rows["polluted"]["plain"] * 1.5
        assert abs(rows["clean"]["gated"] - rows["clean"]["plain"]) < 0.02

    def test_the_polluted_stream_is_cached(self):
        assert mod._polluted(1000) is mod._polluted(1000)
