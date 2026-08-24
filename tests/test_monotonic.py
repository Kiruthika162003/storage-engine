from __future__ import annotations

import pytest

from store import monotonic as mod
from store.errors import ConfigError
from store.monotonic import HybridClock, Stamp


class TestClock:
    def test_a_negative_reading_is_refused(self):
        with pytest.raises(ConfigError):
            HybridClock().now(-1)

    def test_a_fresh_reading_stamps_the_wall(self):
        stamp = HybridClock().now(100)
        assert stamp == Stamp(wall=100, logical=0)

    def test_advancing_wall_resets_the_logical(self):
        clock = HybridClock()
        clock.now(100)
        clock.now(100)
        stamp = clock.now(150)
        assert stamp == Stamp(wall=150, logical=0)

    def test_a_repeated_reading_bumps_the_logical(self):
        clock = HybridClock()
        clock.now(100)
        stamp = clock.now(100)
        assert stamp == Stamp(wall=100, logical=1)

    def test_a_backward_reading_holds_the_wall(self):
        clock = HybridClock()
        clock.now(100)
        stamp = clock.now(50)
        assert stamp == Stamp(wall=100, logical=1)

    def test_stamps_strictly_increase(self):
        clock = HybridClock()
        stamps = [clock.now(r) for r in (5, 5, 3, 8, 8, 2)]
        keys = [stamp.key() for stamp in stamps]
        assert keys == sorted(set(keys))


class TestObserve:
    def test_an_older_stamp_changes_nothing_much(self):
        clock = HybridClock()
        clock.now(100)
        stamp = clock.observe(Stamp(wall=50, logical=3), 120)
        assert stamp == Stamp(wall=120, logical=0)

    def test_a_newer_stamp_is_folded_in(self):
        clock = HybridClock()
        stamp = clock.observe(Stamp(wall=200, logical=0), 100)
        assert stamp.wall == 200 and stamp.logical == 1

    def test_the_reply_exceeds_the_request(self):
        sender = HybridClock()
        receiver = HybridClock()
        request = sender.now(1000)
        reply = receiver.observe(request, 900)
        assert reply.key() > request.key()

    def test_matching_walls_take_the_higher_logical(self):
        clock = HybridClock()
        clock.now(100)
        clock.now(100)
        stamp = clock.observe(Stamp(wall=100, logical=9), 100)
        assert stamp.logical == 10


class TestMeasurements:
    def test_wall_ids_go_backward(self):
        assert mod.wall_ids_go_backward_under_a_step()

    def test_hybrid_stamps_do_not(self):
        assert mod.hybrid_stamps_never_go_backward()

    def test_causality_survives_skew(self):
        assert mod.causality_survives_an_exchange_between_skewed_clocks()

    def test_honest_clocks_cost_nothing(self):
        assert mod.the_wall_component_tracks_true_time_when_clocks_behave()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4
