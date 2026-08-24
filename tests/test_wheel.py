from __future__ import annotations

import pytest

from store import wheel as mod
from store.errors import ConfigError
from store.wheel import SLOTS, HeapTimers, WheelTimers


class TestHeap:
    def test_a_timer_fires_at_its_deadline(self):
        heap = HeapTimers()
        heap.schedule(5, 1)
        assert heap.due(5) == [1]

    def test_a_timer_does_not_fire_early(self):
        heap = HeapTimers()
        heap.schedule(5, 1)
        assert heap.due(4) == []

    def test_timers_fire_soonest_first(self):
        heap = HeapTimers()
        heap.schedule(9, 2)
        heap.schedule(3, 1)
        assert heap.due(10) == [1, 2]

    def test_fired_timers_do_not_refire(self):
        heap = HeapTimers()
        heap.schedule(5, 1)
        heap.due(5)
        assert heap.due(100) == []

    def test_comparisons_are_counted(self):
        heap = HeapTimers()
        for at in range(100):
            heap.schedule(at, at)
        assert heap.comparisons > 0


class TestWheel:
    def test_a_timer_fires_at_its_deadline(self):
        wheel = WheelTimers()
        wheel.schedule(5, 1)
        assert wheel.advance(5) == [1]

    def test_a_timer_does_not_fire_early(self):
        wheel = WheelTimers()
        wheel.schedule(5, 1)
        assert wheel.advance(4) == []

    def test_the_clock_moves_forward(self):
        wheel = WheelTimers()
        wheel.advance(10)
        assert wheel.now == 11

    def test_the_horizon_tracks_the_clock(self):
        wheel = WheelTimers()
        wheel.advance(10)
        assert wheel.horizon == 11 + SLOTS - 1

    def test_a_deadline_past_the_horizon_is_refused(self):
        wheel = WheelTimers()
        with pytest.raises(ConfigError):
            wheel.schedule(SLOTS + 10, 1)

    def test_a_past_deadline_is_refused(self):
        wheel = WheelTimers()
        wheel.advance(10)
        with pytest.raises(ConfigError):
            wheel.schedule(5, 1)

    def test_refusals_are_counted(self):
        wheel = WheelTimers()
        with pytest.raises(ConfigError):
            wheel.schedule(SLOTS + 10, 1)
        assert wheel.refused == 1

    def test_colliding_slots_fire_separately(self):
        wheel = WheelTimers()
        wheel.schedule(3, 1)
        assert wheel.advance(3) == [1]
        wheel.schedule(3 + SLOTS, 2)
        assert wheel.advance(3 + SLOTS) == [2]

    def test_many_timers_in_one_slot_all_fire(self):
        wheel = WheelTimers()
        for name in range(10):
            wheel.schedule(7, name)
        assert sorted(wheel.advance(7)) == list(range(10))

    def test_an_advance_over_many_ticks_fires_everything_due(self):
        wheel = WheelTimers()
        for deadline in (3, 50, 120):
            wheel.schedule(deadline, deadline)
        assert sorted(wheel.advance(130)) == [3, 50, 120]


class TestMeasurements:
    def test_both_fire_alike(self):
        assert mod.both_timers_fire_the_same_names_at_the_same_times()

    def test_the_wheel_skips_the_comparisons(self):
        assert mod.the_wheel_schedules_without_comparing()

    def test_the_horizon_is_a_wall(self):
        assert mod.the_horizon_is_a_wall_not_a_slope()

    def test_collisions_stay_apart(self):
        assert mod.a_slot_holds_colliding_deadlines_apart()

    def test_the_past_is_refused(self):
        assert mod.past_deadlines_are_refused()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5
