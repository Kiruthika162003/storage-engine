from __future__ import annotations

import pytest

from store import aging as mod
from store.aging import Job, Scheduler
from store.errors import ConfigError


class TestScheduler:
    def test_a_negative_rate_is_refused(self):
        with pytest.raises(ConfigError):
            Scheduler(aging_rate=-1)

    def test_an_empty_tick_serves_nobody(self):
        assert Scheduler(aging_rate=0).tick() is None

    def test_the_higher_priority_serves_first(self):
        scheduler = Scheduler(aging_rate=0)
        scheduler.submit(Job(name=1, priority=1, arrived=0))
        scheduler.submit(Job(name=2, priority=9, arrived=0))
        assert scheduler.tick().name == 2

    def test_ties_go_to_the_oldest(self):
        scheduler = Scheduler(aging_rate=0)
        scheduler.submit(Job(name=1, priority=5, arrived=0))
        scheduler.submit(Job(name=2, priority=5, arrived=1))
        assert scheduler.tick().name == 1

    def test_effective_priority_grows_with_wait(self):
        scheduler = Scheduler(aging_rate=1.0)
        job = Job(name=1, priority=0, arrived=0)
        scheduler.now = 5
        assert scheduler.effective(job) == 5.0

    def test_an_aged_job_overtakes_a_fresh_one(self):
        scheduler = Scheduler(aging_rate=1.0)
        scheduler.submit(Job(name=1, priority=0, arrived=0))
        scheduler.now = 20
        scheduler.submit(Job(name=2, priority=10, arrived=20))
        assert scheduler.tick().name == 1

    def test_waits_are_recorded_per_priority(self):
        scheduler = Scheduler(aging_rate=0)
        scheduler.submit(Job(name=1, priority=3, arrived=0))
        scheduler.tick()
        assert scheduler.waits[3] == [1]

    def test_worst_wait_counts_the_pending(self):
        scheduler = Scheduler(aging_rate=0)
        scheduler.submit(Job(name=1, priority=3, arrived=0))
        scheduler.now = 10
        assert scheduler.worst_wait(3) == 10


class TestMeasurements:
    def test_strict_priority_starves(self):
        assert mod.strict_priority_starves_the_low_job_literally()

    def test_the_bound_is_gap_over_rate(self):
        assert mod.aging_bounds_the_wait_by_the_priority_gap_over_the_rate()

    def test_the_rate_is_the_trade(self):
        assert mod.a_faster_rate_shortens_the_worst_wait_and_costs_the_high_class()

    def test_equals_are_fifo(self):
        assert mod.equal_priorities_serve_in_arrival_order()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4
