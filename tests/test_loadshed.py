from __future__ import annotations

import pytest

from store import loadshed as mod
from store.errors import ConfigError
from store.loadshed import Server


class TestServer:
    def test_bad_settings_are_refused(self):
        with pytest.raises(ConfigError):
            Server(capacity_per_tick=0, queue_limit=10, shed=False)

    def test_an_unknown_priority_is_refused(self):
        server = Server(capacity_per_tick=1, queue_limit=10, shed=False)
        with pytest.raises(ConfigError):
            server.offer("cosmic")

    def test_offers_queue_below_the_limit(self):
        server = Server(capacity_per_tick=1, queue_limit=2, shed=True)
        server.offer("normal")
        assert len(server.queue) == 1

    def test_the_unshedded_queue_grows_past_the_limit(self):
        server = Server(capacity_per_tick=1, queue_limit=2, shed=False)
        for _ in range(5):
            server.offer("normal")
        assert len(server.queue) == 5

    def test_the_shedder_evicts_the_lowest_class(self):
        server = Server(capacity_per_tick=1, queue_limit=2, shed=True)
        server.offer("batch")
        server.offer("normal")
        server.offer("critical")
        priorities = [priority for priority, _ in server.queue]
        assert "batch" not in priorities and server.meter.dropped == {"batch": 1}

    def test_a_low_arrival_to_a_full_high_queue_is_dropped(self):
        server = Server(capacity_per_tick=1, queue_limit=2, shed=True)
        server.offer("critical")
        server.offer("critical")
        server.offer("batch")
        assert server.meter.dropped == {"batch": 1}

    def test_service_is_fifo_within_capacity(self):
        server = Server(capacity_per_tick=2, queue_limit=10, shed=False)
        server.offer("normal")
        server.offer("batch")
        server.offer("critical")
        server.tick()
        assert server.meter.served == {"normal": 1, "batch": 1}

    def test_ancient_work_times_out(self):
        server = Server(capacity_per_tick=1, queue_limit=100, shed=False)
        for _ in range(50):
            server.offer("normal")
        for _ in range(40):
            server.tick()
        assert server.meter.timed_out.get("normal", 0) > 0


class TestMeasurements:
    def test_fairness_fails_everyone(self):
        assert mod.the_unshedded_server_fails_everyone_alike()

    def test_critical_rides_through(self):
        assert mod.the_shedder_serves_every_critical_request()

    def test_the_bill_lands_on_batch(self):
        assert mod.the_bill_lands_on_the_batch_class()

    def test_drops_beat_timeouts(self):
        assert mod.drops_are_cheap_and_timeouts_are_not()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4
