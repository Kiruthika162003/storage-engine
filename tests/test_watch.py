from __future__ import annotations

import pytest

from store import watch as mod
from store.errors import Closed, ConfigError
from store.watch import Feed


class TestPublish:
    def test_a_zero_buffer_is_refused(self):
        with pytest.raises(ConfigError):
            Feed(buffer_records=0)

    def test_publishes_take_sequences(self):
        feed = Feed()
        assert feed.publish(b"k", b"v") == 1
        assert feed.publish(b"k", b"v") == 2

    def test_the_buffer_holds_its_bound(self):
        feed = Feed(buffer_records=5)
        for at in range(20):
            feed.publish(f"k{at}".encode(), b"v")
        assert len(feed.held) == 5

    def test_the_floor_tracks_the_evictions(self):
        feed = Feed(buffer_records=5)
        for at in range(20):
            feed.publish(f"k{at}".encode(), b"v")
        assert feed.floor == 15


class TestSubscribe:
    def test_a_subscriber_starts_at_the_present(self):
        feed = Feed()
        feed.publish(b"old", b"v")
        feed.subscribe("a")
        assert feed.poll("a") == []

    def test_a_double_subscribe_is_refused(self):
        feed = Feed()
        feed.subscribe("a")
        with pytest.raises(ConfigError):
            feed.subscribe("a")

    def test_an_unknown_poller_is_refused(self):
        with pytest.raises(Closed):
            Feed().poll("ghost")


class TestPoll:
    def test_a_poll_returns_the_new_writes(self):
        feed = Feed()
        feed.subscribe("a")
        feed.publish(b"k1", b"v1")
        feed.publish(b"k2", b"v2")
        found = feed.poll("a")
        assert [entry[1] for entry in found] == [b"k1", b"k2"]

    def test_a_poll_advances_the_cursor(self):
        feed = Feed()
        feed.subscribe("a")
        feed.publish(b"k", b"v")
        feed.poll("a")
        assert feed.poll("a") == []

    def test_the_limit_is_respected(self):
        feed = Feed()
        feed.subscribe("a")
        for at in range(10):
            feed.publish(f"k{at}".encode(), b"v")
        assert len(feed.poll("a", limit=3)) == 3

    def test_batches_resume_where_they_stopped(self):
        feed = Feed()
        feed.subscribe("a")
        for at in range(6):
            feed.publish(f"k{at}".encode(), b"v")
        first = feed.poll("a", limit=3)
        second = feed.poll("a", limit=3)
        assert [e[0] for e in first + second] == [1, 2, 3, 4, 5, 6]

    def test_a_lapped_subscriber_is_resynced(self):
        feed = Feed(buffer_records=3)
        feed.subscribe("a")
        for at in range(10):
            feed.publish(f"k{at}".encode(), b"v")
        with pytest.raises(Closed):
            feed.poll("a")
        assert feed.resyncs == 1

    def test_lag_measures_the_distance(self):
        feed = Feed()
        feed.subscribe("a")
        for at in range(7):
            feed.publish(f"k{at}".encode(), b"v")
        assert feed.lag("a") == 7


class TestMeasurements:
    def test_keeping_up_sees_everything(self):
        assert mod.a_keeping_up_subscriber_sees_every_write_in_order()

    def test_the_slow_are_cut_loose(self):
        assert mod.a_slow_subscriber_is_cut_loose_not_carried()

    def test_resync_lands_at_the_present(self):
        assert mod.the_resync_position_is_the_present()

    def test_lag_warns_before_the_cut(self):
        assert mod.lag_is_visible_before_it_is_fatal()

    def test_cursors_are_independent(self):
        assert mod.two_subscribers_run_independent_cursors()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5
