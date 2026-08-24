from __future__ import annotations

import pytest

from store import batch as mod
from store.batch import Committer
from store.errors import Closed, ConfigError
from store.wal import recover


class TestSubmit:
    def test_a_submission_gets_a_ticket(self):
        made = Committer(close_at=10)
        assert made.submit(b"k", b"v").sequence == 1

    def test_sequences_climb(self):
        made = Committer(close_at=10)
        first = made.submit(b"a", b"v")
        assert made.submit(b"b", b"v").sequence == first.sequence + 1

    def test_a_zero_batch_size_is_refused(self):
        with pytest.raises(ConfigError):
            Committer(close_at=0)

    def test_a_full_batch_flushes_itself(self):
        made = Committer(close_at=2)
        made.submit(b"a", b"v")
        made.submit(b"b", b"v")
        assert made.batches == 1

    def test_an_unfilled_batch_stays_open(self):
        made = Committer(close_at=3)
        made.submit(b"a", b"v")
        assert made.batches == 0 and len(made.open_batch) == 1

    def test_a_delete_joins_the_batch(self):
        made = Committer(close_at=10)
        made.delete(b"k")
        assert len(made.open_batch) == 1

    def test_tickets_carry_their_batch(self):
        made = Committer(close_at=1)
        first = made.submit(b"a", b"v")
        second = made.submit(b"b", b"v")
        assert first.batch != second.batch


class TestSettlement:
    def test_an_open_ticket_is_unsettled(self):
        made = Committer(close_at=10)
        assert not made.submit(b"k", b"v").settled

    def test_a_flush_settles_the_batch(self):
        made = Committer(close_at=10)
        ticket = made.submit(b"k", b"v")
        made.flush()
        assert ticket.settled

    def test_a_flush_of_nothing_is_a_no_op(self):
        made = Committer(close_at=10)
        assert made.flush() == 0 and made.batches == 0

    def test_every_settled_record_is_durable(self):
        made = Committer(close_at=4)
        for at in range(8):
            made.submit(f"k{at}".encode(), b"v")
        found = recover(bytes(made.log.disk.durable))
        assert len(found.records) == 8

    def test_an_open_batch_is_not_durable(self):
        made = Committer(close_at=10)
        made.submit(b"k", b"v")
        found = recover(bytes(made.log.disk.durable))
        assert len(found.records) == 0

    def test_the_settled_count_tracks_tickets(self):
        made = Committer(close_at=2)
        for at in range(6):
            made.submit(f"k{at}".encode(), b"v")
        assert made.settled == 6


class TestClose:
    def test_close_settles_the_stragglers(self):
        made = Committer(close_at=100)
        ticket = made.submit(b"k", b"v")
        made.close()
        assert ticket.settled

    def test_a_closed_committer_refuses(self):
        made = Committer(close_at=10)
        made.close()
        with pytest.raises(Closed):
            made.submit(b"k", b"v")

    def test_as_dict_counts_the_pending(self):
        made = Committer(close_at=10)
        made.submit(b"k", b"v")
        assert made.as_dict()["pending"] == 1


class TestMeasurements:
    def test_syncs_drop_by_the_batch(self):
        assert mod.the_sync_count_drops_by_the_batch_size()

    def test_nothing_settles_early(self):
        assert mod.nothing_settles_before_its_sync()

    def test_a_batch_settles_together(self):
        assert mod.a_batch_settles_together()

    def test_close_flushes_first(self):
        assert mod.a_closed_committer_refuses_and_flushes_first()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4

    def test_the_size_table_halves_the_syncs(self):
        rows = mod.compare_the_batch_sizes(4000)
        syncs = [row["syncs"] for row in rows]
        assert syncs == sorted(syncs, reverse=True)

    def test_writes_per_sync_tracks_the_threshold(self):
        rows = mod.compare_the_batch_sizes(4000)
        assert rows[3]["writes_per_sync"] == 8.0

    def test_the_committed_stream_is_cached(self):
        assert mod._committed(100, 4) is mod._committed(100, 4)
