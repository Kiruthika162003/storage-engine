from __future__ import annotations

import random

import pytest

from store.jobqueue import (
    CAP,
    Job,
    Queue,
    Worker,
    a_short_timeout_multiplies_duplicates,
    crashes_cause_duplicates_not_losses,
    summarise,
    the_cap_buries_the_poison_after_three_laps,
    the_poison_job_circulates_forever_uncapped,
)


class TestQueue:
    def test_put_numbers_jobs_in_order(self):
        queue = Queue()
        assert queue.put(b"a") == 0
        assert queue.put(b"b") == 1

    def test_take_leases_the_oldest_free_job(self):
        queue = Queue()
        queue.put(b"a")
        queue.put(b"b")
        assert queue.take(0).number == 0
        assert queue.take(0).number == 1

    def test_a_leased_job_is_invisible(self):
        queue = Queue(timeout=10)
        queue.put(b"a")
        queue.take(0)
        assert queue.take(5) is None

    def test_an_expired_lease_returns(self):
        queue = Queue(timeout=10)
        queue.put(b"a")
        queue.take(0)
        job = queue.take(10)
        assert job is not None and job.deliveries == 2

    def test_an_acked_job_never_returns(self):
        queue = Queue(timeout=10)
        queue.put(b"a")
        queue.ack(queue.take(0).number)
        assert queue.take(100) is None

    def test_a_nack_makes_the_job_immediately_visible(self):
        queue = Queue(timeout=10)
        queue.put(b"a")
        queue.nack(queue.take(0).number)
        assert queue.take(1) is not None

    def test_the_cap_buries_at_the_limit(self):
        queue = Queue(timeout=1, cap=2)
        queue.put(b"a")
        queue.take(0)
        queue.take(10)
        assert queue.take(20) is None
        assert queue.jobs[0].dead and queue.buried == 1

    def test_outstanding_ignores_done_and_dead(self):
        queue = Queue()
        queue.put(b"a")
        queue.put(b"b")
        queue.ack(0)
        queue.jobs[1].dead = True
        assert queue.outstanding() == 0


class TestWorker:
    def test_a_worker_completes_a_job(self):
        queue = Queue()
        queue.put(b"a")
        worker = Worker(steps=2, crash_rate=0.0, source=random.Random(1))
        for now in range(5):
            worker.tick(queue, now)
        assert worker.completed == 1
        assert queue.jobs[0].done

    def test_a_crashed_job_is_redelivered(self):
        queue = Queue(timeout=3)
        queue.put(b"a")
        worker = Worker(steps=2, crash_rate=1.0, source=random.Random(1))
        worker.tick(queue, 0)
        worker.tick(queue, 1)
        assert worker.crashes == 1
        assert queue.take(10) is not None


class TestJob:
    def test_a_fresh_job_is_untaken(self):
        job = Job(number=0, body=b"x")
        assert job.taken_at == -1 and job.deliveries == 0


class TestClaims:
    @pytest.mark.parametrize(
        "claim",
        [
            crashes_cause_duplicates_not_losses,
            the_poison_job_circulates_forever_uncapped,
            the_cap_buries_the_poison_after_three_laps,
            a_short_timeout_multiplies_duplicates,
        ],
    )
    def test_claim_holds(self, claim):
        assert claim() is True

    def test_summary_is_all_true(self):
        told = summarise()
        assert all(value for name, value in told.items() if name != "module")

    def test_the_cap_constant_is_three(self):
        assert CAP == 3
