"""A queue on a store: visibility timeouts, redelivery, and the poison job.

At-least-once delivery is a lease, not a handoff: a taken job becomes
invisible for a timeout and returns if no ack arrives. Every guarantee
has a measured cost here: how many duplicates a crashing worker causes,
how long a poison job stays in circulation without a retry cap, and what
the dead letter drawer actually buys.
"""

from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

TIMEOUT = 20
CAP = 3


@dataclass
class Job:
    number: int
    body: bytes
    taken_at: int = -1
    deliveries: int = 0
    done: bool = False
    dead: bool = False


@dataclass
class Queue:
    timeout: int = TIMEOUT
    cap: int = 0
    jobs: list[Job] = field(default_factory=list)
    delivered: int = 0
    buried: int = 0

    def put(self, body: bytes) -> int:
        job = Job(number=len(self.jobs), body=body)
        self.jobs.append(job)
        return job.number

    def take(self, now: int) -> Job | None:
        for job in self.jobs:
            if job.done or job.dead:
                continue
            if job.taken_at >= 0 and now - job.taken_at < self.timeout:
                continue
            if self.cap and job.deliveries >= self.cap:
                job.dead = True
                self.buried += 1
                continue
            job.taken_at = now
            job.deliveries += 1
            self.delivered += 1
            return job
        return None

    def ack(self, number: int) -> None:
        self.jobs[number].done = True

    def nack(self, number: int) -> None:
        self.jobs[number].taken_at = -1

    def outstanding(self) -> int:
        return sum(1 for job in self.jobs if not job.done and not job.dead)


@dataclass
class Worker:
    """Processes a job in `steps` ticks; may crash mid-job."""

    steps: int
    crash_rate: float
    source: random.Random
    working_on: Job | None = None
    finish_at: int = -1
    completed: int = 0
    crashes: int = 0

    def tick(self, queue: Queue, now: int) -> None:
        if self.working_on is not None:
            if self.source.random() < self.crash_rate:
                self.crashes += 1
                self.working_on = None
                return
            if now >= self.finish_at:
                queue.ack(self.working_on.number)
                self.completed += 1
                self.working_on = None
            return
        job = queue.take(now)
        if job is not None:
            self.working_on = job
            self.finish_at = now + self.steps


def _mill(seed: int, cap: int, poison: bool, horizon: int = 3000) -> Queue:
    queue = Queue(cap=cap)
    for number in range(200):
        queue.put(f"job-{number:03d}".encode())
    workers = [
        Worker(steps=4, crash_rate=0.02, source=random.Random(seed + at))
        for at in range(4)
    ]
    for now in range(horizon):
        for worker in workers:
            if poison and worker.working_on is not None and (
                worker.working_on.number == 13
            ):
                worker.working_on = None
                continue
            worker.tick(queue, now)
    return queue


@functools.cache
def crashes_cause_duplicates_not_losses() -> bool:
    """200 jobs, crashing workers: 218 deliveries, 200 completions, 0 lost.

    Every crash abandons a lease that times out and returns the job, so
    the meter shows the guarantee's true name: at least once is 1.09
    deliveries per job here, and exactly once is the 18 duplicates a
    downstream idempotency key must absorb.
    """
    queue = _mill(5, cap=0, poison=False)
    finished = sum(1 for job in queue.jobs if job.done)
    return finished == 200 and queue.delivered == 218 and queue.outstanding() == 0


@functools.cache
def the_poison_job_circulates_forever_uncapped() -> bool:
    """Job 13 is delivered 149 times in 3000 ticks and is still not done.

    A job that kills its worker returns every timeout, forever, spending
    a worker slot each lap. Without a cap the queue finishes every other
    job and keeps feeding the same poison to whoever is free.
    """
    queue = _mill(5, cap=0, poison=True)
    poison = queue.jobs[13]
    others_done = sum(1 for job in queue.jobs if job.done)
    return not poison.done and poison.deliveries == 149 and others_done == 199


@functools.cache
def the_cap_buries_the_poison_after_three_laps() -> bool:
    """With cap 3 the poison is buried on delivery 3 and the mill runs on.

    The dead letter drawer converts an infinite loop into a bounded loss:
    one job set aside for inspection, 199 completed, and the 146 wasted
    deliveries the uncapped run paid never happen.
    """
    queue = _mill(5, cap=CAP, poison=True)
    poison = queue.jobs[13]
    others_done = sum(1 for job in queue.jobs if job.done)
    return (
        poison.dead
        and poison.deliveries == CAP
        and queue.buried == 1
        and others_done == 199
    )


@functools.cache
def a_short_timeout_multiplies_duplicates() -> bool:
    """Timeout 3 against 4-step jobs: 370 deliveries for 200 completions.

    A lease shorter than the work redelivers healthy jobs mid-flight: the
    duplicate rate jumps from 1.09 to 1.85 per job with no extra crashes.
    The timeout must cover honest work, not just detect dead workers.
    """
    queue = Queue(timeout=3, cap=0)
    for number in range(200):
        queue.put(f"job-{number:03d}".encode())
    workers = [
        Worker(steps=4, crash_rate=0.02, source=random.Random(5 + at))
        for at in range(4)
    ]
    for now in range(3000):
        for worker in workers:
            worker.tick(queue, now)
    finished = sum(1 for job in queue.jobs if job.done)
    return finished == 200 and queue.delivered == 370


@functools.cache
def summarise() -> dict:
    return {
        "module": "store.jobqueue",
        "crashes_cause_duplicates_not_losses": crashes_cause_duplicates_not_losses(),
        "the_poison_job_circulates_forever_uncapped": (
            the_poison_job_circulates_forever_uncapped()
        ),
        "the_cap_buries_the_poison_after_three_laps": (
            the_cap_buries_the_poison_after_three_laps()
        ),
        "a_short_timeout_multiplies_duplicates": a_short_timeout_multiplies_duplicates(),
    }
