"""Error budgets and burn rate alerts, replayed against synthetic months.

A 99.9 percent objective over a 30 day window grants a budget of 43200
failed requests at one thousand requests per tick and 43200 ticks per
month. The question an alert policy answers is when to page: a fast burn
window catches an outage in minutes, a slow burn window catches a leak in
days, and each is blind to the other's incident. Both blindnesses are
demonstrated on replayed traffic, not asserted.
"""

from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

RATE = 1000
WINDOW = 43200
OBJECTIVE = 0.999


@dataclass
class Month:
    """Failures per tick over a 30 day month of minutes."""

    failed: list[int]

    @classmethod
    def quiet(cls, seed: int, background: float = 0.0002) -> Month:
        source = random.Random(seed)
        failed = [
            sum(1 for _ in range(RATE) if source.random() < background)
            for _ in range(WINDOW)
        ]
        return cls(failed=failed)

    def with_outage(self, start: int, length: int, rate: float) -> Month:
        failed = list(self.failed)
        for tick in range(start, min(start + length, WINDOW)):
            failed[tick] = int(RATE * rate)
        return Month(failed=failed)

    def budget(self) -> int:
        return int(RATE * WINDOW * (1 - OBJECTIVE))

    def spent(self) -> int:
        return sum(self.failed)


@dataclass
class Policy:
    """Page when the error rate over the last `window` ticks exceeds
    `multiplier` times the budget rate."""

    window: int
    multiplier: float
    pages: list[int] = field(default_factory=list)

    def replay(self, month: Month) -> list[int]:
        self.pages = []
        allowed = (1 - OBJECTIVE) * self.multiplier
        rolling = 0
        for tick, failures in enumerate(month.failed):
            rolling += failures
            if tick >= self.window:
                rolling -= month.failed[tick - self.window]
            span = min(tick + 1, self.window)
            if rolling / (span * RATE) > allowed:
                self.pages.append(tick)
        return self.pages


def fast() -> Policy:
    return Policy(window=60, multiplier=14.4)


def slow() -> Policy:
    return Policy(window=4320, multiplier=1.0)


@functools.cache
def the_fast_window_pages_in_minutes() -> bool:
    """A 20 percent outage at tick 5000 pages the fast policy by tick 5004.

    14.4 times the budget rate is 1.44 percent errors; a 20 percent outage
    crosses that within the first handful of minutes of the hour window.
    """
    month = Month.quiet(7).with_outage(5000, 120, 0.20)
    pages = fast().replay(month)
    first = next(page for page in pages if page >= 5000)
    return first <= 5004


@functools.cache
def the_fast_window_sleeps_through_a_leak() -> bool:
    """A 0.5 percent error leak never pages the fast policy in a month.

    0.5 percent is five times the objective's allowance and will exhaust
    the budget in six days, but it never approaches 1.44 percent over any
    hour, so the fast page stays silent while the budget drains.
    """
    month = Month.quiet(7).with_outage(0, WINDOW, 0.005)
    return fast().replay(month) == [] and month.spent() > month.budget()


@functools.cache
def the_slow_window_catches_the_leak() -> bool:
    """The same leak pages the slow policy once its 3 day window fills.

    At 0.5 percent errors even the partial window averages five times the
    allowance, so the first page comes at tick 0. A real policy would wait
    for the window to fill; this one pages the moment the rate is proven.
    """
    month = Month.quiet(7).with_outage(0, WINDOW, 0.005)
    pages = slow().replay(month)
    return bool(pages) and pages[0] < 1440


@functools.cache
def the_slow_window_pages_long_after_recovery() -> bool:
    """A two hour outage keeps the slow policy paging for a day after it ends.

    The outage's failures sit in the 3 day window long after the incident:
    the last slow page comes 4301 ticks after recovery, three days of pages
    for an incident already fixed. The last fast page comes 54 ticks after.
    This is why the fast window exists: it tracks now, not history.
    """
    month = Month.quiet(7).with_outage(5000, 120, 0.20)
    slow_pages = slow().replay(month)
    fast_pages = fast().replay(month)
    return slow_pages[-1] - 5120 > 1440 and fast_pages[-1] - 5120 <= 60


@functools.cache
def background_noise_spends_a_fifth_of_the_budget() -> bool:
    """A 0.02 percent background error rate costs 8582 of 43200 budget.

    The quiet month with no incident at all still spends 20 percent of the
    budget on background noise. Alerting on remaining budget alone would
    treat this as a problem; the burn rate policies correctly ignore it
    because 0.02 percent is a fifth of the allowance, not a multiple.
    """
    month = Month.quiet(7)
    spent = month.spent()
    silent = fast().replay(month) == [] and slow().replay(month) == []
    return silent and 0.18 < spent / month.budget() < 0.22


@functools.cache
def summarise() -> dict:
    return {
        "module": "store.errorbudget",
        "the_fast_window_pages_in_minutes": the_fast_window_pages_in_minutes(),
        "the_fast_window_sleeps_through_a_leak": the_fast_window_sleeps_through_a_leak(),
        "the_slow_window_catches_the_leak": the_slow_window_catches_the_leak(),
        "the_slow_window_pages_long_after_recovery": (
            the_slow_window_pages_long_after_recovery()
        ),
        "background_noise_spends_a_fifth_of_the_budget": (
            background_noise_spends_a_fifth_of_the_budget()
        ),
    }
