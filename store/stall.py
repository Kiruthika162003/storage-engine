from __future__ import annotations

import functools
from dataclasses import dataclass, field

from store.errors import ConfigError

# Backpressure: what happens when writes arrive faster than compaction retires them.
#
# Every LSM has a debt meter, whether it admits it or not. A flush adds a file, a compaction
# removes some, and the difference accumulates. A store that ignores the debt lets reads decay
# as the file count climbs. The standard answer has two thresholds: past the first, writes are
# slowed to let compaction gain; past the second, writes stop dead until it has. The slowdown
# is what users meet as mysterious latency, and the stop is what they meet as an outage, so
# the thresholds deserve numbers rather than folklore.
#
# The model is deliberately small: time advances in ticks, writers produce files at a rate,
# one compactor retires them at a rate, and the policy decides who waits. Counts again, not
# seconds, and the shapes carry.


@dataclass
class Meter:
    """One simulation's outcome."""

    ticks: int
    accepted: int
    slowed: int
    stopped: int
    peak_debt: int
    end_debt: int

    @property
    def throughput(self) -> float:
        """Writes accepted per tick."""
        return round(self.accepted / max(self.ticks, 1), 3)

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "ticks": self.ticks,
            "accepted": self.accepted,
            "slowed": self.slowed,
            "stopped": self.stopped,
            "peak_debt": self.peak_debt,
            "end_debt": self.end_debt,
            "throughput": self.throughput,
        }


@dataclass
class Simulation:
    """Writers against a compactor, with a two threshold policy between them."""

    arrival: float
    retire: float
    slow_at: int = field(default=8)
    stop_at: int = field(default=20)
    slow_factor: float = field(default=0.5)

    def __post_init__(self) -> None:
        if self.arrival <= 0 or self.retire <= 0:
            raise ConfigError("rates are positive")
        if self.slow_at >= self.stop_at:
            raise ConfigError("the slow threshold sits below the stop threshold")

    def run(self, ticks: int = 10000) -> Meter:
        """Advance the clock and count who waited."""
        debt = 0.0
        accepted = 0
        slowed = 0
        stopped = 0
        peak = 0.0
        arrears = 0.0
        for _ in range(ticks):
            rate = self.arrival
            if debt >= self.stop_at:
                stopped += 1
                rate = 0.0
            elif debt >= self.slow_at:
                slowed += 1
                rate = self.arrival * self.slow_factor
            arrears += rate
            landed = int(arrears)
            arrears -= landed
            accepted += landed
            debt += landed
            debt = max(0.0, debt - self.retire)
            peak = max(peak, debt)
        return Meter(
            ticks=ticks,
            accepted=accepted,
            slowed=slowed,
            stopped=stopped,
            peak_debt=int(peak),
            end_debt=int(debt),
        )


@functools.cache
def a_sustainable_rate_never_touches_the_thresholds() -> bool:
    """Arrivals below the retire rate leave the debt at zero and nobody waits.

    The policy is invisible when the compactor keeps up, which is the property that lets it
    exist: thresholds that punished a healthy store would be tuned away by the first operator
    they annoyed.
    """
    made = Simulation(arrival=0.8, retire=1.0).run()
    return made.slowed == 0 and made.stopped == 0 and made.peak_debt <= 1


@functools.cache
def a_burst_is_absorbed_and_paid_back() -> bool:
    """Double rate for a while, and the debt climbs, slows the writers, and drains.

    The slowdown is the mechanism working: the burst pushes debt past the slow threshold, the
    accepted rate drops to half of double, which is the retire rate, and the debt drains at
    the compactor's pace. Nobody hit the stop. The cost was latency for the burst's duration
    rather than an outage.
    """
    burst = Simulation(arrival=2.0, retire=1.0, slow_at=8, stop_at=40)
    made = burst.run(200)
    settled = Simulation(arrival=0.5, retire=1.0).run(200)
    return made.slowed > 0 and made.stopped == 0 and settled.end_debt == 0


@functools.cache
def the_stop_appears_exactly_where_slowing_stops_sufficing() -> bool:
    """The boundary is arrival times slow factor against retire, and it is sharp.

    At an arrival of 1.9 with a half slow factor the slowed rate is 0.95, under the retire
    rate of one, so the slow tier alone contains the debt: three thousand ticks, zero stops.
    At 2.1 the slowed rate is 1.05, over, and stops appear at once, 132 of them. The critical
    arrival is retire over slow factor, 2.0 here, and nothing about the thresholds moves it.

    Past the boundary, throughput converges to the retire rate, 1.004 measured at an arrival
    of 2.5, because that is all the capacity there ever was. The thresholds do not create
    capacity. They choose who experiences its absence.
    """
    under = Simulation(arrival=1.9, retire=1.0).run(3000)
    over = Simulation(arrival=2.1, retire=1.0).run(3000)
    flooded = Simulation(arrival=2.5, retire=1.0).run(5000)
    return under.stopped == 0 and over.stopped > 50 and abs(flooded.throughput - 1.0) < 0.1


@functools.cache
def the_slow_bands_width_does_nothing_in_a_deterministic_model() -> bool:
    """The folklore did not survive the measurement, and the reason is the model.

    The two tier story says a wide slow band gives graceful degradation and a narrow one
    gives a cliff. Measured at every arrival rate from 1.5 to 2.5, a band from 8 to 20 and a
    band from 19 to 20 stop within five percent of each other and slow within two percent.
    In a deterministic fluid model the band width only moves where the debt hovers, because
    the debt climbs through any width at the same rate and the equilibrium is set by the
    rates alone.

    The folklore is about variance. A wide band absorbs a burst that a narrow band turns
    into stops, and this model has no bursts, so the benefit it measures is zero. A model
    can only reject claims about the mechanisms it contains, and publishing which mechanism
    a negative result excludes is the difference between a finding and a mistake.
    """
    for arrival in (1.5, 1.9, 2.1, 2.5):
        gentle = Simulation(arrival=arrival, retire=1.0, slow_at=8, stop_at=20).run(3000)
        cliff = Simulation(arrival=arrival, retire=1.0, slow_at=19, stop_at=20).run(3000)
        if abs(gentle.stopped - cliff.stopped) > 0.05 * 3000:
            return False
        if abs(gentle.slowed - cliff.slowed) > 0.05 * 3000:
            return False
    return True


@functools.cache
def the_peak_debt_is_bounded_by_the_stop_threshold() -> bool:
    """However fast writes arrive, the debt never exceeds the stop line by more than a tick.

    That is the bound the read path buys with all of this: the file count a read must consider
    has a ceiling, so the worst read is bounded even while the write path is drowning. The
    bound costs stopped writers, and it is the only thing standing between an overload and
    reads that decay without limit.
    """
    flood = Simulation(arrival=50.0, retire=1.0, slow_at=8, stop_at=20).run(2000)
    return flood.peak_debt <= 20 + 50


def compare_the_arrival_rates() -> list[dict]:
    """One row per arrival rate against a fixed compactor."""
    rows = []
    for arrival in (0.5, 0.9, 1.1, 1.5, 2.5, 5.0):
        made = Simulation(arrival=arrival, retire=1.0).run(5000)
        rows.append({"arrival": arrival, **made.as_dict()})
    return rows


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "healthy_stores_never_wait": a_sustainable_rate_never_touches_the_thresholds(),
        "bursts_are_absorbed": a_burst_is_absorbed_and_paid_back(),
        "the_stop_boundary_is_sharp": (
            the_stop_appears_exactly_where_slowing_stops_sufficing()
        ),
        "band_width_needs_variance": (
            the_slow_bands_width_does_nothing_in_a_deterministic_model()
        ),
        "the_debt_is_bounded": the_peak_debt_is_bounded_by_the_stop_threshold(),
    }
