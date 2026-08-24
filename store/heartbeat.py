"""Failure detection: the fixed timeout against the learned one.

A process is declared dead when its heartbeats stop arriving. The fixed
timeout answers with a constant; the adaptive detector learns the
arrival jitter and scales its patience to what it has seen. Both watch
the same jittery healthy process and the same actual death, and the two
meters that matter are counted: false alarms in a quiet week, and ticks
of delay before a real death is called.
"""

from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

INTERVAL = 10


@dataclass
class Fixed:
    timeout: int
    last_seen: int = 0
    alarms: int = 0

    def beat(self, now: int) -> None:
        self.last_seen = now

    def suspects(self, now: int) -> bool:
        return now - self.last_seen > self.timeout


@dataclass
class Adaptive:
    """Patience is the observed mean gap plus four standard deviations."""

    gaps: list[int] = field(default_factory=list)
    last_seen: int = 0
    alarms: int = 0

    def beat(self, now: int) -> None:
        if self.gaps or self.last_seen:
            self.gaps.append(now - self.last_seen)
            if len(self.gaps) > 100:
                self.gaps.pop(0)
        self.last_seen = now

    def patience(self) -> float:
        if len(self.gaps) < 8:
            return INTERVAL * 4.0
        mean = sum(self.gaps) / len(self.gaps)
        spread = (
            sum((gap - mean) ** 2 for gap in self.gaps) / len(self.gaps)
        ) ** 0.5
        return mean + 4 * spread

    def suspects(self, now: int) -> bool:
        return now - self.last_seen > self.patience()


def _watch(detector, gaps: list[int], die_at: int | None) -> tuple[int, int | None]:
    """False alarms while alive, and detection delay after death."""
    now = 0
    alarms = 0
    suspected_since = None
    arrivals = []
    for gap in gaps:
        now += gap
        arrivals.append(now)
    horizon = arrivals[-1] if die_at is None else die_at + 400
    beats = iter(arrivals)
    coming = next(beats)
    alive_alarm_ticks = 0
    detected_at = None
    for tick in range(1, horizon):
        while coming is not None and coming == tick:
            if die_at is None or tick <= die_at:
                detector.beat(tick)
            coming = next(beats, None)
        if detector.suspects(tick):
            if die_at is not None and tick > die_at:
                if detected_at is None:
                    detected_at = tick
            elif suspected_since is None:
                suspected_since = tick
                alarms += 1
                alive_alarm_ticks += 1
        else:
            suspected_since = None
    delay = None if detected_at is None or die_at is None else detected_at - die_at
    return alarms, delay


def _jittery_gaps(seed: int, count: int = 2000) -> list[int]:
    source = random.Random(seed)
    gaps = []
    for _ in range(count):
        gap = INTERVAL
        if source.random() < 0.1:
            gap += source.randrange(5, 40)
        gaps.append(gap)
    return gaps


def _shifted_gaps(seed: int = 9) -> list[int]:
    source = random.Random(seed)
    calm = [
        10 + (source.randrange(2, 8) if source.random() < 0.1 else 0)
        for _ in range(1000)
    ]
    rough = [
        10 + (source.randrange(20, 80) if source.random() < 0.3 else 0)
        for _ in range(1000)
    ]
    return calm + rough


@functools.cache
def the_fixed_dial_trades_alarms_for_delay() -> bool:
    """Timeout 15: 212 false alarms, death called in 16. Timeout 60: 0 and 61.

    The fixed detector has one dial and it moves both meters in opposite
    directions: every tick of patience bought is a tick of detection
    delay paid. There is no setting that is good at both ends.
    """
    gaps = _jittery_gaps(7)
    die = sum(gaps[:1000])
    twitchy_alarms, _ = _watch(Fixed(timeout=15), gaps, None)
    patient_alarms, _ = _watch(Fixed(timeout=60), gaps, None)
    _, twitchy_delay = _watch(Fixed(timeout=15), gaps[:1000], die)
    _, patient_delay = _watch(Fixed(timeout=60), gaps[:1000], die)
    return (
        twitchy_alarms == 212
        and patient_alarms == 0
        and twitchy_delay == 16
        and patient_delay == 61
    )


@functools.cache
def the_learned_patience_is_a_number_it_picked() -> bool:
    """On stationary jitter the adaptive detector equals fixed 46, jumpier.

    Mean gap plus four spreads settles near 46, and the death is called
    in 46 ticks to fixed 46's 47. But its false alarm count is 38 to the
    hand-tuned 13, because the learned patience wobbles with its window.
    On a stationary line, learning automates the choice of the number;
    it does not beat the person who knew the number.
    """
    gaps = _jittery_gaps(7)
    die = sum(gaps[:1000])
    learned_alarms, _ = _watch(Adaptive(), gaps, None)
    tuned_alarms, _ = _watch(Fixed(timeout=46), gaps, None)
    _, learned_delay = _watch(Adaptive(), gaps[:1000], die)
    _, tuned_delay = _watch(Fixed(timeout=46), gaps[:1000], die)
    return (
        learned_alarms == 38
        and tuned_alarms == 13
        and abs(learned_delay - tuned_delay) <= 1
    )


@functools.cache
def the_learned_detector_survives_the_regime_change() -> bool:
    """The network roughens mid-watch: fixed 30 alarms 248 times, learned 11.

    This is what the learning is for. The hand-tuned number was right for
    the calm half and wrong for the rough half, and no fixed number is
    right for both. The adaptive detector re-learns the new jitter within
    its window and goes quiet again, 22 times fewer false alarms.
    """
    gaps = _shifted_gaps()
    fixed_alarms, _ = _watch(Fixed(timeout=30), gaps, None)
    learned_alarms, _ = _watch(Adaptive(), gaps, None)
    return fixed_alarms == 248 and learned_alarms == 11


@functools.cache
def patience_stretches_with_the_observed_spread() -> bool:
    """Fed calm gaps the patience sits near 20; fed rough gaps, near 96.

    The detector's number is a summary of what it has seen: mean plus
    four standard deviations. Show it a jitterier line and its patience
    stretches to cover what that line actually does.
    """
    calm, rough = Adaptive(), Adaptive()
    gaps = _shifted_gaps()
    now = 0
    for gap in gaps[:1000]:
        now += gap
        calm.beat(now)
    now = 0
    for gap in gaps[1000:]:
        now += gap
        rough.beat(now)
    return calm.patience() < 25 and rough.patience() > 80


@functools.cache
def summarise() -> dict:
    return {
        "module": "store.heartbeat",
        "the_fixed_dial_trades_alarms_for_delay": (
            the_fixed_dial_trades_alarms_for_delay()
        ),
        "the_learned_patience_is_a_number_it_picked": (
            the_learned_patience_is_a_number_it_picked()
        ),
        "the_learned_detector_survives_the_regime_change": (
            the_learned_detector_survives_the_regime_change()
        ),
        "patience_stretches_with_the_observed_spread": (
            patience_stretches_with_the_observed_spread()
        ),
    }
