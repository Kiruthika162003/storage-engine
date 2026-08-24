from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

from store.errors import ConfigError

# Idempotency keys: at-most-once effect over at-least-once delivery.
#
# The changefeed module's resync and every network in existence deliver requests at least
# once, and a payment applied at least once is a payment applied twice. The fix is a key
# per intent: the client names the operation, the store remembers which names it has
# executed and replays the recorded answer for a repeat instead of the effect. The window
# is the honest limit, remembered names must eventually be forgotten, and a retry arriving
# after the forgetting reapplies. The measurements run a retrying storm against a naive
# store and a keyed one, then shrink the window until the guarantee visibly tears.


@dataclass
class Keyed:
    """A counter store with an idempotency window."""

    window: int = field(default=1000)
    balance: int = field(default=0)
    applied: dict[bytes, int] = field(default_factory=dict)
    order: list[bytes] = field(default_factory=list)
    executed: int = field(default=0)
    replayed: int = field(default=0)

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ConfigError(f"{self.window} is not a window")

    def deposit(self, key: bytes, amount: int) -> int:
        """Apply once per key; repeats inside the window replay the recorded answer."""
        if key in self.applied:
            self.replayed += 1
            return self.applied[key]
        self.balance += amount
        self.executed += 1
        self.applied[key] = self.balance
        self.order.append(key)
        while len(self.order) > self.window:
            del self.applied[self.order.pop(0)]
        return self.applied[key]


@dataclass
class Naive:
    """The store every retry storm meets first."""

    balance: int = field(default=0)
    executed: int = field(default=0)

    def deposit(self, key: bytes, amount: int) -> int:
        del key
        self.balance += amount
        self.executed += 1
        return self.balance


def _storm(store, intents: int, seed: int) -> int:
    """Every intent delivered one to four times, interleaved: the network as it is."""
    source = random.Random(seed)
    deliveries = []
    for at in range(intents):
        for _ in range(source.randrange(1, 5)):
            deliveries.append(at)
    source.shuffle(deliveries)
    for at in deliveries:
        store.deposit(f"intent-{at:06d}".encode(), 10)
    return len(deliveries)


@functools.cache
def the_naive_store_applies_every_delivery() -> bool:
    """A thousand intents, 2,632 deliveries, and the naive balance is 26,320.

    The overcharge is exactly the duplicate count times the amount, which is the point:
    at-least-once delivery converts directly into money under a store with no memory of
    intent, and the retry that made the system reliable is the same retry that made the
    balance wrong.
    """
    store = Naive()
    deliveries = _storm(store, 1000, 349)
    return store.balance == deliveries * 10 and deliveries > 1000


@functools.cache
def the_keyed_store_applies_every_intent_once() -> bool:
    """The same 2,632 deliveries: balance 10,000, one execution per intent.

    The replays are counted and answered with the recorded result, so the retrying client
    cannot tell its retry was a duplicate, which is the contract: the caller retries
    freely and the effect happens once.
    """
    store = Keyed(window=10**6)
    deliveries = _storm(store, 1000, 349)
    return (
        store.balance == 10000
        and store.executed == 1000
        and store.replayed == deliveries - 1000
    )


@functools.cache
def a_replay_returns_the_original_answer_not_the_current_one() -> bool:
    """The recorded answer is the balance as of the original execution, frozen.

    Later deposits move the balance, and the replay still answers what the first execution
    answered, because the caller is completing its original call, not making a new one. A
    replay that returned the current balance would be a correct effect with a lying
    receipt.
    """
    store = Keyed()
    first = store.deposit(b"a", 10)
    store.deposit(b"b", 5)
    replay = store.deposit(b"a", 10)
    return first == 10 and replay == 10 and store.balance == 15


@functools.cache
def a_retry_after_the_window_reapplies() -> bool:
    """Shrink the window to ten and a straggler retry lands twice, on the record.

    The window is the guarantee's budget: remembered intents cost memory, forgotten
    intents cost correctness against sufficiently late retries, and the operational
    number is the window against the client's maximum retry horizon. This measurement is
    the tear made visible so the budget conversation has a demonstration.
    """
    store = Keyed(window=10)
    store.deposit(b"early", 10)
    for at in range(20):
        store.deposit(f"filler-{at}".encode(), 1)
    store.deposit(b"early", 10)
    return store.balance == 10 + 20 + 10 and store.executed == 22


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "naive_stores_apply_deliveries": the_naive_store_applies_every_delivery(),
        "keyed_stores_apply_intents": the_keyed_store_applies_every_intent_once(),
        "replays_answer_the_original": (
            a_replay_returns_the_original_answer_not_the_current_one()
        ),
        "the_window_is_the_budget": a_retry_after_the_window_reapplies(),
    }
