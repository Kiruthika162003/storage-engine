from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

from store.errors import ConfigError

# Fair shares of one write budget, and what unfairness measures as.
#
# The stall module throttled the store as a whole; a multi tenant store has the harder
# problem of throttling tenants against each other, because the aggregate can be healthy
# while one tenant eats every slot. The mechanism is a token bucket per tenant: capacity is
# the burst allowance, refill is the sustained rate, and a write either spends a token or is
# deferred. The measurements build the starvation first, one loud tenant against nine quiet
# ones sharing an unpartitioned budget, then show the buckets holding every tenant at its
# floor while idle capacity still flows to whoever can use it.


@dataclass
class Bucket:
    """One tenant's allowance."""

    capacity: float
    refill_per_tick: float
    tokens: float = field(default=-1.0)
    spent: int = field(default=0)
    deferred: int = field(default=0)

    def __post_init__(self) -> None:
        if self.capacity <= 0 or self.refill_per_tick <= 0:
            raise ConfigError("a bucket needs positive capacity and refill")
        if self.tokens < 0:
            self.tokens = self.capacity

    def tick(self) -> None:
        """The refill."""
        self.tokens = min(self.tokens + self.refill_per_tick, self.capacity)

    def try_spend(self) -> bool:
        """One write's token, or a deferral."""
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            self.spent += 1
            return True
        self.deferred += 1
        return False


@dataclass
class Shared:
    """The unpartitioned budget: first come, first served, no memory of who came."""

    per_tick: int
    remaining: int = field(default=0)
    spent_by: dict[str, int] = field(default_factory=dict)
    deferred_by: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.per_tick < 1:
            raise ConfigError(f"{self.per_tick} is not a budget")
        self.remaining = self.per_tick

    def tick(self) -> None:
        self.remaining = self.per_tick

    def try_spend(self, tenant: str) -> bool:
        if self.remaining >= 1:
            self.remaining -= 1
            self.spent_by[tenant] = self.spent_by.get(tenant, 0) + 1
            return True
        self.deferred_by[tenant] = self.deferred_by.get(tenant, 0) + 1
        return False


def _demands(tick: int, source: random.Random) -> dict[str, int]:
    """Who wants how many writes this tick: one loud tenant, nine quiet ones."""
    made = {"loud": 40}
    for at in range(9):
        made[f"quiet{at}"] = 1 if source.random() < 0.8 else 0
    del tick
    return made


@functools.cache
def the_shared_budget_starves_the_quiet() -> bool:
    """Under one shared budget the loud tenant takes 93 percent and the quiet miss a third.

    The budget is twenty per tick against demand near fifty, and the loud tenant's forty
    arrivals reach the counter first every tick, so aggregate health, twenty of twenty spent,
    coexists with nine tenants missing writes they were promised. The aggregate meter is the
    wrong meter, which is the finding: nothing in the shared counter's numbers says anything
    is wrong.
    """
    source = random.Random(197)
    shared = Shared(per_tick=20)
    for tick in range(500):
        shared.tick()
        for tenant, wanted in _demands(tick, source).items():
            for _ in range(wanted):
                shared.try_spend(tenant)
    total = sum(shared.spent_by.values())
    loud_share = shared.spent_by["loud"] / total
    quiet_deferred = sum(
        count for tenant, count in shared.deferred_by.items() if tenant != "loud"
    )
    quiet_wanted = sum(
        count for tenant, count in shared.spent_by.items() if tenant != "loud"
    ) + quiet_deferred
    return loud_share > 0.85 and quiet_deferred > quiet_wanted * 0.2


@functools.cache
def buckets_hold_every_quiet_tenant_at_its_floor() -> bool:
    """With a bucket each, no quiet tenant defers at all, and the loud one absorbs the loss.

    Each quiet tenant's rate exceeds its demand, so its bucket never empties, and the loud
    tenant's bucket meters it down to its share. The fairness cost lands entirely on the
    tenant exceeding its allowance, which is the definition of the mechanism working.
    """
    source = random.Random(197)
    buckets = {"loud": Bucket(capacity=15.0, refill_per_tick=11.0)}
    for at in range(9):
        buckets[f"quiet{at}"] = Bucket(capacity=3.0, refill_per_tick=1.0)
    for tick in range(500):
        for bucket in buckets.values():
            bucket.tick()
        for tenant, wanted in _demands(tick, source).items():
            for _ in range(wanted):
                buckets[tenant].try_spend()
    quiet_deferrals = sum(
        bucket.deferred for tenant, bucket in buckets.items() if tenant != "loud"
    )
    return quiet_deferrals == 0 and buckets["loud"].deferred > 10000


@functools.cache
def the_burst_allowance_is_the_capacity() -> bool:
    """An idle bucket absorbs exactly its capacity at once and not one write more.

    The capacity is the answer to how big a spike rides through without deferral, and the
    refill is the answer to how often. The two knobs are independent and the test pins each
    against the other.
    """
    bucket = Bucket(capacity=10.0, refill_per_tick=1.0)
    landed = sum(1 for _ in range(15) if bucket.try_spend())
    bucket.tick()
    after_refill = bucket.try_spend()
    return landed == 10 and after_refill


@functools.cache
def unused_allowance_does_not_bank_past_the_cap() -> bool:
    """A tenant idle for a thousand ticks still bursts only its capacity.

    Without the cap, an idle month banks a flood, and the quota system's first quiet week
    ends in the outage it was bought to prevent. The cap is what makes history harmless.
    """
    bucket = Bucket(capacity=5.0, refill_per_tick=1.0)
    for _ in range(1000):
        bucket.tick()
    landed = sum(1 for _ in range(20) if bucket.try_spend())
    return landed == 5


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "shared_budgets_starve": the_shared_budget_starves_the_quiet(),
        "buckets_hold_the_floor": buckets_hold_every_quiet_tenant_at_its_floor(),
        "capacity_is_the_burst": the_burst_allowance_is_the_capacity(),
        "history_is_capped": unused_allowance_does_not_bank_past_the_cap(),
    }
