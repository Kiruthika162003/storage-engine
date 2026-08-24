"""Tenant quotas: hard caps strand capacity, fair queueing lends it.

A store serving many tenants must divide its capacity. The two honest
designs are hard caps, each tenant owns a slice and unused slices idle,
and work conserving fair sharing, where idle capacity is lent to whoever
is waiting but reclaimed the moment its owner returns. Both are run over
the same demand traces and the served counts are compared.
"""

from __future__ import annotations

import functools
import random
from dataclasses import dataclass

CAPACITY = 100
TENANTS = 4
TICKS = 400


@dataclass
class Demand:
    """Requests per tenant per tick."""

    wants: list[list[int]]

    @classmethod
    def steady(cls, seed: int, heavy: int = 0) -> Demand:
        source = random.Random(seed)
        wants = []
        for _ in range(TICKS):
            row = [source.randrange(10, 20) for _ in range(TENANTS)]
            if heavy:
                row[0] = heavy
            wants.append(row)
        return cls(wants=wants)

    @classmethod
    def bursty(cls, seed: int) -> Demand:
        source = random.Random(seed)
        wants = []
        for tick in range(TICKS):
            row = [source.randrange(5, 15) for _ in range(TENANTS)]
            if (tick // 50) % 2 == 0:
                row[3] = 0
            else:
                row[3] = 80
            wants.append(row)
        return cls(wants=wants)


def hard_caps(demand: Demand) -> list[int]:
    slice_of = CAPACITY // TENANTS
    served = [0] * TENANTS
    for row in demand.wants:
        for tenant, want in enumerate(row):
            served[tenant] += min(want, slice_of)
    return served


def fair_share(demand: Demand) -> list[int]:
    served = [0] * TENANTS
    for row in demand.wants:
        remaining = list(row)
        capacity = CAPACITY
        while capacity > 0 and any(remaining):
            hungry = [tenant for tenant in range(TENANTS) if remaining[tenant] > 0]
            slice_of = max(1, capacity // len(hungry))
            for tenant in hungry:
                granted = min(remaining[tenant], slice_of, capacity)
                remaining[tenant] -= granted
                served[tenant] += granted
                capacity -= granted
                if capacity == 0:
                    break
    return served


@functools.cache
def caps_serve_sixty_percent_where_sharing_serves_ninety_four() -> bool:
    """Bursty demand of 27420: caps serve 16420, fair sharing 25768.

    The guess was that fair sharing would serve everything. It cannot: on
    burst ticks the four tenants together want about 110 against a store
    of 100, and the 1652 requests fair sharing leaves unserved are exactly
    the sum of that over-capacity excess, demand no scheduler can invent
    capacity for. Hard caps lose far more, and twice over: the burster is
    clipped to its 25 slice during bursts, and its idle slice strands
    during the quiet phases.
    """
    demand = Demand.bursty(3)
    capped = sum(hard_caps(demand))
    shared = sum(fair_share(demand))
    wanted = sum(sum(row) for row in demand.wants)
    excess = sum(max(0, sum(row) - CAPACITY) for row in demand.wants)
    return wanted - shared == excess and capped < wanted * 0.61


@functools.cache
def fair_sharing_still_isolates() -> bool:
    """A tenant demanding 300 a tick cannot push a modest tenant below 25.

    Work conservation sounds like a loophole: if tenant zero asks for
    everything, does anyone starve? No: equal shares are recomputed every
    round, so the greedy tenant absorbs only what the modest ones decline.
    Each modest tenant gets every unit it asked for.
    """
    demand = Demand.steady(5, heavy=300)
    served = fair_share(demand)
    wanted = [sum(row[tenant] for row in demand.wants) for tenant in range(TENANTS)]
    modest_whole = all(served[tenant] == wanted[tenant] for tenant in range(1, TENANTS))
    return modest_whole and served[0] < wanted[0]


@functools.cache
def the_greedy_tenant_gets_the_slack_not_the_slice() -> bool:
    """The greedy tenant is served 56.8 a tick: its 25 plus what the rest decline.

    Modest tenants want 10 to 19 each, mean 14.5, leaving about 30 spare.
    The greedy tenant's haul is its own quarter plus exactly that slack,
    22728 over 400 ticks, 56.8 a tick, nowhere near its 300 demand.
    """
    demand = Demand.steady(5, heavy=300)
    served = fair_share(demand)
    per_tick = served[0] / TICKS
    return 53 < per_tick < 57


@functools.cache
def caps_and_shares_agree_when_nobody_bursts() -> bool:
    """Steady demand under 25 a tick: both designs serve every request.

    When every tenant fits its slice the designs are indistinguishable.
    The choice only matters at the edges: idle tenants and greedy ones.
    """
    demand = Demand.steady(7)
    return hard_caps(demand) == fair_share(demand)


@functools.cache
def summarise() -> dict:
    return {
        "module": "store.fairshare",
        "caps_serve_sixty_percent_where_sharing_serves_ninety_four": (
            caps_serve_sixty_percent_where_sharing_serves_ninety_four()
        ),
        "fair_sharing_still_isolates": fair_sharing_still_isolates(),
        "the_greedy_tenant_gets_the_slack_not_the_slice": (
            the_greedy_tenant_gets_the_slack_not_the_slice()
        ),
        "caps_and_shares_agree_when_nobody_bursts": (
            caps_and_shares_agree_when_nobody_bursts()
        ),
    }
