from __future__ import annotations

import pytest

from store.fairshare import (
    CAPACITY,
    TENANTS,
    TICKS,
    Demand,
    caps_and_shares_agree_when_nobody_bursts,
    caps_serve_sixty_percent_where_sharing_serves_ninety_four,
    fair_share,
    fair_sharing_still_isolates,
    hard_caps,
    summarise,
    the_greedy_tenant_gets_the_slack_not_the_slice,
)


class TestDemand:
    def test_steady_is_deterministic(self):
        assert Demand.steady(3).wants == Demand.steady(3).wants

    def test_steady_has_a_row_per_tick(self):
        demand = Demand.steady(3)
        assert len(demand.wants) == TICKS
        assert all(len(row) == TENANTS for row in demand.wants)

    def test_heavy_overrides_tenant_zero(self):
        demand = Demand.steady(3, heavy=99)
        assert all(row[0] == 99 for row in demand.wants)

    def test_bursty_alternates_phases(self):
        demand = Demand.bursty(3)
        assert demand.wants[0][3] == 0
        assert demand.wants[50][3] == 80


class TestSchedulers:
    def test_hard_caps_never_exceed_the_slice(self):
        demand = Demand.steady(3, heavy=500)
        served = hard_caps(demand)
        assert served[0] == TICKS * (CAPACITY // TENANTS)

    def test_fair_share_never_exceeds_capacity(self):
        demand = Demand.steady(3, heavy=500)
        assert sum(fair_share(demand)) <= TICKS * CAPACITY

    def test_fair_share_never_serves_more_than_wanted(self):
        demand = Demand.steady(3)
        served = fair_share(demand)
        wanted = [
            sum(row[tenant] for row in demand.wants) for tenant in range(TENANTS)
        ]
        assert all(served[t] <= wanted[t] for t in range(TENANTS))

    def test_fair_share_serves_everything_under_capacity(self):
        demand = Demand.steady(11)
        wanted = sum(sum(row) for row in demand.wants)
        assert sum(fair_share(demand)) == wanted

    def test_an_idle_tenant_is_served_nothing(self):
        demand = Demand(wants=[[10, 0, 10, 10]] * 5)
        assert fair_share(demand)[1] == 0


class TestClaims:
    @pytest.mark.parametrize(
        "claim",
        [
            caps_serve_sixty_percent_where_sharing_serves_ninety_four,
            fair_sharing_still_isolates,
            the_greedy_tenant_gets_the_slack_not_the_slice,
            caps_and_shares_agree_when_nobody_bursts,
        ],
    )
    def test_claim_holds(self, claim):
        assert claim() is True

    def test_summary_is_all_true(self):
        told = summarise()
        assert all(value for name, value in told.items() if name != "module")
