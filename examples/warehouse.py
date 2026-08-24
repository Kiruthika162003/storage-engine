"""An analytics warehouse: bulk load, zone maps, dictionary codes, a join, a group-by.

Run with: python -m examples.warehouse
"""

from __future__ import annotations

import random

from store.dictionary import Encoded
from store.groupagg import Meter as AggMeter
from store.groupagg import stream_aggregate
from store.joins import Meter as JoinMeter
from store.joins import merge_join
from store.zonemap import Mapped

ROWS = 30000


def main() -> int:
    source = random.Random(41)

    amounts = []
    cities = Encoded()
    clock = 0
    orders = []
    for at in range(ROWS):
        clock += source.randrange(1, 5)
        amount = source.randrange(1, 900)
        amounts.append(clock)
        cities.append(f"city-{source.randrange(40):03d}".encode())
        orders.append((at % 3000, amount))
    orders.sort()

    mapped = Mapped.build(amounts, block_size=500)
    low, high = amounts[ROWS // 2], amounts[ROWS // 2 + 400]
    found = mapped.query(low, high)
    print(f"time range query: {len(found)} rows, skipped {mapped.blocks_skipped} blocks")

    wanted = cities.scan_equal(b"city-007")
    print(
        f"dictionary filter: {len(wanted)} rows for city-007, "
        f"dictionary holds {len(cities.values)} values"
    )

    customers = [(at, f"customer-{at:04d}".encode()) for at in range(3000)]
    join_meter = JoinMeter()
    joined = merge_join(customers, [(c, str(a).encode()) for c, a in orders], join_meter)
    print(f"join: {len(joined)} rows, window held {join_meter.held_rows} rows")

    agg_meter = AggMeter()
    totals = stream_aggregate(orders, agg_meter)
    top = max(totals, key=lambda pair: pair[1])
    print(
        f"group-by: {len(totals)} customers, top spent {top[1]}, "
        f"held {agg_meter.held_groups} group"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
