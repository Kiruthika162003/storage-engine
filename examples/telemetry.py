"""A telemetry pipeline: merge operator counters, sketched cardinality, histogram latencies.

Run with: python -m examples.telemetry
"""

from __future__ import annotations

import random

from store.hll import Sketch
from store.mergeop import Counters
from store.metrics import Histogram

MINUTES = 60
EVENTS_PER_MINUTE = 500


def main() -> int:
    source = random.Random(21)
    counters = Counters()
    visitors = Sketch()
    latencies = Histogram()
    for minute in range(MINUTES):
        for _ in range(EVENTS_PER_MINUTE):
            page = f"page:{source.randrange(30):02d}".encode()
            user = f"user:{source.randrange(4000):05d}".encode()
            counters.add(page, 1)
            visitors.add(user)
            latencies.add(source.lognormvariate(-8.5, 1.0))
        if minute % 15 == 14:
            top = f"page:{0:02d}".encode()
            print(
                f"minute {minute}: page00={counters.get(top)}, "
                f"visitors~{visitors.estimate()}, "
                f"p99={latencies.percentile(99):.6f}s"
            )
    for at in range(3):
        page = f"page:{at:02d}".encode()
        counters.compact(page)
    print(f"counters after compaction: {counters.as_dict()}")
    print(f"sketch: {visitors.as_dict()}")
    print(f"latencies: {latencies.as_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
