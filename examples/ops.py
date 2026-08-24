"""An operations story: one outage seen by every defence at once.

Run with: python -m examples.ops
"""

from __future__ import annotations

from store.circuit import Breaker, Dependency
from store.errorbudget import RATE, WINDOW, Month, fast, slow
from store.retry import peak_after_recovery, run_outage


def main() -> int:
    dependency = Dependency(healthy=True)
    breaker = Breaker(dependency=dependency, threshold=5, cooldown=20)
    failures_per_tick = []
    for tick in range(600):
        dependency.healthy = not (200 <= tick < 500)
        breaker.tick()
        failures = sum(0 if breaker.call() else 1 for _ in range(10))
        failures_per_tick.append(failures)
    print(
        f"outage of 300 ticks: {breaker.fast_failures} calls answered open "
        f"in 1 tick instead of {dependency.timeout_cost}"
    )
    print(f"the dependency itself was bothered {breaker.probes} probe(s) while open")

    padding = [0] * (WINDOW - len(failures_per_tick))
    month = Month(failed=[f * RATE // 10 for f in failures_per_tick] + padding)
    fast_pages = fast().replay(month)
    slow_pages = slow().replay(month)
    print(f"fast burn pages at tick {fast_pages[0]}, outage began at 200")
    print(
        f"slow burn pages from tick {slow_pages[0]} to {slow_pages[-1]}, "
        f"long after recovery at 500"
    )

    for discipline in ("fixed", "jittered"):
        service, finished = run_outage(discipline)
        print(
            f"{discipline} retries: post-recovery peak "
            f"{peak_after_recovery(service)} against capacity 100, "
            f"all clients served by tick {finished}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
