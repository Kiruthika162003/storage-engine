"""A social feed: newest-first keys, a changefeed fanout, heavy hitter authors.

Run with: python -m examples.feed
"""

from __future__ import annotations

import random

from store.timekey import descending, latest, read_descending
from store.topk import Summary
from store.watch import Feed

AUTHORS = 500
POSTS = 4000


def main() -> int:
    source = random.Random(31)
    feed = Feed(buffer_records=800)
    feed.subscribe("notifier")
    feed.subscribe("search-indexer")
    hot_authors = Summary(capacity=6)
    timeline: list[bytes] = []
    clock = 1_700_000_000_000

    for _ in range(POSTS):
        clock += source.randrange(1, 5000)
        author = int(AUTHORS * source.random() ** 3)
        key = descending(clock) + f":a{author:04d}".encode()
        timeline.append(key)
        feed.publish(key, f"post by {author}".encode())
        hot_authors.note(f"a{author:04d}".encode())
        if source.random() < 0.4:
            feed.poll("notifier", limit=50)

    front = latest(timeline, 5)
    moments = [read_descending(key[:8]) for key in front]
    print(f"latest five moments (newest first): {moments == sorted(moments, reverse=True)}")

    feed.poll("notifier", limit=10**6)
    print(f"notifier lag after catch-up: {feed.lag('notifier')}")
    print(f"search-indexer lag (never polled): {feed.lag('search-indexer')}")

    print(f"hot author candidates: {sorted(hot_authors.candidates())}")
    bound = int(hot_authors.bound)
    print(f"guaranteed hot above {bound}: {hot_authors.certainly_above(bound)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
