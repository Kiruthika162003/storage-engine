"""A search box over the store: inverted index, predicates, heavy hitter queries.

Run with: python -m examples.search
"""

from __future__ import annotations

import random

from store.inverted import Index, grep_and
from store.predicate import Both, Compare, Scanner
from store.topk import Summary
from store.zonemap import Mapped

DOCUMENTS = (
    b"the write ahead log records every change",
    b"compaction merges sorted files into fewer files",
    b"the bloom filter answers no with certainty",
    b"a crash loses the memtable and keeps the files",
    b"the manifest records which files count",
    b"every claim in this package is a measurement",
    b"the cache holds the hot blocks in memory",
    b"recovery replays the log into a fresh memtable",
)


def main() -> int:
    source = random.Random(51)
    index = Index()
    for number, text in enumerate(DOCUMENTS):
        index.add(number, text)

    found = index.search_and([b"the", b"files"])
    truth = grep_and(index.documents, [b"the", b"files"])
    print(f"AND(the, files) -> docs {found}, agrees with grep: {found == truth}")

    phrase = index.search_phrase(b"the log")
    print(f"phrase 'the log' -> docs {phrase}")

    sizes = [len(text) for text in DOCUMENTS] * 500
    mapped = Mapped.build(sizes, block_size=100)
    scanner = Scanner(mapped=mapped)
    predicate = Both(left=Compare(op=">=", value=40), right=Compare(op="<", value=46))
    matches = scanner.scan_pushed(predicate)
    print(
        f"length predicate matched {len(matches)} rows, "
        f"skipped {mapped.blocks_skipped} blocks"
    )

    hot = Summary(capacity=4)
    for _ in range(2000):
        if source.random() < 0.7:
            hot.note(b"the files")
        else:
            hot.note(f"query-{source.randrange(200):03d}".encode())
    top = hot.certainly_above(int(hot.bound))
    print(f"provably hot queries: {sorted(top)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
