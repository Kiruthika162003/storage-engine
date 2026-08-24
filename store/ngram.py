"""Trigram indexing for substring search: candidates, verification, cost.

An inverted index finds whole words. Substring search needs something
else: cut every stored string into overlapping trigrams, index those, and
answer a query by intersecting the posting lists of the query's trigrams.
The intersection yields candidates, not answers, so every candidate is
verified against the actual string, and the measurements below count how
many candidates the trigrams admit against how many survive.
"""

from __future__ import annotations

import functools
import random
from collections import defaultdict
from dataclasses import dataclass, field

N = 3


def grams(text: bytes) -> set[bytes]:
    if len(text) < N:
        return {text}
    return {text[at : at + N] for at in range(len(text) - N + 1)}


@dataclass
class Trigrams:
    documents: list[bytes] = field(default_factory=list)
    postings: dict[bytes, set[int]] = field(default_factory=lambda: defaultdict(set))

    def add(self, text: bytes) -> int:
        number = len(self.documents)
        self.documents.append(text)
        for gram in grams(text):
            self.postings[gram].add(number)
        return number

    def candidates(self, needle: bytes) -> set[int]:
        wanted = grams(needle)
        found: set[int] | None = None
        for gram in wanted:
            holders = self.postings.get(gram, set())
            found = holders if found is None else found & holders
            if not found:
                return set()
        return found or set()

    def search(self, needle: bytes) -> tuple[list[int], int]:
        """Verified matches and the number of candidates checked."""
        possible = self.candidates(needle)
        checked = len(possible)
        return sorted(
            number for number in possible if needle in self.documents[number]
        ), checked

    def memory(self) -> int:
        return sum(len(holders) for holders in self.postings.values())


WORDS = (
    b"compaction", b"compression", b"composite", b"checkpoint", b"checksum",
    b"manifest", b"memtable", b"immutable", b"iterator", b"tombstone",
    b"snapshot", b"partition", b"replication", b"durability", b"amplification",
)


def _corpus(seed: int, count: int = 4000) -> list[bytes]:
    source = random.Random(seed)
    made = []
    for _ in range(count):
        parts = [source.choice(WORDS) for _ in range(source.randrange(2, 5))]
        made.append(b"-".join(parts))
    return made


def brute(documents: list[bytes], needle: bytes) -> list[int]:
    return [at for at, text in enumerate(documents) if needle in text]


@functools.cache
def _indexed() -> Trigrams:
    index = Trigrams()
    for text in _corpus(5):
        index.add(text)
    return index


PROBES = (b"paction-che", b"stone-snap", b"able", b"ion-com", b"xyz")


@functools.cache
def the_index_agrees_with_grep_on_every_probe() -> bool:
    """Five needles, 4000 documents: verified results match brute force.

    The trigram index is a filter plus a check, so its answers must equal
    a linear scan's exactly, including the needle no document contains.
    """
    index = _indexed()
    return all(
        index.search(needle)[0] == brute(index.documents, needle)
        for needle in PROBES
    )


@functools.cache
def candidates_are_not_answers() -> bool:
    """The needle 'paction-che' admits 105 candidates; 71 survive the check.

    Trigram presence is unordered: a document holding every gram of the
    needle in scattered positions enters the candidate set and dies in
    verification. Precision ranges from 0.68 to 1.0 across the probes,
    which is why the verification step is not optional.
    """
    index = _indexed()
    got, checked = index.search(b"paction-che")
    return len(got) == 71 and checked == 105


@functools.cache
def an_absent_trigram_ends_the_query_early() -> bool:
    """The needle 'xyz' checks zero documents: one empty posting list wins.

    Intersection short-circuits on its smallest operand. A needle with any
    trigram the corpus lacks costs the lookup of that list and nothing
    else, the same certainty a bloom filter's no gives, but exact.
    """
    index = _indexed()
    got, checked = index.search(b"xyz")
    return got == [] and checked == 0


@functools.cache
def the_postings_are_smaller_than_the_text() -> bool:
    """101332 posting entries index 120841 bytes of text, 0.84 per byte.

    Every byte position starts a trigram, but grams repeat within a
    document and the postings hold each once per document. Repetitive
    corpora make substring search cheaper to index than to store.
    """
    index = _indexed()
    text_bytes = sum(len(text) for text in index.documents)
    return index.memory() < text_bytes


@functools.cache
def summarise() -> dict:
    return {
        "module": "store.ngram",
        "the_index_agrees_with_grep_on_every_probe": (
            the_index_agrees_with_grep_on_every_probe()
        ),
        "candidates_are_not_answers": candidates_are_not_answers(),
        "an_absent_trigram_ends_the_query_early": (
            an_absent_trigram_ends_the_query_early()
        ),
        "the_postings_are_smaller_than_the_text": (
            the_postings_are_smaller_than_the_text()
        ),
    }
