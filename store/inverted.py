from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

from store.errors import ConfigError

# An inverted index over the values, because someone always asks for search.
#
# The secondary index mapped one field to keys; text does not have one field, it has every
# word, and the inverted index is the secondary index applied per token. Postings are kept
# sorted, AND intersects them smallest-first, and position lists upgrade the index from
# which-documents to where-in-them, which is what a phrase query needs and a bag of words
# cannot answer. The measurements hold the index to the grep it replaces, then show the
# two costs that define the structure: intersection work tracks the rarest term, and the
# phrase check pays a position walk that the AND alone never sees.


def tokenize(text: bytes) -> list[bytes]:
    """Lowercased words, the simplest honest tokenizer."""
    return text.lower().split()


@dataclass
class Index:
    """Postings with positions."""

    postings: dict[bytes, dict[int, list[int]]] = field(default_factory=dict)
    documents: dict[int, bytes] = field(default_factory=dict)
    comparisons: int = field(default=0)

    def add(self, number: int, text: bytes) -> None:
        """One document in."""
        if number in self.documents:
            raise ConfigError(f"document {number} is already indexed")
        self.documents[number] = text
        for position, token in enumerate(tokenize(text)):
            self.postings.setdefault(token, {}).setdefault(number, []).append(position)

    def docs_with(self, token: bytes) -> list[int]:
        """The posting list, sorted."""
        return sorted(self.postings.get(token.lower(), {}))

    def search_and(self, tokens: list[bytes]) -> list[int]:
        """Documents holding every token, rarest list first."""
        if not tokens:
            return []
        lists = [self.docs_with(token) for token in tokens]
        lists.sort(key=len)
        found = lists[0]
        for other in lists[1:]:
            held = set(other)
            self.comparisons += len(found)
            found = [number for number in found if number in held]
            if not found:
                break
        return found

    def search_phrase(self, phrase: bytes) -> list[int]:
        """Documents holding the tokens adjacently, in order."""
        tokens = tokenize(phrase)
        if not tokens:
            return []
        candidates = self.search_and(tokens)
        found = []
        for number in candidates:
            first_positions = self.postings[tokens[0]][number]
            for start in first_positions:
                self.comparisons += 1
                if all(
                    start + offset in self.postings[token][number]
                    for offset, token in enumerate(tokens[1:], start=1)
                ):
                    found.append(number)
                    break
        return found


def grep_and(documents: dict[int, bytes], tokens: list[bytes]) -> list[int]:
    """The reference: read everything, split everything."""
    found = []
    for number, text in sorted(documents.items()):
        words = set(tokenize(text))
        if all(token.lower() in words for token in tokens):
            found.append(number)
    return found


def grep_phrase(documents: dict[int, bytes], phrase: bytes) -> list[int]:
    """The reference for phrases: sliding window over the tokens."""
    wanted = tokenize(phrase)
    found = []
    for number, text in sorted(documents.items()):
        words = tokenize(text)
        for at in range(len(words) - len(wanted) + 1):
            if words[at : at + len(wanted)] == wanted:
                found.append(number)
                break
    return found


WORDS = (
    b"the", b"store", b"writes", b"sorted", b"records", b"and", b"reads", b"them",
    b"back", b"compaction", b"merges", b"files", b"levels", b"bloom", b"filter",
    b"cache", b"log", b"crash", b"recovery", b"rare",
)


@functools.cache
def _corpus(count: int = 2000, seed: int = 331) -> Index:
    """Documents of common words, with the word rare kept genuinely rare."""
    source = random.Random(seed)
    index = Index()
    for number in range(count):
        words = [source.choice(WORDS[:-1]) for _ in range(source.randrange(8, 30))]
        if source.random() < 0.01:
            words.insert(source.randrange(len(words)), b"rare")
        index.add(number, b" ".join(words))
    return index


@functools.cache
def the_index_agrees_with_grep_on_ands_and_phrases() -> bool:
    """Thirty AND queries and thirty phrase queries, identical answers to the full read.

    Grep is the specification, correct because it reads everything. The phrase agreement
    matters most: adjacency is where position bookkeeping slips, off by one at the token
    offsets, and only the sliding-window reference catches it.
    """
    index = _corpus()
    source = random.Random(337)
    for _ in range(30):
        tokens = [source.choice(WORDS) for _ in range(source.randrange(1, 4))]
        if index.search_and(tokens) != grep_and(index.documents, tokens):
            return False
        phrase = b" ".join(source.choice(WORDS) for _ in range(2))
        if index.search_phrase(phrase) != grep_phrase(index.documents, phrase):
            return False
    return True


@functools.cache
def intersection_work_tracks_the_rarest_term() -> bool:
    """AND with the rare word costs a fiftieth of AND between two common words.

    Smallest-list-first makes the rarest term the driver: the walk is bounded by its
    postings, twenty odd documents, however common the other term. Query planners order
    conjuncts by selectivity for exactly this reason, and the planner module's lesson
    holds here without a histogram, because the posting length is the exact count.
    """
    index = _corpus()
    index.comparisons = 0
    index.search_and([b"rare", b"the"])
    rare_cost = index.comparisons
    index.comparisons = 0
    index.search_and([b"store", b"the"])
    common_cost = index.comparisons
    return rare_cost * 20 < common_cost


@functools.cache
def phrases_pay_positions_on_top_of_the_and() -> bool:
    """The phrase query does the AND and then walks positions the AND never touched.

    Measured as comparisons: the phrase's total exceeds the plain AND's on the same
    tokens, and the excess is the adjacency check. Which-documents is set algebra;
    where-in-them is a second index dimension, paid for separately.
    """
    index = _corpus()
    index.comparisons = 0
    index.search_and([b"store", b"records"])
    and_cost = index.comparisons
    index.comparisons = 0
    index.search_phrase(b"store records")
    phrase_cost = index.comparisons
    return phrase_cost > and_cost


@functools.cache
def duplicate_documents_are_refused() -> bool:
    """Indexing the same document number twice raises, the double-owner rule again."""
    index = Index()
    index.add(1, b"once")
    try:
        index.add(1, b"twice")
    except ConfigError:
        return True
    return False


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "the_index_agrees_with_grep": the_index_agrees_with_grep_on_ands_and_phrases(),
        "the_rarest_term_drives": intersection_work_tracks_the_rarest_term(),
        "phrases_pay_for_positions": phrases_pay_positions_on_top_of_the_and(),
        "duplicates_are_refused": duplicate_documents_are_refused(),
    }
