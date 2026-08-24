"""A graph in a sorted store: edge keys, fan-out walks, and the supernode.

Edges live as composite keys, out:src:dst and in:dst:src, so a vertex's
neighbours are one contiguous range scan in either direction. The
measurements count range scans and keys touched for two-hop walks, then
build the vertex every real graph grows eventually, the supernode, and
weigh what it does to every query that brushes against it.
"""

from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

PEOPLE = 2000
FOLLOWS = 30


@dataclass
class Graph:
    edges: dict[bytes, bytes] = field(default_factory=dict)
    scans: int = 0
    touched: int = 0

    def link(self, src: int, dst: int) -> None:
        self.edges[f"out:{src:05d}:{dst:05d}".encode()] = b""
        self.edges[f"in:{dst:05d}:{src:05d}".encode()] = b""

    def _scan(self, prefix: bytes) -> list[int]:
        self.scans += 1
        found = []
        for key in self.edges:
            if key.startswith(prefix):
                found.append(int(key.rsplit(b":", 1)[1]))
        self.touched += len(found)
        return found

    def out(self, src: int) -> list[int]:
        return self._scan(f"out:{src:05d}:".encode())

    def into(self, dst: int) -> list[int]:
        return self._scan(f"in:{dst:05d}:".encode())

    def into_without_index(self, dst: int) -> list[int]:
        self.scans += 1
        found = []
        for key in self.edges:
            if not key.startswith(b"out:"):
                continue
            self.touched += 1
            if int(key.rsplit(b":", 1)[1]) == dst:
                found.append(int(key.split(b":")[1]))
        return found

    def two_hop(self, src: int) -> set[int]:
        found = set()
        for middle in self.out(src):
            found.update(self.out(middle))
        found.discard(src)
        return found

    def reset_meters(self) -> None:
        self.scans = 0
        self.touched = 0


def _social(seed: int, supernode: bool) -> Graph:
    source = random.Random(seed)
    graph = Graph()
    for person in range(PEOPLE):
        for _ in range(FOLLOWS):
            other = source.randrange(PEOPLE)
            if other != person:
                graph.link(person, other)
    if supernode:
        for person in range(1, PEOPLE):
            graph.link(person, 0)
            graph.link(0, person)
    return graph


@functools.cache
def neighbours_cost_one_scan_each_way() -> bool:
    """Following and followers are each one range scan touching 31 keys.

    The composite key puts a vertex's out-edges next to each other and,
    thanks to the doubled write, its in-edges too. Both questions cost a
    scan proportional to the answer, nothing more.
    """
    graph = _social(3, supernode=False)
    graph.reset_meters()
    following = graph.out(7)
    followers = graph.into(7)
    return graph.scans == 2 and graph.touched == len(following) + len(followers)


@functools.cache
def the_reverse_question_without_its_index_reads_the_world() -> bool:
    """Who follows 7, answered from out-edges alone: 59565 keys for 31.

    Dropping the in: copy halves the storage, 119130 entries to 59565,
    and turns one reverse lookup into a full scan touching every edge in
    the store. The doubled write is not redundancy, it is the index for
    the question the forward key cannot answer.
    """
    graph = _social(3, supernode=False)
    stored = len(graph.edges)
    graph.reset_meters()
    slow = graph.into_without_index(7)
    fast_cost = len(slow)
    return stored == 119130 and graph.touched == stored // 2 and fast_cost == 31


@functools.cache
def a_two_hop_walk_repeats_a_fifth_of_its_steps() -> bool:
    """Friends of friends: 30 scans touch 892 edges reaching 691 people.

    The walk visits 892 endpoints but only 691 are distinct: in a random
    graph of this density a fifth of two-hop paths collide. The touched
    to reached gap is the walk's built-in duplication, and it grows with
    clustering.
    """
    graph = _social(3, supernode=False)
    graph.reset_meters()
    reached = graph.two_hop(7)
    return graph.scans == 30 and graph.touched == 892 and len(reached) == 691


@functools.cache
def one_celebrity_taxes_every_walk_through_them() -> bool:
    """The same walk with a celebrity in it: 2921 touched, 1999 reached.

    Vertex 0 follows everyone back, so any two-hop that passes through
    them sweeps their 2029 out-edges: 3.3 times the touches for a result
    that is mostly the celebrity's audience, not the walker's circle.
    Real systems cap, sample or precompute exactly this vertex.
    """
    plain = _social(3, supernode=False)
    plain.reset_meters()
    plain.two_hop(7)
    heavy = _social(3, supernode=True)
    heavy.reset_meters()
    reached = heavy.two_hop(7)
    return heavy.touched == 2921 and heavy.touched > plain.touched * 3 and (
        len(reached) == 1999
    )


@functools.cache
def summarise() -> dict:
    return {
        "module": "store.graphadj",
        "neighbours_cost_one_scan_each_way": neighbours_cost_one_scan_each_way(),
        "the_reverse_question_without_its_index_reads_the_world": (
            the_reverse_question_without_its_index_reads_the_world()
        ),
        "a_two_hop_walk_repeats_a_fifth_of_its_steps": (
            a_two_hop_walk_repeats_a_fifth_of_its_steps()
        ),
        "one_celebrity_taxes_every_walk_through_them": (
            one_celebrity_taxes_every_walk_through_them()
        ),
    }
