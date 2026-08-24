from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass, field

from store.errors import BadFormat, ConfigError

# The hash chain: an audit log where editing the past is arithmetic, not access control.
#
# A plain audit log is honest exactly as long as nobody with write access wants it
# otherwise, since rewriting a line is as easy as writing it. The chain changes the game:
# every entry carries the hash of the previous entry's hash and its own content, so the
# newest hash commits to the entire history, and changing any old line changes every hash
# after it. The tamperer's options shrink to rewriting the whole suffix, which the module
# shows is detectable by anyone holding one saved head hash, sixty four bytes of trust
# against an unbounded log. Every tamper style is built and caught: edit, delete, insert,
# reorder, and the suffix rewrite against a stale head.


def _link(previous: bytes, content: bytes) -> bytes:
    return hashlib.blake2b(previous + content, digest_size=32).digest()


GENESIS = bytes(32)


@dataclass
class Chain:
    """The chained log."""

    entries: list[tuple[bytes, bytes]] = field(default_factory=list)

    def append(self, content: bytes) -> bytes:
        """One entry in, the new head out."""
        if not content:
            raise ConfigError("an empty entry records nothing")
        previous = self.entries[-1][1] if self.entries else GENESIS
        head = _link(previous, content)
        self.entries.append((content, head))
        return head

    @property
    def head(self) -> bytes:
        """The hash that commits to everything."""
        return self.entries[-1][1] if self.entries else GENESIS

    def verify(self, expected_head: bytes | None = None) -> int:
        """Walk the chain: every link recomputed, the head compared if one is offered."""
        previous = GENESIS
        for at, (content, stored) in enumerate(self.entries):
            computed = _link(previous, content)
            if computed != stored:
                raise BadFormat(f"the chain breaks at entry {at}")
            previous = computed
        if expected_head is not None and previous != expected_head:
            raise BadFormat("the head does not match the saved head")
        return len(self.entries)


def _grown(count: int = 200) -> Chain:
    chain = Chain()
    for at in range(count):
        chain.append(f"event-{at:04d}: something happened".encode())
    return chain


@functools.cache
def a_clean_chain_verifies_end_to_end() -> bool:
    """Two hundred entries verify, and the head matches a head saved at the end."""
    chain = _grown()
    saved = chain.head
    return chain.verify(saved) == 200


@functools.cache
def editing_any_line_breaks_the_chain_there() -> bool:
    """Rewrite entry fifty's content and verification fails at exactly entry fifty.

    The break point names the tampered line, which is more than detection: the auditor
    knows where to look, and everything before the break is still trustworthy, because
    the prefix's links never involved the edited bytes.
    """
    chain = _grown()
    _, stored = chain.entries[50]
    chain.entries[50] = (b"event-0050: nothing happened", stored)
    try:
        chain.verify()
        return False
    except BadFormat as complaint:
        return "entry 50" in str(complaint)


@functools.cache
def deletion_insertion_and_reorder_all_break() -> bool:
    """The three other tamper styles, each caught by the walk.

    Deleting shifts every later link onto the wrong predecessor; inserting does the same
    in the other direction; swapping two adjacent entries breaks at the first of them.
    None requires the saved head: the internal consistency alone convicts.
    """
    deleted = _grown()
    del deleted.entries[30]
    inserted = _grown()
    inserted.entries.insert(30, (b"forged", bytes(32)))
    swapped = _grown()
    swapped.entries[30], swapped.entries[31] = swapped.entries[31], swapped.entries[30]
    for chain in (deleted, inserted, swapped):
        try:
            chain.verify()
            return False
        except BadFormat:
            continue
    return True


@functools.cache
def a_full_suffix_rewrite_defeats_the_walk_and_meets_the_saved_head() -> bool:
    """The competent tamperer relinks everything after the edit, and the walk passes.

    Rebuilt links are self-consistent, so internal verification alone is beatable by
    anyone willing to rewrite the suffix, and the module says so plainly. What convicts
    them is the sixty four byte head saved outside their reach: the rebuilt chain's head
    cannot match it, because matching would need a hash collision. The design reduces an
    unbounded integrity problem to the custody of one small value, which is the entire
    idea, and why real systems publish the head somewhere expensive to edit.
    """
    chain = _grown()
    saved = chain.head
    contents = [content for content, _ in chain.entries]
    contents[50] = b"event-0050: nothing happened"
    rebuilt = Chain()
    for content in contents:
        rebuilt.append(content)
    rebuilt.verify()
    try:
        rebuilt.verify(saved)
        return False
    except BadFormat:
        return True


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "clean_chains_verify": a_clean_chain_verifies_end_to_end(),
        "edits_break_at_the_line": editing_any_line_breaks_the_chain_there(),
        "all_tamper_styles_break": deletion_insertion_and_reorder_all_break(),
        "suffix_rewrites_need_the_head": (
            a_full_suffix_rewrite_defeats_the_walk_and_meets_the_saved_head()
        ),
    }
