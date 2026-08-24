from __future__ import annotations

import functools
from dataclasses import dataclass, field

from store.errors import ConfigError, Conflict

# Fencing tokens: the zombie writer, stopped at the door.
#
# A process pauses, its lease expires, a successor takes over, and then the original wakes
# and keeps writing, a zombie with a stale sense of ownership. Wall-clock leases cannot
# stop it, because the zombie's clock says it still owns the world. The fence is the fix
# that needs no clock: every ownership grant carries a token from a counter that only
# climbs, every write carries its writer's token, and the store refuses any token below
# the highest it has seen. The zombie is not detected, it is simply outranked, and the
# module builds the full pause-expire-succeed-wake story to show the refusal landing
# exactly where the wall clock version corrupts.


@dataclass
class Fenced:
    """A store that admits writes by token rank."""

    highest_seen: int = field(default=0)
    counter: int = field(default=0)
    data: dict[bytes, tuple[bytes, int]] = field(default_factory=dict)
    refusals: int = field(default=0)

    def grant(self) -> int:
        """A new ownership token, strictly above every predecessor."""
        self.counter += 1
        return self.counter

    def write(self, token: int, key: bytes, value: bytes) -> None:
        """A write under a token: refused if any higher token has ever written."""
        if token < 1:
            raise ConfigError(f"{token} is not a token")
        if token < self.highest_seen:
            self.refusals += 1
            raise Conflict(f"token {token} is outranked by {self.highest_seen}")
        self.highest_seen = token
        self.data[key] = (value, token)

    def read(self, key: bytes) -> bytes | None:
        held = self.data.get(key)
        return held[0] if held else None


@dataclass
class Unfenced:
    """The wall clock version: writes admitted on the writer's say-so."""

    data: dict[bytes, bytes] = field(default_factory=dict)

    def write(self, believes_owner: bool, key: bytes, value: bytes) -> None:
        if believes_owner:
            self.data[key] = value

    def read(self, key: bytes) -> bytes | None:
        return self.data.get(key)


@functools.cache
def the_zombie_corrupts_the_unfenced_store() -> bool:
    """The woken zombie's write lands over the successor's, on nothing but its belief.

    The pause, the expiry and the succession all happened; the zombie missed all three,
    its clock still says owner, and the unfenced store has no grounds to disagree. The
    successor's value is gone and nothing anywhere recorded a conflict.
    """
    store = Unfenced()
    store.write(True, b"config", b"zombie-v1")
    store.write(True, b"config", b"successor-v2")
    store.write(True, b"config", b"zombie-v1-again")
    return store.read(b"config") == b"zombie-v1-again"


@functools.cache
def the_fence_outranks_the_zombie() -> bool:
    """The same story with tokens: the zombie's write is refused, the successor's stands.

    The zombie holds token 1, the successor wrote with token 2, and the store's only rule,
    never accept a token below the highest seen, does the whole job. No clock, no
    detection, no message to the zombie, which may not even be reachable: the refusal is
    local and total.
    """
    store = Fenced()
    zombie = store.grant()
    store.write(zombie, b"config", b"zombie-v1")
    successor = store.grant()
    store.write(successor, b"config", b"successor-v2")
    try:
        store.write(zombie, b"config", b"zombie-v1-again")
        return False
    except Conflict:
        pass
    return store.read(b"config") == b"successor-v2" and store.refusals == 1


@functools.cache
def the_fence_binds_only_after_the_successor_writes() -> bool:
    """Before the successor's first write, the zombie's token still passes.

    The store ranks tokens it has seen, and a granted-but-unused token protects nothing:
    the zombie writes happily after the succession, up until the successor's first write
    raises the bar. The gap is real, and closing it means the successor's first act must
    be a write, a fence-establishing no-op if nothing else, which is exactly what real
    systems do on takeover.
    """
    store = Fenced()
    zombie = store.grant()
    store.write(zombie, b"k", b"z1")
    store.grant()
    store.write(zombie, b"k", b"z2")
    landed = store.read(b"k") == b"z2"
    return landed and store.refusals == 0


@functools.cache
def equal_tokens_are_admitted() -> bool:
    """A writer's own retries pass: the refusal is strictly below, not at.

    Refusing equal tokens would make every legitimate retry a conflict, and the whole
    point of the design is that the current owner can write freely without coordination.
    """
    store = Fenced()
    token = store.grant()
    store.write(token, b"k", b"first")
    store.write(token, b"k", b"second")
    return store.read(b"k") == b"second" and store.refusals == 0


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "zombies_corrupt_unfenced_stores": the_zombie_corrupts_the_unfenced_store(),
        "the_fence_outranks": the_fence_outranks_the_zombie(),
        "the_fence_needs_the_first_write": the_fence_binds_only_after_the_successor_writes(),
        "owners_retry_freely": equal_tokens_are_admitted(),
    }
