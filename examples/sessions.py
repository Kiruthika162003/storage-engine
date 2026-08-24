"""A session store on the TTL shelf: logins, sliding expiry, and the reaper.

Run with: python -m examples.sessions
"""

from __future__ import annotations

import random

from store.ttl import Shelf

SESSION_TICKS = 30
USERS = 200
TICKS = 300


def login(shelf: Shelf, user: int, now_token: int) -> bytes:
    """A login writes a session with a fresh lifetime."""
    token = f"sess:{user:04d}:{now_token}".encode()
    shelf.put(token, f"user-{user}".encode(), ttl=SESSION_TICKS)
    return token


def touch(shelf: Shelf, token: bytes) -> bool:
    """Activity slides the expiry by rewriting with a fresh lifetime."""
    held = shelf.get(token)
    if held is None:
        return False
    shelf.put(token, held, ttl=SESSION_TICKS)
    return True


def main() -> int:
    source = random.Random(3)
    shelf = Shelf()
    tokens: dict[int, bytes] = {}
    expired_on_touch = 0
    logins = 0
    for tick in range(TICKS):
        shelf.tick()
        user = source.randrange(USERS)
        if user not in tokens or source.random() < 0.02:
            tokens[user] = login(shelf, user, tick)
            logins += 1
        elif not touch(shelf, tokens[user]):
            expired_on_touch += 1
            del tokens[user]
        if tick % 100 == 99:
            swept = shelf.sweep()
            print(f"tick {tick}: swept {swept}, held {shelf.held}, live {shelf.live()}")
    print(f"logins {logins}, sessions that expired under a touch {expired_on_touch}")
    print(shelf.as_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
