"""A bank on the transaction manager: transfers, audits, and a proof the books balance.

Run with: python -m examples.bank
"""

from __future__ import annotations

import random

from store.txn import Manager, balance, transfer

ACCOUNTS = 40
OPENING = 1000
DAYS = 5
TRANSFERS_PER_DAY = 400


def open_the_bank() -> Manager:
    """Every account funded in one transaction, so the bank never half exists."""
    manager = Manager()
    txn = manager.begin()
    for at in range(ACCOUNTS):
        txn.put(f"acct:{at:03d}".encode(), OPENING.to_bytes(8, "big"))
    manager.commit(txn)
    return manager


def a_day_of_business(manager: Manager, seed: int) -> dict:
    """Transfers with retries, the way a client actually uses optimistic concurrency."""
    source = random.Random(seed)
    settled = 0
    retried = 0
    refused = 0
    for _ in range(TRANSFERS_PER_DAY):
        giving = f"acct:{source.randrange(ACCOUNTS):03d}".encode()
        taking = f"acct:{source.randrange(ACCOUNTS):03d}".encode()
        if giving == taking:
            continue
        amount = source.randrange(1, 50)
        for attempt in range(3):
            if transfer(manager, giving, taking, amount):
                settled += 1
                if attempt:
                    retried += 1
                break
        else:
            refused += 1
    return {"settled": settled, "retried": retried, "refused": refused}


def audit(manager: Manager) -> dict:
    """The invariant the bank lives by: the money neither grows nor shrinks."""
    total = sum(balance(manager, f"acct:{at:03d}".encode()) for at in range(ACCOUNTS))
    return {
        "total": total,
        "expected": ACCOUNTS * OPENING,
        "balanced": total == ACCOUNTS * OPENING,
    }


def statement(manager: Manager, account: int) -> dict:
    """One account's view, read through a snapshot so it is one moment, not a smear."""
    held = manager.history.snapshot()
    key = f"acct:{account:03d}".encode()
    found = manager.history.get(key, held)
    manager.history.release(held)
    value = int.from_bytes(found.value, "big") if found else 0
    return {"account": account, "balance": value}


def main() -> int:
    manager = open_the_bank()
    for day in range(DAYS):
        outcome = a_day_of_business(manager, seed=day)
        books = audit(manager)
        print(f"day {day}: {outcome} audit={books}")
        if not books["balanced"]:
            print("the books do not balance; stopping")
            return 1
    print(statement(manager, 0))
    print(f"manager: {manager.as_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
