from __future__ import annotations

import functools
from dataclasses import dataclass, field

from store.errors import Closed, ConfigError, TooLarge

# The buffer pool: pages on loan, and the discipline that makes eviction legal.
#
# A cache can evict anything at any time because its contents are copies. A buffer pool cannot,
# because callers hold direct references to its pages: evicting a pinned page pulls memory out
# from under a reader mid scan. So every page carries a pin count, eviction considers only
# unpinned pages, and the pool's real failure mode is not a miss but exhaustion, every slot
# pinned and a caller asking for one more.
#
# The quiet bug this structure breeds is the leaked pin: a caller that returns without
# unpinning. Nothing fails at the leak. The pool just has one less evictable slot forever, and
# after enough leaks it deadlocks under load that used to fit. The pool therefore keeps enough
# accounting to name the leaker, which turns a heisenbug into a line number.


@dataclass
class Page:
    """One slot: the page's number, its bytes, and who holds it."""

    number: int
    payload: bytes
    pins: int = field(default=0)
    dirty: bool = field(default=False)
    holders: list[str] = field(default_factory=list)


@dataclass
class Pool:
    """A fixed set of slots with pin discipline."""

    capacity: int
    slots: dict[int, Page] = field(default_factory=dict)
    hits: int = field(default=0)
    misses: int = field(default=0)
    evictions: int = field(default=0)
    write_backs: int = field(default=0)
    exhausted: int = field(default=0)
    fetches: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ConfigError(f"{self.capacity} is not a capacity")

    def pin(self, number: int, holder: str = "anonymous") -> Page:
        """Borrow a page, fetching it if absent, evicting someone unpinned if full."""
        if number in self.slots:
            self.hits += 1
        else:
            self.misses += 1
            if len(self.slots) >= self.capacity:
                self._evict()
            self.slots[number] = Page(number=number, payload=self._fetch(number))
        page = self.slots[number]
        page.pins += 1
        page.holders.append(holder)
        return page

    def unpin(self, page: Page, dirty: bool = False, holder: str = "anonymous") -> None:
        """Return a loan, optionally marking the page changed."""
        if page.pins < 1:
            raise Closed(f"page {page.number} is not pinned")
        page.pins -= 1
        if holder in page.holders:
            page.holders.remove(holder)
        page.dirty = page.dirty or dirty

    def _fetch(self, number: int) -> bytes:
        """The page's bytes from below, deterministic so tests can check them."""
        self.fetches.append(number)
        return number.to_bytes(8, "little") * 512

    def _evict(self) -> None:
        """Drop one unpinned page, writing it back first if it is dirty."""
        for number, page in self.slots.items():
            if page.pins == 0:
                if page.dirty:
                    self.write_backs += 1
                del self.slots[number]
                self.evictions += 1
                return
        self.exhausted += 1
        raise TooLarge(self._blame())

    def _blame(self) -> str:
        """Name every holder, which is the difference between a bug report and a shrug."""
        holders = sorted(
            {holder for page in self.slots.values() for holder in page.holders}
        )
        return f"every slot is pinned; holders: {', '.join(holders) or 'unknown'}"

    def pinned(self) -> int:
        """Slots currently on loan."""
        return sum(1 for page in self.slots.values() if page.pins > 0)

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "capacity": self.capacity,
            "held": len(self.slots),
            "pinned": self.pinned(),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "write_backs": self.write_backs,
            "exhausted": self.exhausted,
        }


@functools.cache
def a_pinned_page_survives_any_pressure() -> bool:
    """A hundred fetches through a four slot pool never evict the pinned page.

    The pin is the only thing standing between a scan and a page that changes underneath it,
    and pressure is when the guarantee earns its keep: every other slot churned twenty five
    times over while the pinned one sat still.
    """
    pool = Pool(capacity=4)
    held = pool.pin(999, holder="the-scan")
    for number in range(100):
        page = pool.pin(number, holder="churn")
        pool.unpin(page, holder="churn")
    return 999 in pool.slots and pool.slots[999] is held and pool.evictions > 90


@functools.cache
def exhaustion_names_the_holders() -> bool:
    """When every slot is pinned, the refusal lists who is holding them.

    The leaked pin fails nowhere near its cause, so the pool's accounting is the whole
    debugging story: the exception carries the holder names, and the test asserts the name of
    the leaker is actually in it.
    """
    pool = Pool(capacity=2)
    pool.pin(1, holder="reader-a")
    pool.pin(2, holder="leaky-scan")
    try:
        pool.pin(3, holder="victim")
    except TooLarge as refusal:
        return "leaky-scan" in str(refusal) and "reader-a" in str(refusal)
    return False


@functools.cache
def a_dirty_page_is_written_back_exactly_once() -> bool:
    """Eviction writes changed pages down and drops clean ones silently.

    Write backs are the pool's write amplification, and the dirty bit is what keeps it at
    changed pages only: a pool that wrote back everything would double every read heavy
    workload's IO for nothing.
    """
    pool = Pool(capacity=2)
    page = pool.pin(1)
    pool.unpin(page, dirty=True)
    clean = pool.pin(2)
    pool.unpin(clean)
    pool.pin(3)
    pool.pin(4)
    return pool.write_backs == 1 and pool.evictions == 2


@functools.cache
def a_double_unpin_is_refused() -> bool:
    """Unpinning below zero raises, because it is two owners disagreeing again.

    A pin count that goes negative makes the page evictable while somebody still holds it,
    which is the exact corruption the pin exists to prevent, arriving through the exit.
    """
    pool = Pool(capacity=2)
    page = pool.pin(1)
    pool.unpin(page)
    try:
        pool.unpin(page)
    except Closed:
        return True
    return False


@functools.cache
def repinning_is_free_and_counted() -> bool:
    """A page pinned twice is fetched once and held until both pins return.

    The second pin is a hit, the page is one copy, and eviction waits for the last holder,
    which is what lets two scans share a hot page instead of fighting over it.
    """
    pool = Pool(capacity=2)
    first = pool.pin(7, holder="a")
    second = pool.pin(7, holder="b")
    shared = first is second and pool.misses == 1 and pool.hits == 1
    pool.unpin(first, holder="a")
    for number in range(10):
        page = pool.pin(number + 100, holder="churn")
        pool.unpin(page, holder="churn")
    still_here = 7 in pool.slots
    pool.unpin(second, holder="b")
    return shared and still_here


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "pins_survive_pressure": a_pinned_page_survives_any_pressure(),
        "exhaustion_names_names": exhaustion_names_the_holders(),
        "dirty_pages_write_back_once": a_dirty_page_is_written_back_exactly_once(),
        "double_unpin_is_refused": a_double_unpin_is_refused(),
        "repinning_shares": repinning_is_free_and_counted(),
    }
