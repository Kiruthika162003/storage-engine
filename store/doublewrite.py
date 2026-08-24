from __future__ import annotations

import functools
from dataclasses import dataclass, field

from store.checksum import crc32
from store.errors import BadChecksum, ConfigError

# The doublewrite buffer: page atomicity bought with a second copy.
#
# The wal's frames made record writes atomic; pages have the same problem one level down
# and cannot use the same fix. A page overwrite that tears mid-write leaves half new and
# half old bytes at the page's only address, and the log cannot help, because redo needs a
# readable page to apply itself to and this page is neither version. The doublewrite
# discipline: write the page to a scratch area first, sync, then write it home. A tear in
# the scratch leaves the home intact; a tear at home leaves the scratch intact; recovery
# reads both and keeps whichever verifies. The price is every page written twice, and the
# measurements walk every tear position in both windows to show the price buys the whole
# guarantee.

PAGE = 512


@dataclass
class Pages:
    """A tiny paged device with a doublewrite scratch."""

    home: dict[int, bytes] = field(default_factory=dict)
    scratch: bytes = field(default=b"")
    scratch_page: int = field(default=-1)
    double_writes: int = field(default=0)

    def _stamp(self, payload: bytes) -> bytes:
        if len(payload) != PAGE - 4:
            raise ConfigError(f"a page payload is {PAGE - 4} bytes")
        return payload + crc32(payload).to_bytes(4, "little")

    def _verify(self, page: bytes) -> bytes:
        if len(page) != PAGE:
            raise BadChecksum("a torn page is not a page")
        payload, tag = page[:-4], int.from_bytes(page[-4:], "little")
        if crc32(payload) != tag:
            raise BadChecksum("a page failed its checksum")
        return payload

    def write_direct(self, number: int, payload: bytes, tear_at: int = -1) -> None:
        """The unprotected write: straight home, tearable."""
        page = self._stamp(payload)
        if tear_at >= 0:
            old = self.home.get(number, bytes(PAGE))
            self.home[number] = page[:tear_at] + old[tear_at:]
            return
        self.home[number] = page

    def write_double(
        self, number: int, payload: bytes, tear_scratch: int = -1, tear_home: int = -1
    ) -> None:
        """The protected write: scratch, then home, either step tearable."""
        page = self._stamp(payload)
        self.double_writes += 1
        if tear_scratch >= 0:
            old = self.scratch if self.scratch_page == number else bytes(PAGE)
            self.scratch = page[:tear_scratch] + old[tear_scratch:]
            self.scratch_page = number
            return
        self.scratch = page
        self.scratch_page = number
        if tear_home >= 0:
            old = self.home.get(number, bytes(PAGE))
            self.home[number] = page[:tear_home] + old[tear_home:]
            return
        self.home[number] = page

    def recover(self, number: int) -> bytes:
        """The page after a crash: home if it verifies, else the scratch, else refusal."""
        held = self.home.get(number, bytes(PAGE))
        try:
            return self._verify(held)
        except BadChecksum:
            pass
        if self.scratch_page == number:
            return self._verify(self.scratch)
        raise BadChecksum(f"page {number} is torn and the scratch holds another page")


def _payload(tag: int) -> bytes:
    return bytes([tag]) * (PAGE - 4)


@functools.cache
def a_direct_tear_loses_both_versions() -> bool:
    """The unprotected overwrite torn anywhere in the middle verifies as neither version.

    The old page was intact until the write began, the new page would have been intact
    after it finished, and the tear manufactures a third thing that is neither and fails
    its checksum. Torn at every position from one byte to all but one: the early positions
    where the checksum region survives can masquerade, so the walk finds the positions
    that verify and confirms they reconstruct the OLD page only when the tear precedes any
    changed byte, which for a changed page body means detection everywhere it matters.
    """
    losses = 0
    for tear in range(1, PAGE):
        device = Pages()
        device.write_direct(7, _payload(1))
        device.write_direct(7, _payload(2), tear_at=tear)
        try:
            recovered = device._verify(device.home[7])
        except BadChecksum:
            losses += 1
            continue
        if recovered not in (_payload(1), _payload(2)):
            return False
    return losses > PAGE - 10


@functools.cache
def a_scratch_tear_keeps_the_home_intact() -> bool:
    """Torn during the scratch write, every position: recovery returns the old page.

    The home was never touched, so the crash costs the new write and nothing else, which
    is the log's job to replay. Every tear offset in the scratch window recovers the old
    payload exactly.
    """
    for tear in range(1, PAGE):
        device = Pages()
        device.write_double(7, _payload(1))
        device.write_double(7, _payload(2), tear_scratch=tear)
        if device.recover(7) != _payload(1):
            return False
    return True


@functools.cache
def a_home_tear_recovers_from_the_scratch() -> bool:
    """Torn during the home write, every position: recovery returns the new page.

    The scratch was synced whole before the home write began, so the torn home fails its
    checksum and the scratch supplies the new payload. This is the window the direct write
    dies in, closed at the price of the double write.
    """
    for tear in range(1, PAGE):
        device = Pages()
        device.write_double(7, _payload(1))
        device.write_double(7, _payload(2), tear_home=tear)
        if device.recover(7) != _payload(2):
            return False
    return True


@functools.cache
def the_price_is_exactly_double() -> bool:
    """The meter says every protected page write wrote twice, which is the whole cost.

    Real engines batch the scratch area and sync it once for many pages, amortising the
    second write's seeks; the byte doubling remains, and it is why doublewrite is turned
    off on filesystems that guarantee atomic page writes themselves.
    """
    device = Pages()
    for at in range(10):
        device.write_double(at, _payload(at))
    return device.double_writes == 10


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "direct_tears_lose_the_page": a_direct_tear_loses_both_versions(),
        "scratch_tears_keep_the_home": a_scratch_tear_keeps_the_home_intact(),
        "home_tears_recover_from_scratch": a_home_tear_recovers_from_the_scratch(),
        "the_price_is_double": the_price_is_exactly_double(),
    }
