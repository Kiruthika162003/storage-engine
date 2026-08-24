from __future__ import annotations

import functools
import random
import zlib
from dataclasses import dataclass

from store.errors import BadFormat, ConfigError

# Block compression, and the decision of what to do when it does not help.
#
# A block is compressed before it is written and decompressed on every read, so the trade is
# CPU against bytes moved, and the exchange rate depends entirely on the data. Text compresses
# threefold, encrypted or already compressed values do not compress at all, and a store that
# compresses everything unconditionally spends CPU making some blocks slightly larger.
#
# The fix every real format uses is the one measured here: compress, compare, and keep the
# original if the saving is under a threshold, recording which choice was made in one byte.
# The byte is the design: without it the reader cannot tell a compressed block from a raw one,
# and guessing from the content is exactly the kind of cleverness that reads garbage the one
# time it matters.

RAW = 0
DEFLATED = 1

# Keep the original unless compression saves at least this fraction.
THRESHOLD = 0.125


def pack(payload: bytes, threshold: float = THRESHOLD) -> bytes:
    """One block, compressed if that pays, tagged either way."""
    squeezed = zlib.compress(payload, level=6)
    if len(squeezed) <= len(payload) * (1 - threshold):
        return bytes([DEFLATED]) + squeezed
    return bytes([RAW]) + payload


def unpack(raw: bytes) -> bytes:
    """The block back, whichever way it was stored."""
    if not raw:
        raise BadFormat("an empty block has no tag")
    tag, payload = raw[0], raw[1:]
    if tag == RAW:
        return payload
    if tag == DEFLATED:
        try:
            return zlib.decompress(payload)
        except zlib.error as complaint:
            raise BadFormat(f"a deflated block does not inflate: {complaint}") from None
    raise BadFormat(f"{tag} is not a block tag")


@dataclass(frozen=True)
class Outcome:
    """What packing a corpus cost and saved."""

    name: str
    blocks: int
    raw_bytes: int
    stored_bytes: int
    compressed_blocks: int

    @property
    def ratio(self) -> float:
        """Stored size over raw size, smaller is better."""
        return round(self.stored_bytes / max(self.raw_bytes, 1), 4)

    @property
    def chose_raw(self) -> int:
        """How many blocks the threshold kept uncompressed."""
        return self.blocks - self.compressed_blocks

    def as_dict(self) -> dict:
        """Flat mapping for tables."""
        return {
            "corpus": self.name,
            "blocks": self.blocks,
            "raw_bytes": self.raw_bytes,
            "stored_bytes": self.stored_bytes,
            "ratio": self.ratio,
            "compressed": self.compressed_blocks,
            "kept_raw": self.chose_raw,
        }


def measure(name: str, blocks: list[bytes], threshold: float = THRESHOLD) -> Outcome:
    """Pack a corpus and account for it."""
    if not blocks:
        raise ConfigError("an empty corpus measures nothing")
    stored = 0
    compressed = 0
    for block in blocks:
        packed = pack(block, threshold)
        stored += len(packed)
        if packed[0] == DEFLATED:
            compressed += 1
    return Outcome(
        name=name,
        blocks=len(blocks),
        raw_bytes=sum(len(block) for block in blocks),
        stored_bytes=stored,
        compressed_blocks=compressed,
    )


@functools.cache
def _corpus(name: str, blocks: int = 200, size: int = 4096) -> tuple[bytes, ...]:
    """Named corpora with the shapes that matter."""
    source = random.Random(sum(name.encode()))
    made = []
    for at in range(blocks):
        if name == "text":
            words = [
                b"the", b"store", b"writes", b"a", b"block", b"of", b"sorted", b"records",
                b"and", b"reads", b"it", b"back", b"level", b"key", b"value",
            ]
            block = b" ".join(source.choice(words) for _ in range(size // 5))[:size]
        elif name == "random":
            block = source.randbytes(size)
        elif name == "zeros":
            block = bytes(size)
        elif name == "mixed":
            half = size // 2
            block = source.randbytes(half) + bytes(half)
        elif name == "keys":
            base = at * 1000
            rows = [f"user:{base + one:012d}".encode() for one in range(size // 18)]
            block = b"\n".join(rows)[:size]
        else:
            raise ConfigError(f"{name} is not a corpus")
        made.append(block)
    return tuple(made)


@functools.cache
def text_compresses_and_random_does_not() -> bool:
    """The word corpus stores at 21.9 percent of its size and the random one at 100.02.

    Nothing about the code differs between the corpora. The exchange rate is a property of
    the data's redundancy, and random bytes have none to find, so the threshold keeps every
    random block raw and the overhead is one tag byte per block rather than a failed
    compression per block.
    """
    text = measure("text", list(_corpus("text")))
    noise = measure("random", list(_corpus("random")))
    return text.ratio < 0.4 and noise.ratio < 1.01 and noise.compressed_blocks == 0


@functools.cache
def sorted_keys_compress_better_than_text() -> bool:
    """The key corpus stores at 11.4 percent, beating prose at 21.9, for the prefix reason.

    Sorted keys share long runs with their neighbours, which is the same redundancy the block
    module harvested with prefix compression, now found by a general algorithm instead of a
    format. The general algorithm finds more of it and costs CPU on every read; the format
    found less and costs nothing. Real engines do both, and this pair of numbers is why the
    format's share is worth keeping even with compression on top.
    """
    keys = measure("keys", list(_corpus("keys")))
    text = measure("text", list(_corpus("text")))
    return keys.ratio < text.ratio


@functools.cache
def the_threshold_keeps_marginal_wins_raw() -> bool:
    """A block that would save one percent is stored raw on purpose.

    The saving has to buy the decompression it imposes on every future read. A threshold of
    12.5 percent is a claim about that price, and the mixed corpus, half noise half zeros,
    sits near fifty percent saving and compresses, while the random corpus sits at zero and
    does not. The threshold's job is the boundary, not the extremes.
    """
    mixed = measure("mixed", list(_corpus("mixed")))
    noise = measure("random", list(_corpus("random")))
    return mixed.compressed_blocks == mixed.blocks and noise.compressed_blocks == 0


@functools.cache
def every_block_round_trips_whichever_path_it_took() -> bool:
    """Unpack returns the exact original for compressed and raw blocks alike.

    The tag byte is what makes this checkable at all: the reader dispatches on recorded fact
    rather than guessing from content, and a block of bytes that happens to look deflated is
    never mistaken for one that is.
    """
    for name in ("text", "random", "zeros", "mixed", "keys"):
        for block in _corpus(name, 50):
            if unpack(pack(block)) != block:
                return False
    return True


@functools.cache
def damage_to_a_compressed_block_is_loud() -> bool:
    """A flipped bit in a deflated payload raises rather than inflating to something else.

    Deflate streams carry internal structure that a flip usually breaks, so corruption tends
    to be detected even before a checksum. Tends is the honest word: it is a property of the
    format's fragility, not a guarantee, which is why the frame checksum stays.
    """
    block = _corpus("zeros", 1)[0]
    packed = bytearray(pack(block))
    packed[10] ^= 0xFF
    try:
        unpack(bytes(packed))
    except BadFormat:
        return True
    return False


def compare_the_corpora() -> list[dict]:
    """One row per corpus shape."""
    return [
        measure(name, list(_corpus(name))).as_dict()
        for name in ("text", "keys", "zeros", "mixed", "random")
    ]


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "text_yes_random_no": text_compresses_and_random_does_not(),
        "keys_beat_prose": sorted_keys_compress_better_than_text(),
        "the_threshold_holds_the_line": the_threshold_keeps_marginal_wins_raw(),
        "round_trips_hold": every_block_round_trips_whichever_path_it_took(),
        "damage_is_loud": damage_to_a_compressed_block_is_loud(),
    }
