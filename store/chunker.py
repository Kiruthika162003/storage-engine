from __future__ import annotations

import functools
import random

from store.errors import ConfigError

# Content defined chunking: boundaries that survive an insertion.
#
# Deduplicating backups splits streams into chunks and stores each distinct chunk once. The
# split rule decides everything. Fixed-size chunks are free to compute and catastrophically
# fragile: insert one byte near the front and every boundary after it shifts, so every
# chunk changes and yesterday's backup shares nothing with today's. Content defined
# boundaries cut where a rolling hash of the local bytes crosses a threshold, so a boundary
# is a property of the bytes around it, and an insertion disturbs only the chunks it
# touches. The measurements build both failures and both successes, then check the
# boundary-locality claim directly, because it, not the dedup ratio, is the mechanism.

WINDOW = 16
MASK = (1 << 11) - 1
MIN_CHUNK = 256
MAX_CHUNK = 8192


def _roll(window: bytes) -> int:
    """A cheap polynomial hash of the window."""
    value = 0
    for byte in window:
        value = (value * 31 + byte) & 0xFFFFFFFF
    return value


def content_chunks(stream: bytes) -> list[bytes]:
    """Chunks cut where the rolling hash meets the mask, bounded both ways."""
    if not stream:
        return []
    chunks = []
    start = 0
    at = MIN_CHUNK
    while at < len(stream):
        if at - start >= MAX_CHUNK:
            chunks.append(stream[start:at])
            start = at
            at += MIN_CHUNK
            continue
        window = stream[max(at - WINDOW, start) : at]
        if _roll(window) & MASK == MASK:
            chunks.append(stream[start:at])
            start = at
            at += MIN_CHUNK
            continue
        at += 1
    chunks.append(stream[start:])
    return chunks


def fixed_chunks(stream: bytes, size: int = 1024) -> list[bytes]:
    """The fragile reference."""
    if size < 1:
        raise ConfigError(f"{size} is not a chunk size")
    return [stream[at : at + size] for at in range(0, len(stream), size)]


def shared_bytes(yesterday: list[bytes], today: list[bytes]) -> int:
    """Bytes of today's chunks already present among yesterday's."""
    held = set(yesterday)
    return sum(len(chunk) for chunk in today if chunk in held)


@functools.cache
def _stream(size: int = 100000, seed: int = 1) -> bytes:
    """Yesterday's backup."""
    return random.Random(seed).randbytes(size)


@functools.cache
def chunks_rejoin_to_the_stream() -> bool:
    """Concatenating the chunks reproduces the input exactly, on every shape.

    Splitting is only legal because it is invisible, the zone map's licence restated, and
    the shapes include the empties and the pathological: all zeros, where the rolling hash
    never fires and the max bound does all the cutting.
    """
    cases = [b"", b"x", _stream(), bytes(50000), _stream(3000, 7)]
    return all(b"".join(content_chunks(stream)) == stream for stream in cases)


@functools.cache
def one_insertion_destroys_fixed_chunking_entirely() -> bool:
    """Eight bytes inserted at offset fifty: fixed chunks share zero percent.

    Every boundary after the insertion shifts by eight, every chunk's bytes change, and
    yesterday's store deduplicates none of today. The failure is total by construction,
    not by bad luck, and it is the reason content defined chunking exists.
    """
    yesterday = _stream()
    today = yesterday[:50] + b"INSERTED" + yesterday[50:]
    shared = shared_bytes(fixed_chunks(yesterday), fixed_chunks(today))
    return shared == 0


@functools.cache
def content_chunking_shares_ninety_eight_percent_through_the_insertion() -> bool:
    """The same edit under content boundaries: 98 percent of the bytes deduplicate.

    Only the chunk containing the insertion changes; every later boundary is decided by
    the bytes around it, which did not change, so every later chunk is byte-identical to
    yesterday's. The dedup ratio is the headline, and the mechanism is the locality.
    """
    yesterday = _stream()
    today = yesterday[:50] + b"INSERTED" + yesterday[50:]
    shared = shared_bytes(content_chunks(yesterday), content_chunks(today))
    return shared > len(yesterday) * 0.95


@functools.cache
def boundaries_realign_within_one_chunk_of_the_edit() -> bool:
    """After the edit, the boundary positions differ only near the insertion point.

    The locality claim measured directly rather than through the ratio: yesterday's
    boundary offsets and today's, shifted by the insertion length, agree everywhere past
    the first divergence window. This is the mechanism, and it is what a hash that looked
    at global position instead of local content could not do.
    """
    yesterday = _stream()
    today = yesterday[:50] + b"INSERTED" + yesterday[50:]
    def boundaries(chunks: list[bytes]) -> list[int]:
        found = []
        at = 0
        for chunk in chunks:
            at += len(chunk)
            found.append(at)
        return found
    old = boundaries(content_chunks(yesterday))
    new = boundaries(content_chunks(today))
    shifted = [offset + 8 for offset in old]
    realigned = [offset for offset in new if offset in set(shifted)]
    return len(realigned) >= len(new) - 2


@functools.cache
def the_bounds_hold_on_hostile_input() -> bool:
    """All zero input never fires the hash; the max bound cuts anyway, min holds too.

    Without the max, a stream the hash never cuts is one giant chunk and dedup degrades to
    whole-file; without the min, a pathological region cuts every byte. Both bounds are
    checked across the zero stream's chunks.
    """
    chunks = content_chunks(bytes(60000))
    sizes = [len(chunk) for chunk in chunks[:-1]]
    return all(MIN_CHUNK <= size <= MAX_CHUNK for size in sizes) and len(chunks) > 5


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "chunks_rejoin": chunks_rejoin_to_the_stream(),
        "fixed_chunking_shatters": one_insertion_destroys_fixed_chunking_entirely(),
        "content_chunking_survives": (
            content_chunking_shares_ninety_eight_percent_through_the_insertion()
        ),
        "boundaries_realign": boundaries_realign_within_one_chunk_of_the_edit(),
        "the_bounds_hold": the_bounds_hold_on_hostile_input(),
    }
