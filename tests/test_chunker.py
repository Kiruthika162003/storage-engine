from __future__ import annotations

import pytest

from store import chunker as mod
from store.chunker import (
    MAX_CHUNK,
    MIN_CHUNK,
    content_chunks,
    fixed_chunks,
    shared_bytes,
)
from store.errors import ConfigError


class TestFixed:
    def test_a_zero_size_is_refused(self):
        with pytest.raises(ConfigError):
            fixed_chunks(b"abc", size=0)

    def test_the_chunks_tile_the_stream(self):
        assert fixed_chunks(b"abcdefgh", size=3) == [b"abc", b"def", b"gh"]

    def test_an_empty_stream_has_no_chunks(self):
        assert fixed_chunks(b"") == []


class TestContent:
    def test_an_empty_stream_has_no_chunks(self):
        assert content_chunks(b"") == []

    def test_a_tiny_stream_is_one_chunk(self):
        assert content_chunks(b"small") == [b"small"]

    def test_chunks_rejoin(self):
        stream = mod._stream(20000, 3)
        assert b"".join(content_chunks(stream)) == stream

    def test_interior_chunks_respect_the_bounds(self):
        stream = mod._stream(60000, 4)
        for chunk in content_chunks(stream)[:-1]:
            assert MIN_CHUNK <= len(chunk) <= MAX_CHUNK

    def test_the_same_stream_chunks_the_same_way(self):
        stream = mod._stream(20000, 5)
        assert content_chunks(stream) == content_chunks(stream)

    def test_identical_tails_chunk_identically(self):
        base = mod._stream(30000, 6)
        other = b"DIFFERENT-PREFIX" + base
        base_tail = {bytes(c) for c in content_chunks(base)[2:]}
        other_tail = {bytes(c) for c in content_chunks(other)[2:]}
        assert len(base_tail & other_tail) >= len(base_tail) - 3


class TestSharing:
    def test_shared_bytes_counts_common_chunks(self):
        yesterday = [b"aa", b"bb"]
        today = [b"bb", b"cc"]
        assert shared_bytes(yesterday, today) == 2

    def test_disjoint_chunkings_share_nothing(self):
        assert shared_bytes([b"aa"], [b"bb"]) == 0

    def test_an_append_only_change_shares_almost_everything(self):
        yesterday = mod._stream(50000, 7)
        today = yesterday + mod._stream(5000, 8)
        shared = shared_bytes(content_chunks(yesterday), content_chunks(today))
        assert shared > 40000


class TestMeasurements:
    def test_chunks_rejoin(self):
        assert mod.chunks_rejoin_to_the_stream()

    def test_fixed_chunking_shatters(self):
        assert mod.one_insertion_destroys_fixed_chunking_entirely()

    def test_content_chunking_survives(self):
        assert mod.content_chunking_shares_ninety_eight_percent_through_the_insertion()

    def test_boundaries_realign(self):
        assert mod.boundaries_realign_within_one_chunk_of_the_edit()

    def test_the_bounds_hold(self):
        assert mod.the_bounds_hold_on_hostile_input()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5

    def test_the_stream_is_cached(self):
        assert mod._stream(1000) is mod._stream(1000)

    def test_random_streams_differ_by_seed(self):
        assert mod._stream(1000, 1) != mod._stream(1000, 2)
