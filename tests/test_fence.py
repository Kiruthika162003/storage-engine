from __future__ import annotations

import pytest

from store import fence as mod
from store.errors import ConfigError, Conflict
from store.fence import Fenced, Unfenced


class TestFenced:
    def test_grants_climb(self):
        store = Fenced()
        assert store.grant() == 1 and store.grant() == 2

    def test_a_zero_token_is_refused(self):
        with pytest.raises(ConfigError):
            Fenced().write(0, b"k", b"v")

    def test_a_granted_token_writes(self):
        store = Fenced()
        token = store.grant()
        store.write(token, b"k", b"v")
        assert store.read(b"k") == b"v"

    def test_a_higher_token_raises_the_bar(self):
        store = Fenced()
        old = store.grant()
        new = store.grant()
        store.write(new, b"k", b"new")
        with pytest.raises(Conflict):
            store.write(old, b"k", b"old")

    def test_the_bar_holds_across_keys(self):
        store = Fenced()
        old = store.grant()
        new = store.grant()
        store.write(new, b"a", b"new")
        with pytest.raises(Conflict):
            store.write(old, b"b", b"old")

    def test_refusals_are_counted(self):
        store = Fenced()
        old = store.grant()
        new = store.grant()
        store.write(new, b"k", b"v")
        with pytest.raises(Conflict):
            store.write(old, b"k", b"x")
        assert store.refusals == 1

    def test_an_unknown_key_reads_none(self):
        assert Fenced().read(b"missing") is None

    def test_the_same_token_writes_repeatedly(self):
        store = Fenced()
        token = store.grant()
        store.write(token, b"k", b"1")
        store.write(token, b"k", b"2")
        assert store.read(b"k") == b"2"


class TestUnfenced:
    def test_belief_is_the_only_gate(self):
        store = Unfenced()
        store.write(True, b"k", b"v")
        store.write(False, b"k", b"ignored")
        assert store.read(b"k") == b"v"


class TestMeasurements:
    def test_zombies_corrupt_unfenced_stores(self):
        assert mod.the_zombie_corrupts_the_unfenced_store()

    def test_the_fence_outranks(self):
        assert mod.the_fence_outranks_the_zombie()

    def test_the_fence_needs_the_first_write(self):
        assert mod.the_fence_binds_only_after_the_successor_writes()

    def test_owners_retry_freely(self):
        assert mod.equal_tokens_are_admitted()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4
