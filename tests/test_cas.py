from __future__ import annotations

import pytest

from store import cas as mod
from store.cas import Register
from store.errors import ConfigError, Conflict, NotFound


class TestRegister:
    def test_an_empty_key_is_refused(self):
        with pytest.raises(ConfigError):
            Register().put(b"", b"v")

    def test_a_put_reads_back_with_version_one(self):
        register = Register()
        register.put(b"k", b"v")
        assert register.read(b"k") == (b"v", 1)

    def test_a_second_put_bumps_the_version(self):
        register = Register()
        register.put(b"k", b"v1")
        register.put(b"k", b"v2")
        assert register.read(b"k") == (b"v2", 2)

    def test_a_missing_key_raises(self):
        with pytest.raises(NotFound):
            Register().read(b"k")


class TestValueCas:
    def test_a_matching_compare_swaps(self):
        register = Register()
        register.put(b"k", b"old")
        register.cas_value(b"k", b"old", b"new")
        assert register.read(b"k")[0] == b"new"

    def test_a_mismatched_compare_refuses(self):
        register = Register()
        register.put(b"k", b"other")
        with pytest.raises(Conflict):
            register.cas_value(b"k", b"old", b"new")

    def test_a_missing_key_refuses(self):
        with pytest.raises(Conflict):
            Register().cas_value(b"k", b"old", b"new")

    def test_a_swap_bumps_the_version(self):
        register = Register()
        register.put(b"k", b"old")
        register.cas_value(b"k", b"old", b"new")
        assert register.read(b"k")[1] == 2

    def test_refusals_are_counted(self):
        register = Register()
        register.put(b"k", b"x")
        with pytest.raises(Conflict):
            register.cas_value(b"k", b"y", b"z")
        assert register.refusals == 1


class TestVersionCas:
    def test_a_matching_version_swaps(self):
        register = Register()
        register.put(b"k", b"old")
        register.cas_version(b"k", 1, b"new")
        assert register.read(b"k")[0] == b"new"

    def test_a_stale_version_refuses(self):
        register = Register()
        register.put(b"k", b"old")
        register.put(b"k", b"newer")
        with pytest.raises(Conflict):
            register.cas_version(b"k", 1, b"mine")

    def test_the_aba_is_refused_by_version(self):
        register = Register()
        register.put(b"k", b"X")
        register.put(b"k", b"B")
        register.put(b"k", b"X")
        with pytest.raises(Conflict):
            register.cas_version(b"k", 1, b"mine")

    def test_the_aba_is_accepted_by_value(self):
        register = Register()
        register.put(b"k", b"X")
        register.put(b"k", b"B")
        register.put(b"k", b"X")
        register.cas_value(b"k", b"X", b"mine")
        assert register.read(b"k")[0] == b"mine"


class TestMeasurements:
    def test_the_loop_beats_the_race(self):
        assert mod.a_cas_loop_survives_interleaving_that_breaks_read_modify_write()

    def test_value_compares_accept_aba(self):
        assert mod.the_value_compare_accepts_the_aba()

    def test_version_compares_refuse_aba(self):
        assert mod.the_version_compare_refuses_the_aba()

    def test_versions_only_climb(self):
        assert mod.versions_only_climb()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4
