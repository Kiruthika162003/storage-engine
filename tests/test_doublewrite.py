from __future__ import annotations

import pytest

from store import doublewrite as mod
from store.doublewrite import PAGE, Pages
from store.errors import BadChecksum, ConfigError


class TestBasics:
    def test_a_wrong_size_payload_is_refused(self):
        with pytest.raises(ConfigError):
            Pages().write_direct(1, b"short")

    def test_a_direct_write_reads_back(self):
        device = Pages()
        device.write_direct(1, mod._payload(5))
        assert device.recover(1) == mod._payload(5)

    def test_a_double_write_reads_back(self):
        device = Pages()
        device.write_double(1, mod._payload(5))
        assert device.recover(1) == mod._payload(5)

    def test_an_unwritten_page_fails_recovery(self):
        with pytest.raises(BadChecksum):
            Pages().recover(9)

    def test_double_writes_are_counted(self):
        device = Pages()
        device.write_double(1, mod._payload(1))
        device.write_double(2, mod._payload(2))
        assert device.double_writes == 2


class TestTears:
    def test_a_mid_page_direct_tear_is_detected(self):
        device = Pages()
        device.write_direct(1, mod._payload(1))
        device.write_direct(1, mod._payload(2), tear_at=PAGE // 2)
        with pytest.raises(BadChecksum):
            device.recover(1)

    def test_a_scratch_tear_recovers_the_old_page(self):
        device = Pages()
        device.write_double(1, mod._payload(1))
        device.write_double(1, mod._payload(2), tear_scratch=PAGE // 2)
        assert device.recover(1) == mod._payload(1)

    def test_a_home_tear_recovers_the_new_page(self):
        device = Pages()
        device.write_double(1, mod._payload(1))
        device.write_double(1, mod._payload(2), tear_home=PAGE // 2)
        assert device.recover(1) == mod._payload(2)

    def test_a_home_tear_of_another_page_is_unrecoverable(self):
        device = Pages()
        device.write_double(1, mod._payload(1))
        device.write_double(2, mod._payload(2))
        device.write_direct(1, mod._payload(3), tear_at=PAGE // 2)
        with pytest.raises(BadChecksum):
            device.recover(1)


class TestMeasurements:
    def test_direct_tears_lose_the_page(self):
        assert mod.a_direct_tear_loses_both_versions()

    def test_scratch_tears_keep_the_home(self):
        assert mod.a_scratch_tear_keeps_the_home_intact()

    def test_home_tears_recover_from_scratch(self):
        assert mod.a_home_tear_recovers_from_the_scratch()

    def test_the_price_is_double(self):
        assert mod.the_price_is_exactly_double()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_four_claims(self):
        assert len(mod.summarise()) == 4
