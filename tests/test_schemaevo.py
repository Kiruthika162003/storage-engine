from __future__ import annotations

import pytest

from store import schemaevo as mod
from store.errors import BadFormat, ConfigError
from store.schemaevo import V1, V2, V3, FieldSpec, Schema


class TestSchema:
    def test_colliding_tags_are_refused(self):
        with pytest.raises(ConfigError):
            Schema(
                version=1,
                fields=(
                    FieldSpec(tag=1, name="a", default=b""),
                    FieldSpec(tag=1, name="b", default=b""),
                ),
            )

    def test_unknown_field_names_are_refused_at_write(self):
        with pytest.raises(ConfigError):
            V1.write({"user": b"x", "flavour": b"vanilla"})

    def test_a_write_read_round_trips(self):
        values = {"user": b"kim", "amount": b"\x07"}
        assert V1.read(V1.write(values)) == values

    def test_omitted_fields_write_their_defaults(self):
        found = V1.read(V1.write({"user": b"kim"}))
        assert found["amount"] == b"\x00"

    def test_empty_values_survive(self):
        found = V1.read(V1.write({"user": b"", "amount": b""}))
        assert found["user"] == b"" and found["amount"] == b""


class TestCompatibility:
    def test_v1_reads_v2_records(self):
        raw = V2.write({"user": b"kim", "amount": b"\x01", "currency": b"GBP"})
        found = V1.read(raw)
        assert found == {"user": b"kim", "amount": b"\x01"}

    def test_v2_reads_v1_records_with_defaults(self):
        raw = V1.write({"user": b"kim", "amount": b"\x01"})
        found = V2.read(raw)
        assert found["currency"] == b"USD"

    def test_v3_reads_v1_records(self):
        raw = V1.write({"user": b"kim"})
        found = V3.read(raw)
        assert found["region"] == b"unset" and found["user"] == b"kim"

    def test_v1_reads_v3_like_v1(self):
        v3 = V3.write({"user": b"kim", "amount": b"\x05", "currency": b"EUR", "region": b"x"})
        v1 = V1.write({"user": b"kim", "amount": b"\x05"})
        assert V1.read(v3) == V1.read(v1)


class TestDamage:
    def test_a_cut_header_is_refused(self):
        raw = V1.write({"user": b"kim"})
        with pytest.raises(BadFormat):
            V1.read(raw[:2])

    def test_a_cut_body_is_refused(self):
        raw = V1.write({"user": b"kimberly"})
        with pytest.raises(BadFormat):
            V1.read(raw[:-2])


class TestMeasurements:
    def test_the_matrix_holds(self):
        assert mod.every_writer_reader_pairing_behaves()

    def test_old_readers_are_untouched(self):
        assert mod.old_readers_are_untouched_by_new_fields()

    def test_new_readers_default(self):
        assert mod.new_readers_default_what_old_writers_omitted()

    def test_tag_reuse_lies(self):
        assert mod.tag_reuse_is_the_unfixable_mistake()

    def test_torn_records_refuse(self):
        assert mod.torn_records_are_refused()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5
