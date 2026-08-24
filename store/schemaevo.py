from __future__ import annotations

import functools
import struct
from dataclasses import dataclass

from store.errors import BadFormat, ConfigError

# Schema evolution: records that outlive the code that wrote them.
#
# A store's records are read by every version of the application that will ever run, past
# and future, and the encoding either plans for that or the first added field is a fleet
# wide migration. The plan here is the tag-length-value discipline: every field carries a
# numeric tag and its length, readers take the tags they know and step over the rest, and
# the two compatibility directions fall out. Old readers skip fields added after them,
# forward compatibility; new readers default fields the old writers never wrote, backward
# compatibility. The measurements run writers and readers of three schema versions against
# each other, all nine pairings, because compatibility is a matrix, not a property.

FIELD_HEADER = struct.Struct("<HH")


@dataclass(frozen=True)
class FieldSpec:
    """One field a schema version knows: its tag and its default."""

    tag: int
    name: str
    default: bytes


@dataclass
class Schema:
    """One version of the record layout."""

    version: int
    fields: tuple[FieldSpec, ...]

    def __post_init__(self) -> None:
        tags = [spec.tag for spec in self.fields]
        if len(tags) != len(set(tags)):
            raise ConfigError("field tags collide")

    def write(self, values: dict[str, bytes]) -> bytes:
        """A record with this version's fields, unknown names refused."""
        known = {spec.name for spec in self.fields}
        strange = set(values) - known
        if strange:
            raise ConfigError(f"unknown fields: {sorted(strange)}")
        made = bytearray()
        for spec in self.fields:
            value = values.get(spec.name, spec.default)
            made.extend(FIELD_HEADER.pack(spec.tag, len(value)))
            made.extend(value)
        return bytes(made)

    def read(self, raw: bytes) -> dict[str, bytes]:
        """Every field this version knows, defaults filled, unknown tags stepped over."""
        found: dict[int, bytes] = {}
        at = 0
        while at < len(raw):
            if at + FIELD_HEADER.size > len(raw):
                raise BadFormat("a field header ended early")
            tag, length = FIELD_HEADER.unpack_from(raw, at)
            at += FIELD_HEADER.size
            if at + length > len(raw):
                raise BadFormat("a field body ended early")
            found[tag] = raw[at : at + length]
            at += length
        return {
            spec.name: found.get(spec.tag, spec.default) for spec in self.fields
        }


V1 = Schema(
    version=1,
    fields=(
        FieldSpec(tag=1, name="user", default=b""),
        FieldSpec(tag=2, name="amount", default=b"\x00"),
    ),
)
V2 = Schema(
    version=2,
    fields=(
        *V1.fields,
        FieldSpec(tag=3, name="currency", default=b"USD"),
    ),
)
V3 = Schema(
    version=3,
    fields=(
        *V2.fields,
        FieldSpec(tag=4, name="region", default=b"unset"),
    ),
)
VERSIONS = (V1, V2, V3)


def _record_for(schema: Schema) -> dict[str, bytes]:
    """A full record in a version's own vocabulary."""
    values = {"user": b"kim", "amount": b"\x2a"}
    if schema.version >= 2:
        values["currency"] = b"EUR"
    if schema.version >= 3:
        values["region"] = b"eu-1"
    return values


@functools.cache
def every_writer_reader_pairing_behaves() -> bool:
    """All nine version pairings: shared fields survive, gaps default, extras skip.

    The matrix is the claim. For every writer version and every reader version, the fields
    both know come through byte for byte, the fields only the reader knows arrive as its
    defaults, and the fields only the writer knows are stepped over without complaint.
    One cell failing is a fleet incident on some specific deploy day, which is why all
    nine run.
    """
    for writer in VERSIONS:
        raw = writer.write(_record_for(writer))
        for reader in VERSIONS:
            found = reader.read(raw)
            for spec in reader.fields:
                writer_knows = any(own.tag == spec.tag for own in writer.fields)
                wanted = (
                    _record_for(writer)[spec.name] if writer_knows else spec.default
                )
                if found[spec.name] != wanted:
                    return False
    return True


@functools.cache
def old_readers_are_untouched_by_new_fields() -> bool:
    """A v1 reader reads a v3 record exactly as it reads a v1 record.

    Forward compatibility in one line: deploys of new writers require nothing of old
    readers, which is what lets a fleet roll gradually instead of atomically.
    """
    v3_raw = V3.write(_record_for(V3))
    v1_raw = V1.write(_record_for(V1))
    return V1.read(v3_raw) == V1.read(v1_raw)


@functools.cache
def new_readers_default_what_old_writers_omitted() -> bool:
    """A v3 reader of a v1 record sees USD and unset, not an error and not garbage.

    Backward compatibility is a statement about defaults: the new code must have an answer
    for the missing field, chosen at schema design time, in the one place, rather than by
    every call site's None handling on the deploy day.
    """
    found = V3.read(V1.write(_record_for(V1)))
    return found["currency"] == b"USD" and found["region"] == b"unset"


@functools.cache
def tag_reuse_is_the_unfixable_mistake() -> bool:
    """A retired tag reused for a new field makes old records lie silently.

    The rogue schema reuses tag 3 for a field meaning something else, and a v2 record's
    currency arrives in the rogue reader's field, bytes intact, meaning wrong, no error
    anywhere. The discipline that tags are never reused is not in the code, it cannot be,
    it is in the review, and this measurement is the demonstration to point the review at.
    """
    rogue = Schema(
        version=9,
        fields=(
            FieldSpec(tag=1, name="user", default=b""),
            FieldSpec(tag=3, name="discount_code", default=b"none"),
        ),
    )
    found = rogue.read(V2.write(_record_for(V2)))
    return found["discount_code"] == b"EUR"


@functools.cache
def torn_records_are_refused() -> bool:
    """A record cut inside a header or a body raises rather than half parsing."""
    raw = V2.write(_record_for(V2))
    for cut in (1, 3, len(raw) - 1):
        try:
            V2.read(raw[:cut])
            return False
        except BadFormat:
            continue
    return True


def summarise() -> dict:
    """Every claim in this module, run."""
    return {
        "the_matrix_holds": every_writer_reader_pairing_behaves(),
        "old_readers_are_untouched": old_readers_are_untouched_by_new_fields(),
        "new_readers_default": new_readers_default_what_old_writers_omitted(),
        "tag_reuse_lies": tag_reuse_is_the_unfixable_mistake(),
        "torn_records_refuse": torn_records_are_refused(),
    }
