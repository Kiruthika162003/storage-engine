from __future__ import annotations

import pytest

from store import record as records
from store.errors import BadFormat, ConfigError, TooLarge
from store.keys import MAX_VALUE
from store.record import (
    DELETE,
    HEADER,
    KINDS,
    MERGE,
    PUT,
    Record,
    decode,
    decode_all,
    live,
    newest_first,
)


def test_records_round_trip():
    assert records.a_record_round_trips_through_its_encoding()["they_round_trip"]


def test_a_record_predicts_its_size():
    assert records.a_record_round_trips_through_its_encoding()["and_the_length_is_predicted"]


def test_the_smallest_record_is_a_header_and_a_key():
    assert records.a_record_round_trips_through_its_encoding()["which_is_the_header_and_a_key"]


def test_the_keys_ascend_in_order():
    assert records.sorting_by_key_then_sequence_puts_the_newest_first()["the_keys_ascend"]


def test_the_sequences_descend_within_a_key():
    assert records.sorting_by_key_then_sequence_puts_the_newest_first()["and_they_descend"]


def test_the_first_record_for_a_key_is_the_newest():
    assert records.sorting_by_key_then_sequence_puts_the_newest_first()[
        "the_first_for_k_is_the_newest"
    ]


def test_a_delete_makes_the_store_bigger():
    assert records.a_delete_makes_the_store_bigger()["it_grew"]


def test_a_delete_leaves_no_live_keys():
    assert records.a_delete_makes_the_store_bigger()["and_there_are_none"]


def test_a_tombstone_is_smaller_than_the_value_it_hides():
    assert records.a_delete_makes_the_store_bigger()["a_tombstone_is_smaller_than_the_value"]


def test_but_it_does_not_remove_it():
    assert records.a_delete_makes_the_store_bigger()["but_it_does_not_remove_it"]


def test_a_tombstone_with_a_value_is_refused():
    assert records.a_tombstone_with_a_value_is_refused()


def test_a_record_without_a_key_is_refused():
    assert records.a_record_without_a_key_is_refused()


def test_an_oversized_value_is_refused():
    assert records.an_oversized_value_is_refused()


def test_an_unknown_kind_is_refused():
    assert records.an_unknown_kind_is_refused()


def test_every_truncated_record_is_refused():
    assert records.a_truncated_record_is_a_bad_format_and_not_a_crash()[
        "every_prefix_was_refused"
    ]


def test_the_whole_record_still_decodes():
    assert records.a_truncated_record_is_a_bad_format_and_not_a_crash()[
        "and_the_whole_record_decodes"
    ]


def test_the_kind_table_covers_three():
    assert len(records.compare_the_kinds()) == 3


def test_two_kinds_are_complete():
    assert records.only_a_merge_record_needs_the_ones_below_it()["two_of_three_are_complete"]


def test_the_merge_is_not():
    assert records.only_a_merge_record_needs_the_ones_below_it()["and_the_merge_is_not"]


def test_the_tombstone_is_the_smallest_record():
    assert records.only_a_merge_record_needs_the_ones_below_it()["the_tombstone_is_smallest"]


def test_the_summary_says_records_round_trip():
    assert records.summarise()["records_round_trip"]


def test_the_summary_counts_the_kinds():
    assert records.summarise()["kinds"] == len(KINDS)


def test_a_record_reports_its_size():
    made = Record(key=b"ab", sequence=1, value=b"cde")
    assert made.nbytes == HEADER.size + 2 + 3


def test_a_put_is_not_a_tombstone():
    assert not Record(key=b"k", sequence=1, value=b"v").tombstone


def test_a_delete_is():
    assert Record(key=b"k", sequence=1, kind=DELETE).tombstone


def test_a_record_orders_by_key_then_sequence():
    assert Record(key=b"k", sequence=5).order == (b"k", -5)


def test_a_record_summarises():
    assert Record(key=b"k", sequence=1, value=b"v").as_dict()["sequence"] == 1


def test_a_record_prints_its_key_and_sequence():
    assert str(Record(key=b"k", sequence=7, value=b"v")) == "k@7"


def test_a_tombstone_says_so_when_printed():
    assert "deleted" in str(Record(key=b"k", sequence=7, kind=DELETE))


def test_an_empty_key_raises():
    with pytest.raises(ConfigError):
        Record(key=b"", sequence=1)


def test_a_negative_sequence_raises():
    with pytest.raises(ConfigError):
        Record(key=b"k", sequence=-1)


def test_an_unknown_kind_raises():
    with pytest.raises(ConfigError):
        Record(key=b"k", sequence=1, kind=7)


def test_a_tombstone_with_a_value_raises():
    with pytest.raises(ConfigError):
        Record(key=b"k", sequence=1, kind=DELETE, value=b"v")


def test_an_oversized_value_raises():
    with pytest.raises(TooLarge):
        Record(key=b"k", sequence=1, value=b"x" * (MAX_VALUE + 1))


def test_an_empty_value_is_allowed():
    assert Record(key=b"k", sequence=1, value=b"").value == b""


def test_a_merge_carries_a_value():
    assert Record(key=b"k", sequence=1, kind=MERGE, value=b"+1").value == b"+1"


def test_decoding_returns_the_offset():
    raw = Record(key=b"k", sequence=1, value=b"v").encode()
    assert decode(raw)[1] == len(raw)


def test_decoding_an_empty_buffer_raises():
    with pytest.raises(BadFormat):
        decode(b"")


def test_decoding_a_header_without_a_body_raises():
    with pytest.raises(BadFormat):
        decode(HEADER.pack(1, PUT, 10, 10))


def test_decoding_all_of_nothing_is_nothing():
    assert decode_all(b"") == []


def test_decoding_all_returns_every_record():
    made = [Record(key=b"a", sequence=1), Record(key=b"b", sequence=2, value=b"x")]
    assert decode_all(b"".join(one.encode() for one in made)) == made


def test_newest_first_sorts_by_key():
    made = [Record(key=b"b", sequence=1), Record(key=b"a", sequence=1)]
    assert [one.key for one in newest_first(made)] == [b"a", b"b"]


def test_newest_first_sorts_versions_downward():
    made = [Record(key=b"a", sequence=1), Record(key=b"a", sequence=9)]
    assert [one.sequence for one in newest_first(made)] == [9, 1]


def test_the_live_view_takes_the_newest():
    made = [
        Record(key=b"a", sequence=1, value=b"old"),
        Record(key=b"a", sequence=5, value=b"new"),
    ]
    assert live(made)[b"a"].value == b"new"


def test_the_live_view_drops_deleted_keys():
    made = [
        Record(key=b"a", sequence=1, value=b"v"),
        Record(key=b"a", sequence=5, kind=DELETE),
    ]
    assert b"a" not in live(made)


def test_a_delete_followed_by_a_write_is_live_again():
    made = [
        Record(key=b"a", sequence=1, value=b"v"),
        Record(key=b"a", sequence=5, kind=DELETE),
        Record(key=b"a", sequence=9, value=b"back"),
    ]
    assert live(made)[b"a"].value == b"back"


def test_the_live_view_of_nothing_is_empty():
    assert live([]) == {}


def test_the_live_view_keeps_untouched_keys():
    made = [Record(key=b"a", sequence=1, value=b"v"), Record(key=b"b", sequence=2, value=b"w")]
    assert set(live(made)) == {b"a", b"b"}


def test_the_header_is_small():
    assert HEADER.size < 32


def test_there_are_three_kinds():
    assert len(KINDS) == 3
