from __future__ import annotations

import pytest

from store import block as mod
from store.block import (
    BLOCK_BYTES,
    RESTART_INTERVAL,
    Block,
    Builder,
    Cost,
    Reader,
    build,
)
from store.errors import BadFormat, ConfigError
from store.record import DELETE, PUT, Record


def one(key: bytes, sequence: int = 1, value: bytes = b"v", kind: int = PUT) -> Record:
    return Record(key=key, sequence=sequence, kind=kind, value=value)


def run(keys: list[bytes], value: bytes = b"v") -> list[Record]:
    return [one(key, at + 1, value) for at, key in enumerate(keys)]


def letters(count: int, prefix: bytes = b"key") -> list[bytes]:
    return [prefix + f"{at:08d}".encode() for at in range(count)]


class TestBuilder:
    def test_one_record_makes_a_block(self):
        made = Builder()
        made.add(one(b"a"))
        assert made.finish().count == 1

    def test_an_empty_builder_will_not_close(self):
        with pytest.raises(ConfigError):
            Builder().finish()

    def test_the_first_record_is_always_a_restart(self):
        made = Builder()
        made.add(one(b"a"))
        assert made.restarts == [0]

    def test_a_repeated_key_is_refused(self):
        made = Builder()
        made.add(one(b"a"))
        with pytest.raises(ConfigError):
            made.add(one(b"a"))

    def test_a_key_out_of_order_is_refused(self):
        made = Builder()
        made.add(one(b"b"))
        with pytest.raises(ConfigError):
            made.add(one(b"a"))

    def test_a_zero_interval_is_refused(self):
        with pytest.raises(ConfigError):
            Builder(interval=0)

    def test_a_negative_interval_is_refused(self):
        with pytest.raises(ConfigError):
            Builder(interval=-4)

    def test_an_interval_of_one_shares_nothing(self):
        made = Builder(interval=1)
        for record in run(letters(20)):
            made.add(record)
        assert made.shared_bytes == 0

    def test_a_large_interval_shares_almost_everything(self):
        made = Builder(interval=1000)
        for record in run(letters(20)):
            made.add(record)
        assert made.shared_bytes > 0

    def test_the_restart_count_follows_the_interval(self):
        made = Builder(interval=4)
        for record in run(letters(20)):
            made.add(record)
        assert len(made.restarts) == 5

    def test_a_partial_run_still_gets_its_restart(self):
        made = Builder(interval=4)
        for record in run(letters(21)):
            made.add(record)
        assert len(made.restarts) == 6

    def test_key_bytes_counts_the_whole_key(self):
        made = Builder()
        made.add(one(b"abcdef"))
        assert made.key_bytes == 6

    def test_full_is_false_on_a_small_block(self):
        made = Builder()
        made.add(one(b"a"))
        assert not made.full

    def test_full_turns_true_past_the_size(self):
        made = Builder()
        for record in run(letters(400), value=b"x" * 40):
            made.add(record)
        assert made.full

    def test_full_is_measured_against_the_block_size(self):
        made = Builder()
        for record in run(letters(400), value=b"x" * 40):
            made.add(record)
        assert len(made.payload) >= BLOCK_BYTES

    def test_the_count_matches_what_went_in(self):
        made = Builder()
        for record in run(letters(37)):
            made.add(record)
        assert made.finish().count == 37

    def test_the_previous_key_is_the_last_one_added(self):
        made = Builder()
        for record in run(letters(5)):
            made.add(record)
        assert made.previous == letters(5)[-1]


class TestRoundTrip:
    def test_a_single_record_survives(self):
        made = run([b"a"])
        assert build(made).records() == made

    def test_many_records_survive(self):
        made = run(letters(500))
        assert build(made).records() == made

    def test_an_empty_value_survives(self):
        made = [one(b"a", value=b"")]
        assert build(made).records() == made

    def test_a_large_value_survives(self):
        made = [one(b"a", value=b"z" * 5000)]
        assert build(made).records()[0].value == b"z" * 5000

    def test_a_tombstone_survives(self):
        made = [one(b"a", kind=DELETE, value=b"")]
        assert build(made).records()[0].tombstone

    def test_the_sequence_survives(self):
        made = [one(b"a", sequence=2**40)]
        assert build(made).records()[0].sequence == 2**40

    def test_keys_with_no_common_prefix_survive(self):
        made = run([b"aaa", b"bbb", b"ccc"])
        assert build(made).records() == made

    def test_keys_that_are_prefixes_of_each_other_survive(self):
        made = run([b"a", b"ab", b"abc", b"abcd"])
        assert build(made).records() == made

    def test_binary_keys_survive(self):
        made = run([bytes([0, 1]), bytes([0, 2]), bytes([255, 0])])
        assert build(made).records() == made

    def test_every_interval_round_trips(self):
        made = run(letters(100))
        for interval in (1, 2, 3, 7, 16, 64, 500):
            assert build(made, interval=interval).records() == made

    def test_the_order_is_kept(self):
        made = run(letters(200))
        keys = [record.key for record in build(made).records()]
        assert keys == sorted(keys)

    def test_a_key_of_one_byte_survives(self):
        made = run([bytes([at]) for at in range(1, 256)])
        assert build(made).records() == made


class TestGet:
    def test_a_present_key_is_found(self):
        block = build(run(letters(200)))
        assert block.get(b"key00000100").sequence == 101

    def test_a_missing_key_below_the_range_is_not_found(self):
        block = build(run(letters(200)))
        assert block.get(b"aaa") is None

    def test_a_missing_key_above_the_range_is_not_found(self):
        block = build(run(letters(200)))
        assert block.get(b"zzz") is None

    def test_a_missing_key_inside_the_range_is_not_found(self):
        block = build(run([b"a", b"c", b"e"]))
        assert block.get(b"d") is None

    def test_the_first_key_is_found(self):
        block = build(run(letters(200)))
        assert block.get(b"key00000000") is not None

    def test_the_last_key_is_found(self):
        block = build(run(letters(200)))
        assert block.get(b"key00000199") is not None

    def test_every_key_is_found(self):
        keys = letters(300)
        block = build(run(keys))
        assert all(block.get(key) is not None for key in keys)

    def test_no_key_between_two_stored_keys_is_found(self):
        keys = [b"a" * at for at in range(1, 20)]
        block = build(run(keys))
        assert block.get(b"b") is None

    def test_a_get_works_at_every_interval(self):
        made = run(letters(100))
        for interval in (1, 5, 16, 200):
            assert build(made, interval=interval).get(b"key00000050") is not None


class TestScan:
    def test_a_scan_from_nothing_gives_everything(self):
        made = run(letters(50))
        assert list(build(made).scan()) == made

    def test_a_scan_from_the_middle_gives_the_tail(self):
        made = run(letters(50))
        assert list(build(made).scan(b"key00000025")) == made[25:]

    def test_a_scan_from_past_the_end_gives_nothing(self):
        made = run(letters(50))
        assert list(build(made).scan(b"zzz")) == []

    def test_a_scan_from_before_the_start_gives_everything(self):
        made = run(letters(50))
        assert list(build(made).scan(b"aaa")) == made

    def test_a_scan_from_a_missing_key_starts_at_the_next_one(self):
        made = run([b"a", b"c", b"e"])
        assert [record.key for record in build(made).scan(b"b")] == [b"c", b"e"]

    def test_a_scan_is_ordered(self):
        made = run(letters(120))
        keys = [record.key for record in build(made).scan(b"key00000060")]
        assert keys == sorted(keys)

    def test_a_scan_at_every_interval_agrees(self):
        made = run(letters(80))
        wanted = made[40:]
        for interval in (1, 3, 16, 100):
            assert list(build(made, interval=interval).scan(b"key00000040")) == wanted


class TestRestarts:
    def test_the_restart_below_the_first_key_is_the_first(self):
        block = build(run(letters(100)))
        assert block.restart_below(b"aaa") == block.restarts[0]

    def test_the_restart_below_the_last_key_is_the_last(self):
        block = build(run(letters(100)))
        assert block.restart_below(b"zzz") == block.restarts[-1]

    def test_every_restart_decodes_on_its_own(self):
        keys = letters(200)
        block = build(run(keys))
        for at, offset in enumerate(block.restarts):
            found, _, _ = block.decode_at(offset, b"")
            assert found.key == keys[at * RESTART_INTERVAL]

    def test_the_restart_below_a_restart_key_is_that_restart(self):
        keys = letters(200)
        block = build(run(keys))
        assert block.restart_below(keys[RESTART_INTERVAL]) == block.restarts[1]

    def test_the_restart_below_a_key_just_under_a_restart_is_the_one_before(self):
        keys = letters(200)
        block = build(run(keys))
        assert block.restart_below(keys[RESTART_INTERVAL][:-1]) == block.restarts[0]

    def test_an_interval_of_one_restarts_everywhere(self):
        block = build(run(letters(50)), interval=1)
        assert len(block.restarts) == 50

    def test_a_huge_interval_restarts_once(self):
        block = build(run(letters(50)), interval=1000)
        assert len(block.restarts) == 1


class TestBlockShape:
    def test_a_block_with_no_restarts_is_refused(self):
        with pytest.raises(ConfigError):
            Block(payload=b"x", restarts=(), count=1)

    def test_a_block_with_no_records_is_refused(self):
        with pytest.raises(ConfigError):
            Block(payload=b"x", restarts=(0,), count=0)

    def test_nbytes_counts_the_restart_array(self):
        block = build(run(letters(100)))
        assert block.nbytes > len(block.payload)

    def test_the_interval_is_the_records_per_restart(self):
        block = build(run(letters(160)))
        assert block.interval == 16.0

    def test_as_dict_carries_the_count(self):
        block = build(run(letters(30)))
        assert block.as_dict()["records"] == 30

    def test_as_dict_carries_the_bytes(self):
        block = build(run(letters(30)))
        assert block.as_dict()["bytes"] == block.nbytes

    def test_as_dict_gives_bytes_per_record(self):
        block = build(run(letters(30)))
        assert block.as_dict()["bytes_per_record"] > 0

    def test_a_truncated_payload_is_caught(self):
        block = build(run(letters(50)))
        broken = Block(payload=block.payload[:-3], restarts=block.restarts, count=50)
        with pytest.raises(BadFormat):
            broken.records()

    def test_an_impossible_shared_length_is_caught(self):
        block = build(run(letters(50)))
        payload = bytearray(block.payload)
        payload[block.restarts[1]] = 250
        broken = Block(payload=bytes(payload), restarts=block.restarts, count=50)
        with pytest.raises(BadFormat):
            broken.records()


class TestReader:
    def test_a_reader_counts_what_it_decodes(self):
        block = build(run(letters(100)))
        reader = Reader(block=block)
        reader.get(b"key00000099")
        assert 0 < reader.decoded <= RESTART_INTERVAL

    def test_a_reader_at_interval_one_decodes_one(self):
        block = build(run(letters(100)), interval=1)
        reader = Reader(block=block)
        reader.get(b"key00000099")
        assert reader.decoded == 1

    def test_a_reader_with_no_restarts_walks_the_block(self):
        block = build(run(letters(100)), interval=1000)
        reader = Reader(block=block)
        reader.get(b"key00000099")
        assert reader.decoded == 100

    def test_a_reader_stops_early_on_a_missing_key(self):
        block = build(run(letters(100)), interval=1000)
        reader = Reader(block=block)
        reader.get(b"key00000000a")
        assert reader.decoded < 100

    def test_a_reader_finds_the_same_record_the_block_does(self):
        block = build(run(letters(100)))
        assert Reader(block=block).get(b"key00000042") == block.get(b"key00000042")

    def test_a_reader_accumulates_across_lookups(self):
        block = build(run(letters(100)))
        reader = Reader(block=block)
        reader.get(b"key00000010")
        first = reader.decoded
        reader.get(b"key00000090")
        assert reader.decoded > first


class TestCost:
    def test_the_ratio_is_zero_when_nothing_is_shared(self):
        assert Cost(interval=1, bytes=100, saved=0, decoded=1).ratio == 0.0

    def test_the_ratio_is_a_half_when_half_is_shared(self):
        assert Cost(interval=1, bytes=100, saved=100, decoded=1).ratio == 0.5

    def test_the_ratio_survives_a_zero_block(self):
        assert Cost(interval=1, bytes=0, saved=0, decoded=0).ratio == 0.0

    def test_as_dict_carries_every_field(self):
        made = Cost(interval=4, bytes=10, saved=2, decoded=7).as_dict()
        assert set(made) == {"interval", "bytes", "saved", "ratio", "decoded"}


class TestMeasurements:
    def test_sharing_pays_on_sorted_keys(self):
        assert mod.prefix_sharing_pays_on_sorted_keys_and_not_on_hashed_ones()

    def test_the_interval_trades_space_for_the_scan(self):
        assert mod.the_restart_interval_trades_space_against_the_scan()

    def test_the_middle_of_the_curve_is_flat(self):
        assert mod.the_middle_of_the_interval_curve_is_flat()

    def test_the_binary_search_beats_the_walk(self):
        assert mod.a_binary_search_over_restarts_beats_the_scan_it_replaces()

    def test_nothing_is_shared_across_a_restart(self):
        assert mod.a_block_never_shares_across_a_restart()

    def test_the_value_dominates_once_it_is_large(self):
        assert mod.the_value_dominates_once_it_is_large()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_six_claims(self):
        assert len(mod.summarise()) == 6

    def test_the_interval_table_has_a_row_per_interval(self):
        assert len(mod.compare_the_intervals(1000)) == 9

    def test_the_interval_table_shrinks_monotonically(self):
        sizes = [row["bytes"] for row in mod.compare_the_intervals(1000)]
        assert sizes == sorted(sizes, reverse=True)

    def test_the_interval_table_grows_the_decode_count(self):
        reads = [row["decoded"] for row in mod.compare_the_intervals(1000)]
        assert reads[-1] > reads[0]

    def test_the_key_shape_table_has_two_rows(self):
        assert len(mod.compare_the_key_shapes(1000)) == 2

    def test_the_sorted_row_shares_more(self):
        rows = mod.compare_the_key_shapes(1000)
        assert rows[0]["saved"] > rows[1]["saved"]

    def test_measure_is_stable(self):
        assert mod.measure(1000, 8, "sorted") == mod.measure(1000, 8, "sorted")

    def test_random_keys_are_distinct(self):
        keys = mod._random_keys(5000)
        assert len(set(keys)) == 5000

    def test_random_keys_come_back_sorted(self):
        keys = mod._random_keys(5000)
        assert list(keys) == sorted(keys)

    def test_sorted_keys_share_a_prefix(self):
        keys = mod._sorted_keys(10)
        assert all(key.startswith(b"user:") for key in keys)
