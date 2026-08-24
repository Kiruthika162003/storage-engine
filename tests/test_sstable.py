from __future__ import annotations

import itertools

import pytest

from store import sstable as mod
from store.errors import BadFormat, ConfigError
from store.record import DELETE, PUT, Record
from store.sstable import FOOTER, MAGIC, Handle, Lookup, Table, probe, read_footer, write


def one(key: bytes, sequence: int = 1, value: bytes = b"v", kind: int = PUT) -> Record:
    return Record(key=key, sequence=sequence, kind=kind, value=value)


def run(count: int, value: int = 50, prefix: bytes = b"user:") -> list[Record]:
    return [
        one(prefix + f"{at:08d}".encode(), at + 1, bytes(value)) for at in range(count)
    ]


class TestWrite:
    def test_an_empty_run_is_refused(self):
        with pytest.raises(ConfigError):
            write([])

    def test_one_record_makes_one_block(self):
        assert len(write(run(1)).blocks) == 1

    def test_a_small_run_makes_one_block(self):
        assert len(write(run(10)).blocks) == 1

    def test_a_large_run_makes_many_blocks(self):
        assert len(write(run(20000)).blocks) > 300

    def test_every_record_is_kept(self):
        made = run(5000)
        assert write(made).records() == made

    def test_the_count_matches(self):
        assert write(run(3000)).count == 3000

    def test_a_smaller_block_makes_more_blocks(self):
        small = write(run(5000), block_bytes=1024)
        large = write(run(5000), block_bytes=16384)
        assert len(small.blocks) > len(large.blocks)

    def test_a_table_without_a_filter_has_none(self):
        assert write(run(100), filtered=False).filter is None

    def test_a_table_with_a_filter_has_one(self):
        assert write(run(100)).filter is not None

    def test_the_filter_holds_every_key(self):
        made = run(2000)
        table = write(made)
        assert all(table.filter.might_contain(record.key) for record in made)

    def test_the_handles_are_one_per_block(self):
        table = write(run(9000))
        assert len(table.handles) == len(table.blocks)

    def test_the_handles_are_in_order(self):
        table = write(run(9000))
        keys = [handle.last for handle in table.handles]
        assert keys == sorted(keys)

    def test_the_handle_offsets_are_contiguous(self):
        table = write(run(9000))
        assert all(
            table.handles[at].end == table.handles[at + 1].offset
            for at in range(len(table.handles) - 1)
        )

    def test_the_handle_counts_add_up(self):
        table = write(run(9000))
        assert sum(handle.count for handle in table.handles) == 9000

    def test_the_last_handle_carries_the_last_key(self):
        made = run(9000)
        assert write(made).handles[-1].last == made[-1].key

    def test_tombstones_are_written(self):
        made = [one(b"a", 1, b"", DELETE), one(b"b", 2)]
        assert write(made).records()[0].tombstone


class TestGet:
    def test_a_present_key_is_found(self):
        table = write(run(20000))
        assert table.get(b"user:00010000").sequence == 10001

    def test_every_key_is_found(self):
        made = run(4000)
        table = write(made)
        assert all(table.get(record.key) is not None for record in made)

    def test_a_key_below_the_range_is_not_found(self):
        assert write(run(2000)).get(b"aaa") is None

    def test_a_key_above_the_range_is_not_found(self):
        assert write(run(2000)).get(b"zzz") is None

    def test_a_key_inside_the_range_that_is_absent_is_not_found(self):
        made = [one(b"a"), one(b"c"), one(b"e")]
        assert write(made).get(b"d") is None

    def test_a_lookup_reads_one_block(self):
        table = write(run(20000))
        table.get(b"user:00010000")
        assert table.reads == 1

    def test_a_lookup_without_a_filter_reads_on_a_miss(self):
        table = write(run(20000), filtered=False)
        table.get(b"user:00010000x")
        assert table.reads == 1

    def test_a_lookup_with_a_filter_usually_skips_a_miss(self):
        table = write(run(20000))
        for at in range(1000):
            table.get(f"gone:{at:08d}".encode())
        assert table.skipped > 900

    def test_a_filtered_miss_reports_no_read(self):
        table = write(run(20000))
        table.get(b"gone")
        assert table.reads == 0

    def test_the_first_key_is_found(self):
        assert write(run(9000)).get(b"user:00000000") is not None

    def test_the_last_key_is_found(self):
        assert write(run(9000)).get(b"user:00008999") is not None

    def test_reads_accumulate(self):
        table = write(run(20000))
        table.get(b"user:00000100")
        table.get(b"user:00019000")
        assert table.reads == 2


class TestScan:
    def test_a_scan_gives_everything(self):
        made = run(3000)
        assert write(made).records() == made

    def test_a_scan_from_a_key_gives_the_tail(self):
        made = run(3000)
        assert list(write(made).scan(b"user:00001500")) == made[1500:]

    def test_a_scan_from_past_the_end_gives_nothing(self):
        assert list(write(run(3000)).scan(b"zzz")) == []

    def test_a_scan_from_before_the_start_gives_everything(self):
        made = run(3000)
        assert list(write(made).scan(b"aaa")) == made

    def test_a_scan_is_ordered(self):
        keys = [record.key for record in write(run(3000)).scan(b"user:00001000")]
        assert keys == sorted(keys)

    def test_a_short_scan_crosses_few_blocks(self):
        table = write(run(20000))
        list(itertools.islice(table.scan(b"user:00010000"), 50))
        assert table.reads <= 2

    def test_a_full_scan_crosses_every_block(self):
        table = write(run(20000))
        table.records()
        assert table.reads == len(table.blocks)


class TestRange:
    def test_the_first_key_is_the_lowest(self):
        assert write(run(3000)).first == b"user:00000000"

    def test_the_last_key_is_the_highest(self):
        assert write(run(3000)).last == b"user:00002999"

    def test_a_key_inside_the_range_is_held(self):
        assert write(run(3000)).holds(b"user:00001500")

    def test_a_key_below_the_range_is_not_held(self):
        assert not write(run(3000)).holds(b"aaa")

    def test_a_key_above_the_range_is_not_held(self):
        assert not write(run(3000)).holds(b"zzz")

    def test_the_boundaries_are_inclusive(self):
        table = write(run(3000))
        assert table.holds(table.first) and table.holds(table.last)


class TestShape:
    def test_a_table_with_no_blocks_is_refused(self):
        with pytest.raises(ConfigError):
            Table(blocks=[], handles=[])

    def test_a_mismatched_handle_count_is_refused(self):
        table = write(run(3000))
        with pytest.raises(ConfigError):
            Table(blocks=table.blocks, handles=table.handles[:-1])

    def test_nbytes_is_more_than_the_blocks(self):
        table = write(run(3000))
        assert table.nbytes > sum(block.nbytes for block in table.blocks)

    def test_the_index_is_a_small_part_of_the_file(self):
        table = write(run(20000))
        assert table.index_bytes < table.nbytes * 0.02

    def test_as_dict_carries_the_record_count(self):
        assert write(run(500)).as_dict()["records"] == 500

    def test_as_dict_carries_the_filter_size(self):
        assert write(run(500)).as_dict()["filter_bytes"] > 0

    def test_as_dict_reports_no_filter_when_there_is_none(self):
        assert write(run(500), filtered=False).as_dict()["filter_bytes"] == 0

    def test_as_dict_carries_the_read_counters(self):
        table = write(run(500))
        table.get(b"user:00000100")
        assert table.as_dict()["reads"] == 1

    def test_a_handle_end_is_offset_plus_length(self):
        made = Handle(offset=10, length=5, last=b"z", count=1)
        assert made.end == 15


class TestFooter:
    def test_the_footer_is_a_fixed_size(self):
        assert len(write(run(500)).footer()) == FOOTER.size

    def test_the_footer_survives_a_prefix(self):
        table = write(run(500))
        assert read_footer(b"\x00" * 900 + table.footer())["count"] == 500

    def test_the_footer_carries_the_block_count(self):
        table = write(run(9000))
        assert read_footer(table.footer())["blocks"] == len(table.blocks)

    def test_the_footer_carries_the_index_size(self):
        table = write(run(9000))
        assert read_footer(table.footer())["index"] == table.index_bytes

    def test_a_short_file_is_refused(self):
        with pytest.raises(BadFormat):
            read_footer(b"\x00" * 4)

    def test_a_wrong_magic_is_refused(self):
        table = write(run(500))
        broken = bytearray(table.footer())
        broken[-8:] = (MAGIC + 1).to_bytes(8, "little")
        with pytest.raises(BadFormat):
            read_footer(bytes(broken))

    def test_the_magic_is_what_the_module_says(self):
        assert read_footer(write(run(10)).footer())["count"] == 10


class TestProbe:
    def test_a_probe_counts_hits(self):
        table = write(run(3000))
        assert probe(table, [b"user:00000001", b"user:00000002"]).hits == 2

    def test_a_probe_counts_misses(self):
        table = write(run(3000))
        assert probe(table, [b"gone", b"also-gone"]).misses == 2

    def test_a_probe_counts_reads(self):
        table = write(run(20000), filtered=False)
        assert probe(table, [b"user:00000001"]).reads == 1

    def test_a_probe_counts_skips(self):
        table = write(run(20000))
        assert probe(table, [b"gone"]).skipped == 1

    def test_reads_per_miss_is_one_without_a_filter(self):
        table = write(run(20000), filtered=False)
        made = probe(table, [f"gone:{at:04d}".encode() for at in range(200)])
        assert made.reads_per_miss == 1.0

    def test_reads_per_miss_is_near_zero_with_a_filter(self):
        table = write(run(20000))
        made = probe(table, [f"gone:{at:04d}".encode() for at in range(200)])
        assert made.reads_per_miss < 0.05

    def test_reads_per_miss_survives_no_misses(self):
        assert Lookup(hits=1, misses=0, reads=1, skipped=0).reads_per_miss == 1.0

    def test_as_dict_carries_every_field(self):
        made = Lookup(hits=1, misses=2, reads=3, skipped=4).as_dict()
        assert set(made) == {"hits", "misses", "reads", "skipped", "reads_per_miss"}

    def test_a_probe_is_measured_against_the_starting_counters(self):
        table = write(run(20000), filtered=False)
        probe(table, [b"user:00000001"])
        assert probe(table, [b"user:00000002"]).reads == 1


class TestMeasurements:
    def test_a_lookup_reads_one_block(self):
        assert mod.a_lookup_reads_one_block_of_a_file_of_hundreds()

    def test_the_filter_removes_the_read(self):
        assert mod.the_filter_turns_a_miss_into_no_read_at_all()

    def test_the_range_answers_before_the_filter(self):
        assert mod.a_miss_outside_the_key_range_never_needed_the_filter()

    def test_the_index_is_small_and_the_filter_is_larger(self):
        assert mod.the_index_is_a_fraction_of_the_file_and_the_filter_is_not()

    def test_the_block_size_is_a_trade(self):
        assert mod.a_smaller_block_makes_a_bigger_index_and_a_cheaper_read()

    def test_a_scan_pays_per_block(self):
        assert mod.a_scan_costs_the_blocks_it_crosses_and_nothing_else()

    def test_the_footer_anchors_the_file(self):
        assert mod.the_footer_is_the_only_part_whose_position_is_known()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_seven_claims(self):
        assert len(mod.summarise()) == 7

    def test_the_block_size_table_has_six_rows(self):
        assert len(mod.compare_the_block_sizes(5000)) == 6

    def test_the_block_size_table_shrinks_the_index(self):
        rows = mod.compare_the_block_sizes(5000)
        sizes = [row["index_bytes"] for row in rows]
        assert sizes == sorted(sizes, reverse=True)

    def test_the_block_size_table_grows_the_block(self):
        rows = mod.compare_the_block_sizes(5000)
        means = [row["mean_block"] for row in rows]
        assert means == sorted(means)

    def test_the_filter_table_has_two_rows(self):
        assert len(mod.compare_the_filter(5000, 500)) == 2

    def test_the_filter_table_shows_the_saving(self):
        rows = mod.compare_the_filter(5000, 500)
        assert rows[0]["reads"] > rows[1]["reads"] * 10

    def test_the_unfiltered_row_skips_nothing(self):
        assert mod.compare_the_filter(5000, 500)[0]["skipped"] == 0

    def test_the_cached_table_is_shared(self):
        assert mod._table(1000) is mod._table(1000)

    def test_the_cached_run_is_sorted(self):
        keys = [record.key for record in mod._run(1000)]
        assert keys == sorted(keys)
