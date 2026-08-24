from __future__ import annotations

import pytest

from store import slotted as mod
from store.errors import ConfigError, NotFound, TooLarge
from store.slotted import PAGE_BYTES, Page


class TestInsert:
    def test_an_empty_record_is_refused(self):
        with pytest.raises(ConfigError):
            Page().insert(b"")

    def test_an_insert_reads_back(self):
        page = Page()
        slot = page.insert(b"hello")
        assert page.read(slot) == b"hello"

    def test_ids_are_sequential(self):
        page = Page()
        assert page.insert(b"a") == 0 and page.insert(b"b") == 1

    def test_records_grow_from_the_back(self):
        page = Page()
        page.insert(b"abc")
        assert bytes(page.payload[-3:]) == b"abc"

    def test_an_oversized_record_is_refused(self):
        with pytest.raises(TooLarge):
            Page().insert(b"x" * (PAGE_BYTES + 1))

    def test_the_page_fills_and_refuses(self):
        page = Page()
        with pytest.raises(TooLarge):
            for _ in range(100):
                page.insert(b"x" * 100)
        assert page.contiguous_free < 104


class TestReadDelete:
    def test_an_unknown_slot_raises(self):
        with pytest.raises(NotFound):
            Page().read(0)

    def test_a_deleted_slot_raises(self):
        page = Page()
        slot = page.insert(b"x")
        page.delete(slot)
        with pytest.raises(NotFound):
            page.read(slot)

    def test_a_double_delete_raises(self):
        page = Page()
        slot = page.insert(b"x")
        page.delete(slot)
        with pytest.raises(NotFound):
            page.delete(slot)

    def test_deleting_one_leaves_the_others(self):
        page = Page()
        keep = page.insert(b"keep")
        gone = page.insert(b"gone")
        page.delete(gone)
        assert page.read(keep) == b"keep"

    def test_deleted_bytes_become_reclaimable(self):
        page = Page()
        slot = page.insert(b"x" * 100)
        page.delete(slot)
        assert page.reclaimable == 100


class TestUpdate:
    def test_a_same_size_update_lands(self):
        page = Page()
        slot = page.insert(b"aaaa")
        page.update(slot, b"bbbb")
        assert page.read(slot) == b"bbbb"

    def test_a_shrinking_update_lands(self):
        page = Page()
        slot = page.insert(b"aaaaaaaa")
        page.update(slot, b"bb")
        assert page.read(slot) == b"bb"

    def test_a_growing_update_lands(self):
        page = Page()
        slot = page.insert(b"aa")
        page.update(slot, b"b" * 500)
        assert page.read(slot) == b"b" * 500

    def test_an_impossible_growth_is_refused_and_rolled_back(self):
        page = Page()
        slot = page.insert(b"small")
        filler = []
        with pytest.raises(TooLarge):
            for _ in range(100):
                filler.append(page.insert(b"x" * 100))
        with pytest.raises(TooLarge):
            page.update(slot, b"y" * 2000)
        assert page.read(slot) == b"small"


class TestCompact:
    def test_compaction_reports_its_recovery(self):
        page = Page()
        slots = [page.insert(b"x" * 50) for _ in range(10)]
        for slot in slots[::2]:
            page.delete(slot)
        assert page.compact() == 250

    def test_compaction_preserves_every_read(self):
        page = Page()
        slots = [page.insert(bytes([at]) * 30) for at in range(20)]
        for slot in slots[::4]:
            page.delete(slot)
        page.compact()
        for at, slot in enumerate(slots):
            if at % 4:
                assert page.read(slot) == bytes([at]) * 30

    def test_a_clean_page_compacts_to_nothing(self):
        page = Page()
        page.insert(b"x" * 40)
        assert page.compact() == 0


class TestMeasurements:
    def test_ids_survive_upheaval(self):
        assert mod.ids_survive_every_internal_upheaval()

    def test_compaction_matches_the_meter(self):
        assert mod.compaction_recovers_exactly_the_holes()

    def test_full_refuses_holey_compacts(self):
        assert mod.a_full_page_refuses_and_a_holey_page_compacts_first()

    def test_shrink_stays_growth_moves(self):
        assert mod.shrinking_updates_stay_in_place_and_growing_ones_move()

    def test_dead_ids_stay_dead(self):
        assert mod.a_deleted_id_stays_dead()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_five_claims(self):
        assert len(mod.summarise()) == 5
