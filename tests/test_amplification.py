from __future__ import annotations

from store import amplification as mod
from store.amplification import Point, btree_point, levelled_point, table, tiered_point


class TestPoints:
    def test_the_levelled_point_names_itself(self):
        assert levelled_point().design == "levelled"

    def test_the_tiered_point_names_itself(self):
        assert tiered_point().design == "tiered"

    def test_the_btree_point_names_itself(self):
        assert btree_point().design == "btree"

    def test_every_axis_is_at_least_one(self):
        for point in (levelled_point(), tiered_point(), btree_point()):
            assert point.write >= 1.0 or point.design == "tiered"
            assert point.read >= 1.0
            assert point.space >= 1.0

    def test_the_points_are_cached(self):
        assert levelled_point() is levelled_point()

    def test_as_dict_carries_every_axis(self):
        made = Point(design="x", write=1.0, read=2.0, space=3.0).as_dict()
        assert set(made) == {
            "design",
            "write_amplification",
            "read_amplification",
            "space_amplification",
        }


class TestShape:
    def test_tiered_writes_least(self):
        assert tiered_point().write < min(levelled_point().write, btree_point().write)

    def test_the_tree_writes_most(self):
        assert btree_point().write > max(levelled_point().write, tiered_point().write)

    def test_tiered_reads_most(self):
        assert tiered_point().read > max(levelled_point().read, btree_point().read)

    def test_the_tree_holds_least(self):
        assert btree_point().space < min(levelled_point().space, tiered_point().space)

    def test_tiered_holds_most(self):
        assert tiered_point().space > levelled_point().space

    def test_the_tree_and_levelled_tie_on_reads(self):
        assert abs(btree_point().read - levelled_point().read) < 0.1


class TestMeasurements:
    def test_every_design_sits_at_a_different_corner(self):
        assert mod.every_design_sits_at_a_different_corner()

    def test_tiered_pays_twice(self):
        assert mod.tiered_wins_writes_and_pays_twice()

    def test_the_tree_wins_space_not_reads(self):
        assert mod.the_tree_wins_space_not_reads_which_was_not_the_guess()

    def test_every_claim_holds(self):
        assert all(mod.summarise().values())

    def test_the_summary_names_three_claims(self):
        assert len(mod.summarise()) == 3

    def test_the_table_has_three_rows(self):
        assert len(table()) == 3

    def test_the_table_rows_are_the_points(self):
        assert table()[0] == levelled_point().as_dict()
