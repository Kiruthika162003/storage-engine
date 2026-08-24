from __future__ import annotations

import json

import pytest

from store import ledger
from store.cli.main import build_parser, main
from store.report import _table, claims_section, render


class TestLedger:
    def test_every_claiming_module_summarises(self):
        assert set(ledger.claims()) == set(ledger.CLAIMING)

    def test_the_flat_view_prefixes_the_module(self):
        assert all("." in name for name in ledger.flat())

    def test_the_flat_view_is_all_booleans(self):
        assert all(isinstance(held, bool) for held in ledger.flat().values())

    def test_nothing_is_failing(self):
        assert ledger.failures() == []

    def test_the_counts_add_up(self):
        counts = ledger.counts()
        assert counts["claims"] == len(ledger.flat())
        assert counts["failing"] == len(ledger.failures())

    def test_the_ledger_spans_the_package(self):
        assert ledger.counts()["modules"] >= 30

    def test_the_claims_are_plentiful(self):
        assert ledger.counts()["claims"] >= 150


class TestReport:
    def test_a_table_renders_its_rows(self):
        made = _table([{"a": 1, "b": "x"}, {"a": 22, "b": "yy"}], "demo")
        assert "demo" in made and "22" in made and "yy" in made

    def test_an_empty_table_says_so(self):
        assert "(no rows)" in _table([], "empty")

    def test_the_claims_section_names_every_module(self):
        made = claims_section()
        assert all(name in made for name in ledger.CLAIMING)

    def test_the_claims_section_reports_zero_failing(self):
        assert "0 failing" in claims_section()

    def test_the_short_report_skips_the_tables(self):
        assert "levelled against tiered" not in render(full=False)

    def test_the_full_report_carries_the_tables(self):
        made = render(full=True)
        assert "levelled against tiered" in made
        assert "three amplifications" in made
        assert "queueing" in made


class TestCli:
    def test_the_parser_requires_a_command(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_measure_returns_success(self, capsys):
        assert main(["measure"]) == 0
        assert "read_heavy" in capsys.readouterr().out

    def test_measure_json_is_json(self, capsys):
        assert main(["measure", "--json"]) == 0
        assert isinstance(json.loads(capsys.readouterr().out), list)

    def test_report_short_prints_claims(self, capsys):
        assert main(["report", "--short"]) == 0
        assert "claims:" in capsys.readouterr().out

    def test_report_writes_a_file(self, tmp_path, capsys):
        out = tmp_path / "report.txt"
        assert main(["report", "--short", "--out", str(out)]) == 0
        assert out.read_text(encoding="utf-8").startswith("storage-engine")

    def test_verify_returns_success(self, capsys):
        assert main(["verify", "--runs", "2", "--steps", "150"]) == 0
        made = json.loads(capsys.readouterr().out)
        assert made["ledger"]["failing"] == 0
