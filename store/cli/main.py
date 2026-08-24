from __future__ import annotations

import argparse
import json
import sys

from store import ledger
from store.eval.run import table as workload_table
from store.report import render
from store.verify import crashfuzz, differential, model, torn

# One entry point: measure, verify, report.
#
# The commands are the package's three verbs. measure runs the workload table, verify runs
# the checkers that need no oracle beyond themselves, and report renders everything. Each
# command prints json or text and exits nonzero on a failure, because the exit code is the
# only output a script reads.


def _measure(arguments: argparse.Namespace) -> int:
    """The workload table, as json or text."""
    rows = workload_table()
    if arguments.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            cells = ", ".join(f"{key}={value}" for key, value in row.items())
            print(cells)
    return 0


def _verify(arguments: argparse.Namespace) -> int:
    """Every checker, with the failing count as the exit code."""
    outcomes = {
        "model": model.sweep(runs=arguments.runs, steps=arguments.steps),
        "crash": crashfuzz.sweep(runs=arguments.runs, writes=arguments.steps),
        "differential": differential.run(arguments.steps * 2, 400, 1),
        "torn": torn.sweep(points=20),
        "ledger": ledger.counts(),
    }
    failing = (
        outcomes["model"]["failed"]
        + outcomes["crash"]["failed"]
        + (0 if outcomes["differential"]["clean"] else 1)
        + (0 if outcomes["torn"]["clean"] else 1)
        + outcomes["ledger"]["failing"]
    )
    print(json.dumps(outcomes, indent=2))
    return 1 if failing else 0


def _report(arguments: argparse.Namespace) -> int:
    """The full report to stdout or a file."""
    made = render(full=not arguments.short)
    if arguments.out:
        with open(arguments.out, "w", encoding="utf-8") as handle:
            handle.write(made)
        print(f"wrote {len(made)} characters to {arguments.out}")
    else:
        print(made)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The argument grammar."""
    parser = argparse.ArgumentParser(prog="store", description="measure the storage engine")
    commands = parser.add_subparsers(dest="command", required=True)
    measure = commands.add_parser("measure", help="run the workloads and print the meters")
    measure.add_argument("--json", action="store_true", help="print json instead of text")
    measure.set_defaults(run=_measure)
    verify = commands.add_parser("verify", help="run every checker")
    verify.add_argument("--runs", type=int, default=8, help="fuzz runs per checker")
    verify.add_argument("--steps", type=int, default=600, help="steps per fuzz run")
    verify.set_defaults(run=_verify)
    report = commands.add_parser("report", help="render the measurement report")
    report.add_argument("--out", default="", help="write to a file instead of stdout")
    report.add_argument("--short", action="store_true", help="claims only, no tables")
    report.set_defaults(run=_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse and dispatch."""
    arguments = build_parser().parse_args(argv)
    return arguments.run(arguments)


if __name__ == "__main__":
    sys.exit(main())
