from __future__ import annotations

import io

from store import ledger
from store.amplification import table as amplification_table
from store.compaction import compare_the_policies
from store.eval.latency import compare_the_utilisations
from store.eval.readpath import sweep as readpath_sweep
from store.eval.run import table as workload_table
from store.eval.writepath import sweep as writepath_sweep
from store.eval.writepath import worst_case_lost
from store.shard import compare_the_growth

# The package's findings rendered as text, which is where findings go to be read.
#
# Everything here is formatting. The numbers come from the modules that measured them, the
# ledger supplies the claim counts, and the report's one design decision is that it renders
# from live measurements rather than pasted output, so a report that disagrees with the code
# cannot exist.


def _table(rows: list[dict], title: str) -> str:
    """One list of dicts as an aligned text table."""
    if not rows:
        return f"{title}\n  (no rows)\n"
    columns = list(rows[0])
    widths = {
        column: max(len(str(column)), *(len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }
    out = io.StringIO()
    out.write(f"{title}\n")
    header = "  ".join(str(column).ljust(widths[column]) for column in columns)
    out.write(f"  {header}\n")
    out.write(f"  {'-' * len(header)}\n")
    for row in rows:
        line = "  ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns)
        out.write(f"  {line}\n")
    return out.getvalue()


def claims_section() -> str:
    """The ledger, one line per module."""
    out = io.StringIO()
    counts = ledger.counts()
    out.write(
        f"claims: {counts['claims']} across {counts['modules']} modules, "
        f"{counts['failing']} failing\n"
    )
    for name, summary in ledger.claims().items():
        booleans = {claim: held for claim, held in summary.items() if isinstance(held, bool)}
        if not booleans:
            continue
        failing = [claim for claim, held in booleans.items() if not held]
        state = "ok" if not failing else f"FAILING: {', '.join(failing)}"
        out.write(f"  {name}: {len(booleans)} claims, {state}\n")
    return out.getvalue()


def render(full: bool = True) -> str:
    """The whole report."""
    out = io.StringIO()
    out.write("storage-engine measurement report\n")
    out.write("=" * 50 + "\n\n")
    out.write(claims_section())
    out.write("\n")
    if full:
        out.write(_table(compare_the_policies(), "compaction: levelled against tiered"))
        out.write("\n")
        out.write(_table(amplification_table(), "the three amplifications"))
        out.write("\n")
        out.write(_table(workload_table(), "workloads through the engine"))
        out.write("\n")
        out.write(_table(compare_the_growth(), "sharding: the cost of growing"))
        out.write("\n")
        out.write(_table(compare_the_utilisations(), "queueing: utilisation against waits"))
        out.write("\n")
        out.write(_table(list(readpath_sweep()), "the read path swept"))
        out.write("\n")
        honest = [
            {
                "sync_every": row["sync_every"],
                "flush_at": row["flush_at"],
                "charges": row["charges"],
                "worst_lost": worst_case_lost(row["sync_every"], row["flush_at"]),
                "replay": row["replay"],
                "flushes": row["flushes"],
            }
            for row in writepath_sweep()
        ]
        out.write(_table(honest, "the write path swept"))
    return out.getvalue()
