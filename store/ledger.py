from __future__ import annotations

import importlib

# Every module's claims, gathered into one place and run as one suite.
#
# Each module in the package ends with a summarise function that runs its own claims. The
# ledger walks them all, which gives the package a single question with a single answer: is
# everything this code claims about itself still true. A regression that flips any claim
# anywhere turns exactly one flag False, with the module and claim named.

CLAIMING = (
    "store.keys",
    "store.record",
    "store.wal",
    "store.memtable",
    "store.bloom",
    "store.block",
    "store.sstable",
    "store.iterator",
    "store.compaction",
    "store.manifest",
    "store.cache",
    "store.btree",
    "store.mvcc",
    "store.engine",
    "store.txn",
    "store.amplification",
    "store.metrics",
    "store.varint",
    "store.checksum",
    "store.ttl",
    "store.batch",
    "store.shard",
    "store.mergeop",
    "store.backup",
    "store.stall",
    "store.secondary",
    "store.compress",
    "store.rangedel",
    "store.scrub",
    "store.cuckoo",
    "store.hll",
    "store.composite",
    "store.radix",
    "store.admission",
    "store.bufferpool",
    "store.prefetch",
    "store.hotcold",
    "store.manifest_compact",
    "store.lockmanager",
    "store.hashlog",
    "store.externalsort",
    "store.zonemap",
    "store.timekey",
    "store.arena",
    "store.vlog",
    "store.predicate",
    "store.planner",
    "store.bulkload",
    "store.writebatch",
    "store.wheel",
    "store.snapshotscan",
    "store.topk",
    "store.reservoir",
    "store.failpoints",
    "store.interval",
    "store.btreebulk",
    "store.quota",
    "store.columnar",
    "store.dictionary",
    "store.timeseries",
    "store.bitmap",
    "store.eval.findings",
    "store.eval.recovery",
    "store.eval.scaling",
    "store.eval.latency",
    "store.verify.metamorphic",
)


def claims() -> dict[str, dict]:
    """Every module's summary, keyed by module."""
    made = {}
    for name in CLAIMING:
        module = importlib.import_module(name)
        made[name] = module.summarise()
    return made


def flat() -> dict[str, bool]:
    """Every boolean claim flattened to module.claim, for a single assertion.

    The earliest modules mix facts into their summaries, counts and constants alongside the
    booleans, so the ledger takes only the entries that are claims. A count is not a claim:
    it cannot fail, and a suite padded with entries that cannot fail reads as stronger than
    it is.
    """
    made = {}
    for name, summary in claims().items():
        for claim, held in summary.items():
            if isinstance(held, bool):
                made[f"{name}.{claim}"] = held
    return made


def failures() -> list[str]:
    """The claims that do not hold, empty when the package is telling the truth."""
    return [name for name, held in flat().items() if not held]


def counts() -> dict:
    """The ledger in three numbers."""
    held = flat()
    return {
        "modules": len(CLAIMING),
        "claims": len(held),
        "failing": sum(1 for value in held.values() if not value),
    }
