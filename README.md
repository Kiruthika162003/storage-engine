# storage-engine

A single-node storage engine built as a set of measured claims. The
package implements a log-structured store end to end, write-ahead log,
memtable, sorted tables, bloom filters, compaction, manifest, crash
recovery, and then keeps going into everything a storage engine lives
next to: caches, quotas, queues, quantile sketches, failure detectors,
spatial keys, pagination, backfills.

The house rule is that every module ends in a `summarise()` of boolean
claims, and every claim is a sentence backed by a number the module
itself computes when asked. When a measurement disagreed with the
sentence, the sentence was rewritten to say what was measured, and the
docstring keeps the original wrong guess next to the truth. The ledger
(`store/ledger.py`) currently tracks 550 claims across 119 modules with
none failing, and the report (`python -m store.cli.main report`) renders
the live tables.

## Layout

- `store/` - the engine and the measured modules
- `store/verify/` - invariants, model checking, differential testing,
  crash fuzzing, torn writes, metamorphic properties, operation fuzzing
- `store/eval/` - workloads, scaling, latency, recovery, regression
  baselines, and the swept read and write paths
- `examples/` - eight runnable stories, from a bank to an outage
- `tests/` - 3247 tests, including every claim in the ledger

## Running

```
python -m pytest tests/
python -m store.cli.main report
python -m examples.ops
```

## Findings, briefly

Some of the sentences the numbers wrote: a larger compaction fan-out
writes more, not less. The bloom filter pays on present keys too, and
loses to a hot cache. Two half caches equal the whole one only if the
layers exclude each other. Averaging two p99s answers a question nobody
asked. The flame graph crowns the wrong operation under concurrency.
A backfill that can regress its index is corruption with a progress
bar. The z-order curve is only as good as its query planner. On a
quiet log, group commit's group size is a knob connected to nothing.

## Authorship

Written by Kiruthika Subramani in collaboration with Claude, Anthropic's
AI assistant.
