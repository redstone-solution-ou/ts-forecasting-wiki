# Benchmarks — Head-to-Head Results

This section is the *performance* layer of the wiki. While
[datasets-benchmarks/](../datasets-benchmarks/README.md) describes the
corpora and evaluation suites themselves, this section reads across
them: which models win on which suite, under which metric, and with
which caveats.

Every numeric claim on these pages is attributed to a specific paper,
table or figure. Where numbers are not directly comparable (different
context lengths, normalization, or splits), the caveats page says so
rather than forcing a false ranking.

## Sub-pages

- [leaderboard.md](leaderboard.md) — normalized head-to-head tables on
  Monash, Chronos Benchmark II, GIFT-Eval, fev-bench, and the LTSF
  long-horizon suite.
- [state-of-the-art.md](state-of-the-art.md) — narrative "which model
  wins where," organized by forecasting regime, ending in
  recommendations by use case.
- [methodology-caveats.md](methodology-caveats.md) — how to read
  TS-FM benchmark results: leakage, normalization, metric semantics,
  aggregation, and contested findings.
- [efficiency-and-cost.md](efficiency-and-cost.md) — the second axis
  beyond accuracy: parameters, training compute, inference latency,
  memory, and CPU deployability.

## Related wiki pages

- [../datasets-benchmarks/README.md](../datasets-benchmarks/README.md) —
  descriptions of the corpora and eval suites themselves.
- [../foundation-models/taxonomy.md](../foundation-models/taxonomy.md) —
  the cluster map of TS-FMs referenced throughout this section.
