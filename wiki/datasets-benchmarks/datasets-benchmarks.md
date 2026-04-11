# Datasets and Benchmarks

Modern time-series foundation models are defined as much by the corpora they train on and the benchmarks they are evaluated against as by their architectures. Pretraining corpora (LOTSA, Time Series Pile, Time-300B, TimeBench) dictate the domain and frequency coverage a model can learn; benchmarks (Monash, GIFT-Eval) provide the shared yardsticks by which zero-shot claims are compared across papers. Together they anchor the field's empirical progress.

This section describes the corpora and evaluation suites themselves. For head-to-head *results* on these benchmarks — which model wins on which suite, under which metric, with which caveats — see the dedicated [benchmarks/](../benchmarks/benchmarks.md) section.

## Sub-pages

- [Monash Archive](monash-archive.md) — the original multi-domain forecasting benchmark.
- [LOTSA](lotsa.md) — 27B-observation pretraining corpus assembled for [MOIRAI](../papers/moirai.md).
- [Time Series Pile](time-series-pile.md) — multi-task corpus assembled for [MOMENT](../papers/moment.md).
- [Time-300B](time-300b.md) — 300B-point 9-domain corpus assembled for [Time-MoE](../papers/time-moe.md).
- [TimeBench](timebench.md) — ~1T-point corpus assembled for [Sundial](../papers/sundial.md).
- [GIFT-Eval](gift-eval.md) — cross-cutting benchmark used by MOIRAI, [Chronos-2](../papers/chronos-2.md), [Moirai-MoE](../papers/moirai-moe.md), and others.

## Related wiki pages

- [../benchmarks/benchmarks.md](../benchmarks/benchmarks.md) — head-to-head leaderboard, state-of-the-art analysis, methodology caveats, and efficiency comparison.
- [../benchmarks/leaderboard.md](../benchmarks/leaderboard.md) — normalized result tables on GIFT-Eval, Monash, [Chronos](../papers/chronos.md) Benchmark II, fev-bench, and LTSF.
- [../benchmarks/state-of-the-art.md](../benchmarks/state-of-the-art.md) — which model wins for which regime.
