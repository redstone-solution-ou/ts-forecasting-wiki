# GIFT-Eval

GIFT-Eval is a cross-cutting zero-shot evaluation benchmark that has become a shared yardstick for recent time-series foundation models. Unlike Monash, which functions primarily as a forecasting archive with its own canonical baselines, GIFT-Eval is explicitly framed as a foundation-model leaderboard across many datasets and domains.

## Overview

GIFT-Eval aggregates a broad set of time-series datasets into a unified evaluation protocol: fixed splits, fixed horizons, and a defined set of metrics that probabilistic and point forecasters can both report. The motivation is that as TS foundation models have proliferated — [TimesFM](../papers/timesfm.md), [Chronos](../papers/chronos.md), MOIRAI, Moirai-MoE, Chronos-2, [Time-MoE](../papers/time-moe.md), [Sundial](../papers/sundial.md), and others — papers have needed a common venue in which to compare zero-shot accuracy without each author picking a favorable subset of datasets. GIFT-Eval provides that venue.

It is now routinely used alongside Monash in the evaluation section of new TS foundation-model papers. MOIRAI reports GIFT-Eval numbers as part of its zero-shot evaluation. Moirai-MoE reports GIFT-Eval to show that sparse routing beats the dense Moirai baseline at a lower active-parameter cost. Chronos-2 positions itself as SOTA on fev-bench, GIFT-Eval, and Chronos Benchmark II simultaneously, using GIFT-Eval as the cross-paper reference point.

The practical effect is that a new TS foundation model is expected to publish GIFT-Eval numbers as a baseline of credibility, much as a new LLM is expected to publish MMLU, GSM8K, and similar.

## Key ideas / variants

- Cross-paper unified evaluation protocol for zero-shot TS forecasting.
- Fixed splits, fixed horizons, defined probabilistic and point metrics.
- Complements Monash as a foundation-model-specific leaderboard.
- Used in parallel with fev-bench and Chronos Benchmark II by recent papers.

## Papers that exemplify this (or use this)

- [MOIRAI](../papers/moirai.md) — reports GIFT-Eval alongside Monash.
- [Moirai-MoE](../papers/moirai-moe.md) — GIFT-Eval results showing sparse routing gains.
- [Chronos-2](../papers/chronos-2.md) — SOTA on GIFT-Eval alongside fev-bench and Chronos Benchmark II.

## Related wiki pages

- [Monash Archive](monash-archive.md)
- [Zero-shot forecasting](../concepts/zero-shot-forecasting.md)
- [LOTSA](lotsa.md)
- [../benchmarks/wikipedia-pageviews-leakage.md](../benchmarks/wikipedia-pageviews-leakage.md) — concrete audit: GIFT-Eval test has no Wikipedia series; GIFT-Eval Pretrain includes Wiki-Rolling + Kaggle Web Traffic (weekly + extended).
- [../benchmarks/rebuilding-timebench.md](../benchmarks/rebuilding-timebench.md) — how to assemble a TimeBench-style corpus that stays GIFT-Eval-clean.
