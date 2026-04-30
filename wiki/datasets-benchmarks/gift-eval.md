# GIFT-Eval

GIFT-Eval is a cross-cutting zero-shot evaluation benchmark that has become a shared yardstick for recent time-series foundation models. Unlike Monash, which functions primarily as a forecasting archive with its own canonical baselines, GIFT-Eval is explicitly framed as a foundation-model leaderboard across many datasets and domains.

## Overview

GIFT-Eval aggregates a broad set of time-series datasets into a unified evaluation protocol: fixed splits, fixed horizons, and a defined set of metrics that probabilistic and point forecasters can both report. The motivation is that as TS foundation models have proliferated — [TimesFM](../papers/timesfm.md), [Chronos](../papers/chronos.md), MOIRAI, Moirai-MoE, Chronos-2, [Time-MoE](../papers/time-moe.md), [Sundial](../papers/sundial.md), and others — papers have needed a common venue in which to compare zero-shot accuracy without each author picking a favorable subset of datasets. GIFT-Eval provides that venue.

It is now routinely used alongside Monash in the evaluation section of new TS foundation-model papers. MOIRAI reports GIFT-Eval numbers as part of its zero-shot evaluation. Moirai-MoE reports GIFT-Eval to show that sparse routing beats the dense Moirai baseline at a lower active-parameter cost. Chronos-2 positions itself as SOTA on fev-bench, GIFT-Eval, and Chronos Benchmark II simultaneously, using GIFT-Eval as the cross-paper reference point.

The practical effect is that a new TS foundation model is expected to publish GIFT-Eval numbers as a baseline of credibility, much as a new LLM is expected to publish MMLU, GSM8K, and similar.

## Composition

Per the GIFT-Eval paper (Aksu et al., arXiv:2410.10393), the benchmark
has two explicitly separate components:

- **Train/Test (evaluation)** — 23 datasets / ~144k series / ~177M
  observations across 7 domains (Econ/Fin, Energy, Healthcare, Nature,
  Sales, Transport, Web/CloudOps). Details in Table 13.
- **GIFT-Eval Pretrain** — 88 datasets (71 univariate + 17
  multivariate) / ~4.5M series / ~230B observations, curated to be
  **disjoint from the test split**. Available as
  `Salesforce/GiftEvalPretrain` on HuggingFace. Details in Table 14.

The 23 test datasets (Table 13):

| Domain | Test datasets |
|---|---|
| Econ/Fin | `M4 Yearly/Quarterly/Monthly/Weekly/Daily/Hourly`, `Hierarchical Sales` (Mancuso et al.) |
| Energy | `ETT1`, `ETT2` (Informer), `Solar` (LSTNet), `Electricity` (UCI) |
| Healthcare | `Hospital`, `COVID Deaths`, `US Births` (Monash) |
| Nature | `Jena Weather` (Autoformer), `Saugeen`, `Temperature Rain`, `KDD Cup 2018` (Monash) |
| Sales | `Restaurant` (Recruit Rest. Comp.), `Car Parts` (Monash) |
| Transport | `Loop Seattle`, `SZ-Taxi`, `M_DENSE` (LibCity) |
| Web/CloudOps | `BizITObs - Application / Service / L2C` (AutoMixer), `Bitbrains - Fast Storage / rnd` (Grid Workloads Archive) |

**The Web/CloudOps test subset contains zero Wikipedia-derived
series** — all entries are CloudOps telemetry. Wikipedia-derived data
(Wiki-Rolling, Kaggle Web Traffic Weekly, Extended Web Traffic) lives
only in GIFT-Eval **Pretrain**, not in the test split. See
[../benchmarks/wikipedia-pageviews-leakage.md](../benchmarks/wikipedia-pageviews-leakage.md).

## What you should expect from `GiftEvalPretrain` alone

The [Aksu et al. 2024](../papers/aksu-gift-eval.md) paper retrains the
entire `Moirai`-1 family from scratch on `GiftEvalPretrain` only (no
LOTSA, no synthetic, no auxiliary corpora) and reports the headline
numbers in Table 10. This makes those rows the **only published
reference for what training on `GiftEvalPretrain` exclusively buys
you** at the time of writing. Lower is better; both metrics are
relative to Seasonal Naive on the same series. Rank is the average
ranking across all 97 configs (out of the 17 evaluated baselines, plus
`Moirai-Leakage` in the leakage-audit appendix).

| Model                                    | Corpus                                                    | MAPE   | CRPS   | Rank  |
| ---------------------------------------- | --------------------------------------------------------- | ------ | ------ | ----- |
| **Moirai-Large** (retrained)             | `GiftEvalPretrain` (~240B obs) only                       | 0.864  | 0.593  | 5.99  |
| **Moirai-Base** (retrained)              | `GiftEvalPretrain` only                                   | 1.01   | 0.648  | 7.57  |
| **Moirai-Small** (retrained)             | `GiftEvalPretrain` only                                   | 0.882  | 0.642  | 7.81  |
| PatchTST (full-shot reference)           | per-dataset training                                      | 0.860  | 0.563  | 5.72  |
| iTransformer (full-shot reference)       | per-dataset training                                      | 0.985  | 0.626  | 6.37  |
| Chronos-Large (leaked)                   | Chronos public corpus (partial overlap)                   | 0.930  | 0.629  | 8.31  |
| TimesFM v1 (leaked)                      | TimesFM v1 corpus (partial overlap)                       | 1.25   | 0.683  | 8.55  |
| Seasonal Naive                           | —                                                         | 1.000  | 1.000  | 16.64 |

Source: Aksu et al. Table 10 (paper's own all-aggregate, 97 configs).
Note: this paper measures **MAPE**, not MASE. The public leaderboard
that successor papers report against later (`Sundial` Table 2,
`Chronos-2` Table 4) uses MASE-derived skill scores, so the values
here are not directly comparable to those tables — see
[../benchmarks/methodology-caveats.md](../benchmarks/methodology-caveats.md)
§2.

The takeaway: Moirai-Large pretrained on `GiftEvalPretrain` alone is
**statistically tied with full-shot PatchTST in aggregate Rank**
(5.99 vs 5.72) and beats every other foundation model evaluated in
the paper, despite using ~25× less data than LOTSA. So the floor for
"what does `GiftEvalPretrain` alone get you" is roughly *parity with
the best per-dataset deep-learning baseline*. Whether a more modern
architecture (Moirai-2.0 decoder-only, Chronos-2 group-attention,
Sundial flow-matching, Timer-S1 STP) trained on `GiftEvalPretrain`
alone would clear PatchTST is currently an open question — no paper
in the wiki has reported that experiment.

## Leakage-audit detail (§F.2 / Table 23)

The paper also reports the original LOTSA-trained Moirai checkpoints
under the alias `Moirai-Leakage` on the eight test datasets where
LOTSA and GIFT-Eval test overlap (`hierarchical_sales`,
`loop_seattle 5T/D/H`, `m_dense D/H`, `restaurant`, `sz_taxi 15T/H`).
The leakage benefit is **highly concentrated**: on `loop_seattle 5T`
medium-horizon MAPE the LOTSA-leaked Moirai-Large hits 0.33 vs the
retrained Moirai-Large at 0.85 (2.6× degradation when leakage is
removed); on `m_dense H` long-horizon MAPE the gap is 0.49 vs 1.69
(3.4×). On `hierarchical_sales` and `restaurant` the leakage delta is
≤0.02 either way. The dominant pattern matches the paper's text:
**leakage benefit grows with prediction length** and is concentrated
on a small subset of datasets where the LOTSA training portion
shares the exact series (not just the dataset family) with the
GIFT-Eval test split.

This is the reason the field treats the headline `Moirai_S/B/L`
numbers in this paper (the retrained versions, not `Moirai-Leakage`)
as the canonical clean Moirai-1 baseline.

## Key ideas / variants

- Cross-paper unified evaluation protocol for zero-shot TS forecasting.
- Fixed splits, fixed horizons, defined probabilistic and point metrics.
- Complements Monash as a foundation-model-specific leaderboard.
- Used in parallel with fev-bench and Chronos Benchmark II by recent papers.

## Papers that exemplify this (or use this)

- [Aksu et al. 2024](../papers/aksu-gift-eval.md) — the GIFT-Eval benchmark paper itself, including the §F.2 leakage audit and the only published `GiftEvalPretrain`-only Moirai numbers.
- [MOIRAI](../papers/moirai.md) — reports GIFT-Eval alongside Monash.
- [Moirai-MoE](../papers/moirai-moe.md) — GIFT-Eval results showing sparse routing gains.
- [Moirai 2.0](../papers/moirai-2.md) — uses `GiftEvalPretrain` as the public-data backbone of its 36M-series mixture; evaluates exclusively on GIFT-Eval.
- [Chronos-2](../papers/chronos-2.md) — SOTA on GIFT-Eval alongside fev-bench and Chronos Benchmark II.
- [Sundial](../papers/sundial.md) — reports GIFT-Eval as one of its primary suites.
- [Timer-S1](../papers/timer-s1.md) — current pre-trained-model SOTA on GIFT-Eval; explicitly scrubs GIFT-Eval leakage from TimeBench.

## Related wiki pages

- [Monash Archive](monash-archive.md)
- [Zero-shot forecasting](../concepts/zero-shot-forecasting.md)
- [LOTSA](lotsa.md)
- [../benchmarks/wikipedia-pageviews-leakage.md](../benchmarks/wikipedia-pageviews-leakage.md) — concrete audit: GIFT-Eval test has no Wikipedia series; GIFT-Eval Pretrain includes Wiki-Rolling + Kaggle Web Traffic (weekly + extended).
- [../benchmarks/rebuilding-timebench.md](../benchmarks/rebuilding-timebench.md) — how to assemble a TimeBench-style corpus that stays GIFT-Eval-clean.
