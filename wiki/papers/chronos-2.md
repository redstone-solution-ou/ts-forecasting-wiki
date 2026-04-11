# Chronos-2: From Univariate to Universal Forecasting

> **Short name:** `chronos-2` · **arXiv:** [2510.15821](https://arxiv.org/abs/2510.15821) · **PDF:** [local](../../papers/chronos2_2510.15821.pdf) · **Date:** 2025-10 · **Venue:** preprint

**Authors:** Abdul Fatir Ansari, Oleksandr Shchur, Jonas Küken et al. (AWS)

## Abstract
Chronos-2 generalizes the Chronos recipe from univariate next-token prediction to a universal forecaster that natively handles univariate, multivariate, and covariate-informed tasks. A 120M-parameter encoder-only patch-based backbone uses group attention to perform in-context learning over collections of related series.

## Key contributions
- Group attention mechanism enabling in-context learning across related series and panels within a single forward pass.
- Unified interface covering univariate, multivariate, and covariate-informed forecasting tasks.
- Quantile multi-step decoder for direct probabilistic forecasts.
- Heavy synthetic-data pretraining and state-of-the-art results on fev-bench, GIFT-Eval, and Chronos Benchmark II.

## Architecture at a glance
Chronos-2 is a 120M-parameter encoder-only transformer operating on value patches rather than a discrete vocabulary. Group attention lets tokens attend within and across member series of a group, so covariates and related panel series become in-context signals. A quantile head predicts multiple horizons simultaneously without autoregressive rollout.

## Why it matters
Chronos-2 closes the gap between univariate TS foundation models and the multivariate, covariate-rich settings common in practice. By making related-series in-context learning a first-class mechanism, it unifies what used to require separate architectures or task-specific fine-tuning.

## In the knowledge graph
- **Cluster:** [Masked-encoder / encoder-decoder TS-FMs](../foundation-models/taxonomy.md#cluster-2-masked-encoder--encoder-decoder-ts-fms)
- **Architecture family:** [Masked encoder](../architectures/masked-encoder.md)
- **Related concepts:** [patch tokenization](../concepts/patch-tokenization.md), [in-context learning](../concepts/in-context-learning.md), [probabilistic forecasting](../concepts/probabilistic-forecasting.md), [multi-task universal](../concepts/multi-task-universal.md)
- **Dataset / corpus:** [GIFT-Eval](../datasets-benchmarks/gift-eval.md)
- **See also:** [chronos](./chronos.md), [moirai](./moirai.md), [timer-xl](./timer-xl.md)
