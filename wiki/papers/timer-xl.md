# Timer-XL: Long-Context Transformers for Unified Time Series Forecasting

> **Short name:** `timer-xl` · **arXiv:** [2410.04803](https://arxiv.org/abs/2410.04803) · **PDF:** [local](../../papers/timerxl_2410.04803.pdf) · **Date:** 2024-10 · **Venue:** ICLR 2025

**Authors:** Yong Liu, Guo Qin, Xiangdong Huang et al. (Tsinghua)

## Abstract
Timer-XL extends Timer's univariate generative pretraining to the multivariate setting through a universal TimeAttention mechanism with causal positional embeddings. The result is a long-context decoder-only transformer that handles univariate, multivariate, and covariate-informed forecasting in one formulation.

## Key contributions
- TimeAttention that generalizes autoregression to multivariate series via flattened-token sequences.
- Causal positional embeddings suitable for arbitrary variate arrangements.
- Unified handling of univariate, multivariate, and covariate-informed tasks.
- State-of-the-art results in both task-specific and zero-shot regimes.

## Architecture at a glance
Timer-XL keeps Timer's decoder-only backbone and next-token training objective but flattens multivariate patches into a single long token sequence. TimeAttention combined with carefully designed positional embeddings enforces causality along the time axis while allowing free attention across variates.

## Why it matters
Timer-XL shows that long-context decoder-only transformers can absorb multivariate and covariate structure without bolt-on components, unifying a broader class of forecasting problems than univariate TS foundation models.

## In the knowledge graph
- **Cluster:** [Decoder-only autoregressive TS-FMs](../foundation-models/taxonomy.md#cluster-1-decoder-only-autoregressive-ts-fms)
- **Architecture family:** [Decoder-only autoregressive](../architectures/decoder-only-autoregressive.md)
- **Related concepts:** [patch tokenization](../concepts/patch-tokenization.md), [multi-task universal](../concepts/multi-task-universal.md), [zero-shot forecasting](../concepts/zero-shot-forecasting.md)
- **See also:** [timer](./timer.md), [chronos-2](./chronos-2.md), [sundial](./sundial.md)
