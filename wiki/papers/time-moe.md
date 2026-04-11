# Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts

> **Short name:** `time-moe` · **arXiv:** [2409.16040](https://arxiv.org/abs/2409.16040) · **PDF:** [local](../../papers/timemoe_2409.16040.pdf) · **Date:** 2024-09 · **Venue:** ICLR 2025 Spotlight

**Authors:** Xiaoming Shi, Shiyu Wang, Yuqi Nie et al.

## Abstract
Time-MoE introduces the first billion-parameter-class time-series foundation model by combining a sparse mixture-of-experts decoder-only transformer with the large Time-300B pretraining corpus. It shows empirical neural scaling behavior for time series under MoE scaling.

## Key contributions
- First billion-parameter TS foundation model, with the largest variant at 2.4B parameters.
- Release of Time-300B, roughly 300B points across nine domains.
- Sparse MoE decoder-only transformer with only selected experts activated per token.
- Context length up to 4096 tokens, enabling long-history forecasting.
- Empirical demonstration of scaling laws for TS foundation models.

## Architecture at a glance
Time-MoE is a decoder-only transformer whose feed-forward blocks are replaced with sparse mixture-of-experts layers. Each token is routed to a small subset of experts, keeping activated parameters bounded even as total capacity grows. Training is autoregressive next-patch prediction on Time-300B.

## Why it matters
Time-MoE is the landmark result for scaling time-series foundation models beyond the hundred-million-parameter regime. It provides the clearest published evidence so far that neural scaling laws translate from language to time series when capacity is added via MoE.

## In the knowledge graph
- **Cluster:** [Mixture-of-experts TS-FMs](../foundation-models/taxonomy.md#cluster-3-mixture-of-experts-ts-fms)
- **Architecture family:** [Mixture of experts](../architectures/mixture-of-experts.md)
- **Related concepts:** [scaling laws](../concepts/scaling-laws.md), [patch tokenization](../concepts/patch-tokenization.md), [zero-shot forecasting](../concepts/zero-shot-forecasting.md)
- **Dataset / corpus:** [Time-300B](../datasets-benchmarks/time-300b.md)
- **See also:** [moirai-moe](./moirai-moe.md), [timesfm](./timesfm.md), [timer](./timer.md)
