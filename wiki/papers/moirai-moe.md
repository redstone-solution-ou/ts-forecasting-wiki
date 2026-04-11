# Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts

> **Short name:** `moirai-moe` · **arXiv:** [2410.10469](https://arxiv.org/abs/2410.10469) · **PDF:** [local](../../papers/moiraimoe_2410.10469.pdf) · **Date:** 2024-10 · **Venue:** preprint

**Authors:** Xu Liu, Juncheng Liu, Gerald Woo et al. (Salesforce)

## Abstract
Moirai-MoE replaces MOIRAI's frequency-specific input projection layers with a single unified projection and moves specialization into sparse mixture-of-experts blocks inside the transformer. Expert routing happens at the token level rather than the frequency level, letting the model learn its own specialization structure.

## Key contributions
- Unified input projection replacing hand-designed multi-patch-size frequency projections.
- Token-level sparse MoE routing inside the transformer, delegating specialization to experts.
- State-of-the-art in-distribution and zero-shot performance across 39 datasets.
- Reduced activated compute per token compared with dense MOIRAI-Large.

## Architecture at a glance
Moirai-MoE keeps MOIRAI's masked-encoder backbone and any-variate attention but injects sparse mixture-of-experts feed-forward blocks. Each token is routed to a small subset of experts, so the total parameter count grows while per-token compute stays bounded. The frequency-specific projection heads are removed and replaced with one shared projection.

## Why it matters
Moirai-MoE shows that specialization in TS foundation models is better handled by learned routing than by hand-designed frequency heuristics. It delivers stronger accuracy than dense MOIRAI at lower activated compute, making MoE a natural scaling axis for universal TS models.

## In the knowledge graph
- **Cluster:** [Mixture-of-experts TS-FMs](../foundation-models/taxonomy.md#cluster-3-mixture-of-experts-ts-fms)
- **Architecture family:** [Mixture of experts](../architectures/mixture-of-experts.md)
- **Related concepts:** [patch tokenization](../concepts/patch-tokenization.md), [zero-shot forecasting](../concepts/zero-shot-forecasting.md), [scaling laws](../concepts/scaling-laws.md)
- **Dataset / corpus:** [LOTSA](../datasets-benchmarks/lotsa.md)
- **See also:** [moirai](./moirai.md), [time-moe](./time-moe.md)
