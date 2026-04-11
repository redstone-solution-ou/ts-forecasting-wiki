# Mamba4Cast: Efficient Zero-Shot Time Series Forecasting with State Space Models

> **Short name:** `mamba4cast` · **arXiv:** [2410.09385](https://arxiv.org/abs/2410.09385) · **PDF:** [local](../../papers/mamba4cast_2410.09385.pdf) · **Date:** 2024-10 · **Venue:** preprint

**Authors:** Sathya Kamesh Bhethanabhotla, Omar Swelam, Julien Siems et al.

## Abstract
Mamba4Cast replaces transformer backbones with a Mamba state-space model and trains it entirely on synthetic data in the Prior-data Fitted Network style. The resulting zero-shot forecaster produces a full horizon in a single forward pass and scales gracefully with horizon length.

## Key contributions
- Mamba / SSM backbone as an alternative to transformer TS foundation models.
- Purely synthetic pretraining via Prior-data Fitted Networks (PFN).
- Single-pass horizon generation with no autoregressive rollout.
- Substantially lower inference latency and better scaling with horizon length.

## Architecture at a glance
Mamba4Cast stacks Mamba selective state-space blocks that process input patches in linear time. A PFN-style training pipeline generates large quantities of synthetic series from prior distributions, and the model learns to map a history directly to the full forecast horizon in one forward pass.

## Why it matters
Mamba4Cast demonstrates that competitive zero-shot TS forecasting is achievable without transformers and without real-data pretraining, combining efficient SSMs with PFN-style synthetic training to reach favorable accuracy-latency trade-offs.

## In the knowledge graph
- **Cluster:** [Lightweight / non-transformer FMs](../foundation-models/taxonomy.md#cluster-5-lightweight--non-transformer-fms)
- **Architecture family:** [Lightweight non-transformer](../architectures/lightweight-non-transformer.md)
- **Related concepts:** [synthetic data augmentation](../concepts/synthetic-data-augmentation.md), [zero-shot forecasting](../concepts/zero-shot-forecasting.md), [in-context learning](../concepts/in-context-learning.md)
- **See also:** [ttm](./ttm.md), [timesfm](./timesfm.md)
