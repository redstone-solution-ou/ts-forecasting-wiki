# MOIRAI: Unified Training of Universal Time Series Forecasting Transformers

> **Short name:** `moirai` · **arXiv:** [2402.02592](https://arxiv.org/abs/2402.02592) · **PDF:** [local](../../papers/moirai_2402.02592.pdf) · **Date:** 2024-02 · **Venue:** ICML 2024

**Authors:** Gerald Woo, Chenghao Liu, Akshat Kumar et al. (Salesforce)

## Abstract
MOIRAI introduces a unified masked-encoder transformer trained on LOTSA, a curated open corpus of roughly 27B observations spanning nine domains. The model tackles variable frequencies, variable number of variates, and probabilistic outputs in a single architecture.

## Key contributions
- Release of LOTSA, one of the largest open TS pretraining corpora.
- Multi-patch-size input projection layers for explicit frequency specialization.
- Any-variate attention supporting variable multivariate inputs without architectural changes.
- Mixture-of-Student-t output head producing probabilistic forecasts with robust tails.
- Strong zero-shot results on Monash and GIFT-Eval.

## Architecture at a glance
MOIRAI is an encoder-only transformer that patches each input series with a frequency-dependent patch size. Tokens from multiple variates are flattened into a single sequence, and any-variate attention lets the model condition each token on arbitrary others. A mixture-of-Student-t head outputs the forecast distribution.

## Why it matters
MOIRAI was one of the first TS foundation models to fully commit to open, large-scale pretraining and to solve the variable-frequency, variable-variate problem inside one architecture, setting a standard for universal TS forecasters.

## In the knowledge graph
- **Cluster:** [Masked-encoder / encoder-decoder TS-FMs](../foundation-models/taxonomy.md#cluster-2-masked-encoder--encoder-decoder-ts-fms)
- **Architecture family:** [Masked encoder](../architectures/masked-encoder.md)
- **Related concepts:** [patch tokenization](../concepts/patch-tokenization.md), [probabilistic forecasting](../concepts/probabilistic-forecasting.md), [zero-shot forecasting](../concepts/zero-shot-forecasting.md), [multi-task universal](../concepts/multi-task-universal.md)
- **Dataset / corpus:** [LOTSA](../datasets-benchmarks/lotsa.md)
- **See also:** [moirai-moe](./moirai-moe.md), [chronos](./chronos.md), [moment](./moment.md)
