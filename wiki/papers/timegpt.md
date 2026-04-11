# TimeGPT-1

> **Short name:** `timegpt` · **arXiv:** [2310.03589](https://arxiv.org/abs/2310.03589) · **PDF:** [local](../../papers/timegpt_2310.03589.pdf) · **Date:** 2023-10 · **Venue:** preprint

**Authors:** Azul Garza, Cristian Challu, Max Mergenthaler-Canseco (Nixtla)

## Abstract
TimeGPT-1 is presented as the first end-to-end commercial time-series foundation model delivered as an API. It is trained on a large multi-domain proprietary corpus and produces zero-shot forecasts along with conformal-prediction uncertainty intervals, outperforming classical, ML, and deep-learning baselines across many datasets.

## Key contributions
- First commercially deployed TS foundation model exposed through an API.
- Large multi-domain pretraining corpus aggregated from varied sources.
- Zero-shot forecasts reported to beat classical, ML, and deep-learning baselines out of the box.
- Conformal prediction for calibrated uncertainty intervals.
- Popularized the "time-series foundation model" framing.

## Architecture at a glance
TimeGPT is described as a transformer encoder-decoder backbone trained on heterogeneous time series. Model details are closed source, but the system integrates tokenization, training, serving, and conformal uncertainty into a single API that accepts arbitrary univariate or multivariate histories.

## Why it matters
TimeGPT demonstrated demand and feasibility for a production-grade zero-shot forecasting service and introduced the vocabulary of "TS foundation models" to the broader practitioner community, setting the stage for the open research wave that followed.

## In the knowledge graph
- **Cluster:** [Masked-encoder / encoder-decoder TS-FMs](../foundation-models/taxonomy.md#cluster-2-masked-encoder--encoder-decoder-ts-fms)
- **Architecture family:** [Encoder-decoder T5](../architectures/encoder-decoder-t5.md)
- **Related concepts:** [zero-shot forecasting](../concepts/zero-shot-forecasting.md), [probabilistic forecasting](../concepts/probabilistic-forecasting.md)
- **See also:** [chronos](./chronos.md), [moirai](./moirai.md), [timesfm](./timesfm.md)
