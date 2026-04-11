# TimesFM: A Decoder-Only Foundation Model for Time-Series Forecasting

> **Short name:** `timesfm` · **arXiv:** [2310.10688](https://arxiv.org/abs/2310.10688) · **PDF:** [local](../../papers/timesfm_2310.10688.pdf) · **Date:** 2024-04 · **Venue:** ICML 2024

**Authors:** Abhimanyu Das, Weihao Kong, Rajat Sen et al. (Google Research)

## Abstract
TimesFM introduces a decoder-only transformer pretrained as a general-purpose zero-shot forecaster. The model is trained on a large heterogeneous corpus mixing Google Trends, Wikipedia Pageviews, and synthetic series, and reaches accuracy close to fully supervised baselines on standard benchmarks without any task-specific training.

## Key contributions
- Patched-decoder architecture where the output patch length is larger than the input patch length, enabling efficient multi-step autoregression.
- Handles arbitrary history length, forecast horizon, and sampling frequency from a single model.
- ~200M parameter backbone pretrained on roughly 100B time points.
- Demonstrates that zero-shot forecasting with a single TS foundation model can rival dataset-specific supervised models.

## Architecture at a glance
TimesFM is a GPT-style decoder-only transformer operating on patched real-valued tokens. Input patches are projected into the residual stream, and each output step predicts an entire output patch, shortening the autoregressive rollout. Pretraining uses a next-patch prediction loss on Google Trends, Wiki Pageviews, and synthetic seasonal/trend mixtures.

## Why it matters
TimesFM was among the first TS models to clearly demonstrate that a single pretrained transformer can generalize zero-shot across Monash, Darts, and ETT with quality comparable to supervised specialists, establishing the decoder-only patched paradigm for time-series foundation models.

## In the knowledge graph
- **Cluster:** [Decoder-only autoregressive TS-FMs](../foundation-models/taxonomy.md#cluster-1-decoder-only-autoregressive-ts-fms)
- **Architecture family:** [Decoder-only autoregressive](../architectures/decoder-only-autoregressive.md)
- **Related concepts:** [patch tokenization](../concepts/patch-tokenization.md), [zero-shot forecasting](../concepts/zero-shot-forecasting.md), [scaling laws](../concepts/scaling-laws.md), [synthetic data augmentation](../concepts/synthetic-data-augmentation.md)
- **See also:** [timer](./timer.md), [lag-llama](./lag-llama.md), [time-moe](./time-moe.md)
