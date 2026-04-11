# Chronos: Learning the Language of Time Series

> **Short name:** `chronos` · **arXiv:** [2403.07815](https://arxiv.org/abs/2403.07815) · **PDF:** [local](../../papers/chronos_2403.07815.pdf) · **Date:** 2024-11 · **Venue:** TMLR 2024

**Authors:** Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen et al. (AWS)

## Abstract
Chronos treats forecasting as language modeling on quantized time series. Values are mean-scaled and uniformly binned into a fixed vocabulary, then fed to an off-the-shelf T5 encoder-decoder trained with categorical cross-entropy. At inference, sampled tokens are dequantized to produce probabilistic forecasts.

## Key contributions
- Tokenization scheme combining mean-scaling with uniform value quantization into a fixed categorical vocabulary.
- Reuse of unmodified T5 architectures across a family of sizes (20M to 710M parameters).
- TSMix data augmentation (convex combinations of real series) and KernelSynth (Gaussian-process-based synthetic generation).
- Broad benchmarking across 42 datasets, showing strong zero-shot performance.

## Architecture at a glance
Chronos is an encoder-decoder transformer based directly on the T5 family. Time series are preprocessed into discrete token sequences via mean-scaling and uniform binning. Training optimizes next-token cross-entropy, and forecasts are obtained by sampling multiple trajectories and dequantizing them to recover continuous predictive distributions.

## Why it matters
Chronos showed that minimal tokenization plus an existing language-model architecture, without any TS-specific inductive bias, can compete with purpose-built time-series models. It crystallized the "language modeling for time series" viewpoint and became a standard baseline for subsequent TS foundation models.

## In the knowledge graph
- **Cluster:** [Masked-encoder / encoder-decoder TS-FMs](../foundation-models/taxonomy.md#cluster-2-masked-encoder--encoder-decoder-ts-fms)
- **Architecture family:** [Encoder-decoder T5](../architectures/encoder-decoder-t5.md)
- **Related concepts:** [value quantization](../concepts/value-quantization.md), [probabilistic forecasting](../concepts/probabilistic-forecasting.md), [zero-shot forecasting](../concepts/zero-shot-forecasting.md), [synthetic data augmentation](../concepts/synthetic-data-augmentation.md)
- **See also:** [chronos-2](./chronos-2.md), [moirai](./moirai.md), [moment](./moment.md)
