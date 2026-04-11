# Synthetic Data Augmentation

Synthetic data augmentation is the use of algorithmically generated time series — from Gaussian processes, random mixtures of real series, or explicit generative priors — to extend or replace the pretraining corpus. For time-series foundation models, synthetic data is essential because the total volume of truly diverse public TS data is small compared to web-scale text, and because carefully designed priors can cover regimes that are underrepresented in real data.

## Overview

Four techniques dominate the current practice.

KernelSynth, introduced with Chronos, samples functions from a Gaussian process whose kernel is itself sampled from a random combination of base kernels (linear, periodic, RBF, white noise). This yields a rich library of synthetic series with known smoothness, periodicity, and trend properties, and it can be generated cheaply at scale. TSMix, also from the Chronos paper, performs mixup-like augmentation on real series: convex combinations of existing series produce novel but plausible training examples. Chronos's 42-dataset benchmark was a direct demonstration that adding KernelSynth and TSMix to real pretraining data improves zero-shot accuracy.

TimesFM blends synthetic functions with real corpora (Google Trends, Wikipedia pageviews) so that the decoder-only model sees both natural and controlled distributions during pretraining. Chronos-2 relies heavily on synthetic augmentation to support its in-context group-attention training, where related-series context must be abundant enough for the model to learn to use it.

At the extreme, Mamba4Cast is trained only on synthetic data, using the Prior-data Fitted Network (PFN) paradigm: samples from a rich prior over TS generative processes are drawn at every training step, and the model learns the implied Bayesian posterior predictive in a single forward pass. No real series are used in pretraining, yet the resulting checkpoint forecasts competitively in zero-shot. Sundial's TimeBench corpus (~1T points) likewise includes synthetic components as part of its diverse mixture.

## Key ideas / variants

- KernelSynth: Gaussian-process samples with randomly composed kernels.
- TSMix: mixup-style convex combinations of real series.
- Real + synthetic mixtures for broad coverage (TimesFM).
- Synthetic-only PFN pretraining for Bayesian one-shot forecasting (Mamba4Cast).
- Heavy synthetic components inside large mixed corpora (Chronos-2, TimeBench).

## Papers that exemplify this (or use this)

- [Chronos](../papers/chronos.md) — KernelSynth + TSMix augmentation alongside real data.
- [TimesFM](../papers/timesfm.md) — synthetic functions blended with Google Trends / Wikipedia pageviews.
- [Mamba4Cast](../papers/mamba4cast.md) — synthetic-only PFN pretraining.
- [Chronos-2](../papers/chronos-2.md) — heavy synthetic augmentation to enable in-context training.
- [Sundial](../papers/sundial.md) — synthetic components inside TimeBench.

## Related wiki pages

- [Lightweight non-transformer](../architectures/lightweight-non-transformer.md)
- [TimeBench](../datasets-benchmarks/timebench.md)
- [Zero-shot forecasting](zero-shot-forecasting.md)
