# Value Quantization

Value quantization is the conversion of continuous time-series values into a finite set of discrete tokens, so that forecasting can be posed as a classification problem trained with categorical cross-entropy and decoded by sampling. It is what lets language-model architectures be reused for TS without modification, at the cost of some precision and the need to design a good binning or codebook scheme.

## Overview

The simplest recipe, used by Chronos, is mean-scaling followed by uniform binning: each series is divided by the mean absolute value of a context window, the scaled values are clipped to a range, and that range is split into a fixed number of equal-width bins (typically a few thousand). Each bin index is a "token." Because the vocabulary is small and fixed, an unmodified T5 encoder-decoder can be trained directly with standard language-modeling machinery. At inference time, the decoder samples token trajectories, bin indices are dequantized to the bin center (or a learned continuous representative), and trajectories are multiplied back by the mean-scale and aggregated into quantiles. This yields calibrated probabilistic forecasts almost for free.

A richer alternative is learned discrete representation via a VQ-VAE. TOTEM trains a cross-domain VQ-VAE codebook over patches of time series from multiple domains and then uses the resulting discrete tokens as the input and output of a TS transformer, comparing specialist (per-domain) and generalist (shared-codebook) settings. Because the codebook is learned, each token corresponds to a prototype waveform rather than a scalar bin, and the reconstruction MSE of the VQ-VAE controls precision loss.

At the far end of the spectrum, LLMTime avoids learning any TS-specific vocabulary at all: numbers are tokenized as ASCII digits (with explicit handling of sign, decimal point, and digit grouping), fed to a general-purpose LLM, and decoded back to the real line through a discrete-to-continuous density conversion. This amounts to using the LLM's native BPE vocabulary as the quantizer.

## Key ideas / variants

- Mean scaling + uniform binning + categorical cross-entropy (Chronos).
- Dequantization and sampling for probabilistic forecasts.
- Learned VQ-VAE codebooks over patches, specialist vs. generalist (TOTEM).
- Numbers-as-text digit tokenization using the LLM's native vocabulary (LLMTime).
- Precision-vs-vocabulary-size tradeoff is explicit and tunable.

## Papers that exemplify this (or use this)

- [Chronos](../papers/chronos.md) — mean-scaling + uniform quantization, unmodified T5, categorical training.
- [TOTEM](../papers/totem.md) — VQ-VAE cross-domain codebook, specialist vs. generalist comparison.
- [LLMTime](../papers/llmtime.md) — ASCII-digit tokenization with discrete-to-continuous density.

## Related wiki pages

- [Encoder-decoder T5](../architectures/encoder-decoder-t5.md)
- [Probabilistic forecasting](probabilistic-forecasting.md)
- [LLM reprogramming](../architectures/llm-reprogramming.md)
