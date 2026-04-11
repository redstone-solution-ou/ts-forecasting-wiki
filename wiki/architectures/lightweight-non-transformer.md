# Lightweight Non-Transformer

Lightweight non-transformer foundation models drop the quadratic self-attention backbone in favor of MLP-Mixer style or state-space backbones, trading some raw capacity for dramatically lower parameter counts and latency. The goal is not to beat the largest models on every benchmark but to match them closely enough while running on a CPU or in a single-pass horizon generation.

## Overview

The TTM (Tiny Time Mixer) line replaces the transformer with a TSMixer — alternating channel-mixing and time-mixing MLP blocks over patched inputs. On top of the mixer, TTM adds adaptive patching, a resolution prefix-tuning mechanism that conditions the model on the sampling frequency, and a multi-level design that separates channel modeling from exogenous-variable modeling. Models are between 1M and 5M parameters and can be deployed on CPU, yet beat several much larger transformer foundation models in zero-shot and few-shot settings.

Mamba4Cast replaces attention with the Mamba selective state-space backbone. The selective SSM propagates a fixed-size hidden state across the sequence, giving linear time and memory in the context length, which is attractive for long horizons. Mamba4Cast is trained entirely on synthetic data generated under the Prior-data Fitted Network (PFN) paradigm: samples from a rich prior over TS generative processes are drawn at training time, and the model learns the implied Bayesian posterior predictive. At inference, a single forward pass emits the entire horizon, avoiding autoregressive rollout latency.

Both designs argue that when the inductive biases are right — patching, channel independence, per-frequency conditioning — a small model trained on the right data can substitute for a much larger transformer, especially for applications with edge-compute constraints or strict latency budgets.

## Key ideas / variants

- TSMixer MLP-Mixer backbone on patched inputs (TTM).
- Adaptive patching and resolution prefix tuning for multi-frequency.
- Multi-level channel and exogenous heads.
- Mamba / selective SSM backbone with linear-time inference (Mamba4Cast).
- PFN-style synthetic-only pretraining for one-shot Bayesian forecasting.
- Single forward pass for the full horizon (no autoregressive rollout).

## Papers that exemplify this (or use this)

- [TTM (Tiny Time Mixer)](../papers/ttm.md) — 1–5M-param TSMixer with adaptive patching, CPU-deployable, strong zero-shot.
- [Mamba4Cast](../papers/mamba4cast.md) — Mamba SSM backbone, synthetic-only PFN pretraining, single-pass horizon generation.

## Related wiki pages

- [Patch tokenization](../concepts/patch-tokenization.md)
- [Synthetic data augmentation](../concepts/synthetic-data-augmentation.md)
- [Zero-shot forecasting](../concepts/zero-shot-forecasting.md)
