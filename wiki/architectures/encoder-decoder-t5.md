# Encoder-Decoder (T5-style)

Encoder-decoder T5-style time-series foundation models treat forecasting as a sequence-to-sequence problem: the encoder consumes a history window, and the decoder autoregressively emits the forecast horizon. The appeal is that unmodified T5 architectures, and in Chronos the unmodified T5 codebase, can be repurposed for time series simply by redefining the tokenizer.

## Overview

In Chronos the key design move is to avoid changing the transformer at all. Instead, real-valued inputs are mean-scaled per series and then uniformly quantized into a fixed vocabulary of integer bins, exactly as if they were subword tokens. Training then reduces to standard categorical cross-entropy on the decoder side, and forecasts are produced by sampling from the decoder, dequantizing back to the real line, and aggregating trajectories into quantiles. This makes the family fully probabilistic by construction, and transfers all the engineering around T5 (sizes from 20M to 710M, deterministic training, HuggingFace compatibility) to the TS setting.

Chronos-2 moves away from the vocabulary but keeps the encoder side: it replaces the T5 decoder with a 120M encoder-only patch-based transformer augmented with "group attention," which lets multiple related series attend to one another so the model can exploit in-context examples, covariates, and multivariate panels in a single forward pass. A quantile decoder head then produces probabilistic multi-horizon outputs directly. TimeGPT-1, the earliest commercial TS foundation model, is also an encoder-decoder transformer exposed via an API and paired with conformal prediction for uncertainty.

The common thread across the family is a clean training signal on a fixed-size output and an inference interface that naturally decouples history and horizon.

## Key ideas / variants

- Unmodified T5 stack plus tokenizer redefinition (Chronos).
- Mean scaling + uniform quantization to reach a discrete vocabulary.
- Encoder-only patch variant with group attention and quantile decoder (Chronos-2).
- Categorical sampling and dequantization as a probabilistic forecaster.
- API-hosted encoder-decoder with conformal uncertainty (TimeGPT-1).

## Papers that exemplify this (or use this)

- [Chronos](../papers/chronos.md) — unmodified T5 encoder-decoder with value quantization and categorical training.
- [Chronos-2](../papers/chronos-2.md) — 120M encoder-only patch model with group attention and quantile decoder.
- [TimeGPT-1](../papers/timegpt.md) — first commercial TS foundation model, encoder-decoder with conformal uncertainty.

## Related wiki pages

- [Value quantization](../concepts/value-quantization.md)
- [Probabilistic forecasting](../concepts/probabilistic-forecasting.md)
- [In-context learning](../concepts/in-context-learning.md)
- [Synthetic data augmentation](../concepts/synthetic-data-augmentation.md)
