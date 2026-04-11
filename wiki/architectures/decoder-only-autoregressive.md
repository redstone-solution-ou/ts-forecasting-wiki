# Decoder-Only Autoregressive

Decoder-only autoregressive time-series models are GPT-style causal transformers that predict the next time step (or the next patch of steps) conditioned on all preceding ones. The objective is identical in spirit to language-model next-token prediction, but the "tokens" are numeric patches or quantized bins rather than words.

## Overview

An autoregressive TS foundation model factorizes the joint distribution over a series as a product of conditional distributions, each modeled by a causal transformer block stack. During training, every position contributes a loss against its immediate successor, which makes the objective data-efficient and enables variable-length inference: the same model can take any history length and roll out any horizon by repeated sampling or direct multi-step output.

Modern decoder-only TS models almost always operate on patches rather than individual timestamps. A patch of contiguous values is projected into a hidden vector that plays the role of an input embedding; a linear or MLP head decodes each output position back to a patch of real-valued predictions (or the parameters of a predictive distribution, or a categorical distribution over a learned vocabulary). Some designs, notably TimesFM, use an output patch that is longer than the input patch, so a single forward pass emits many future steps at once — this shortens the rollout and reduces exposure bias.

The family scales naturally: stacking more layers and training on more data produces consistent loss improvements, and empirical scaling laws have been measured directly on time series (Lag-Llama, Time-MoE, Timer). It also supports sparse scaling via mixture-of-experts (Time-MoE) and long contexts up to 4096+ tokens.

## Key ideas / variants

- Patched next-token prediction with optional longer output patches (TimesFM).
- Single-Series Sequence (S3) unified format for heterogeneous corpora (Timer).
- Multivariate flattening with universal causal attention (Timer-XL).
- Lag features as extra covariates into the causal stream (Lag-Llama).
- Probabilistic heads (Student-t, categorical) for sampled forecasts.

## Papers that exemplify this (or use this)

- [TimesFM](../papers/timesfm.md) — 200M-param decoder with output-patch longer than input-patch for efficient rollout.
- [Timer](../papers/timer.md) — GPT-style decoder trained on ~1B points under the unified S3 format.
- [Timer-XL](../papers/timer-xl.md) — extends Timer to multivariate next-token with universal TimeAttention.
- [Lag-Llama](../papers/lag-llama.md) — first open decoder-only probabilistic TS FM with a Student-t head.
- [Time-MoE](../papers/time-moe.md) — 2.4B sparse decoder-only MoE trained on Time-300B.

## Related wiki pages

- [Patch tokenization](../concepts/patch-tokenization.md)
- [Mixture-of-experts](mixture-of-experts.md)
- [Scaling laws](../concepts/scaling-laws.md)
- [Probabilistic forecasting](../concepts/probabilistic-forecasting.md)
