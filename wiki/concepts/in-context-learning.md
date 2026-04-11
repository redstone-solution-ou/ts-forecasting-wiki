# In-Context Learning

In-context learning (ICL) is the ability to adapt a pretrained model's predictions by conditioning on additional examples or related series at inference time, without any parameter updates. For time-series foundation models, ICL lets a single checkpoint exploit related panels, covariates, or demonstrations whenever they exist, bridging the gap between pure zero-shot and full fine-tuning.

## Overview

In language modeling, ICL arises naturally from left-to-right autoregression: the transformer attends to examples placed earlier in the prompt and generalizes from them. For time-series, related-series ICL requires the architecture to accept multiple series in the same forward pass and let them attend to one another. Two recent designs make this explicit.

Chronos-2 replaces the T5 decoder with a 120M encoder-only patch-based transformer that uses "group attention," a structured attention mask in which tokens from related series can attend across the group while preserving per-series causal order where needed. This lets the model ingest multivariate inputs, exogenous covariates, and small panels of related time series as in-context context for the target series — all in one forward pass. The training objective encourages the model to leverage such context when it reduces the quantile loss on the target, so at inference the user can supply related series and observe improved forecasts without fine-tuning.

Timer-XL takes a related route on the decoder-only side: it flattens a multivariate panel into a single sequence of patch tokens and uses a "universal TimeAttention" with causal position embeddings that reflect both the time and variate dimensions. The same architecture then handles univariate forecasting, multivariate forecasting, and covariate-conditioned forecasting uniformly, and can generalize to variate layouts unseen at training time — an ICL-style behavior at the architectural level.

## Key ideas / variants

- Group attention across related series in one forward pass (Chronos-2).
- Universal TimeAttention over flattened multivariate patches (Timer-XL).
- Conditioning on covariates, panels, or demonstration series at inference.
- No parameter updates on target data — distinct from fine-tuning.

## Papers that exemplify this (or use this)

- [Chronos-2](../papers/chronos-2.md) — group attention lets multivariate, covariate, and panel context propagate to the target series.
- [Timer-XL](../papers/timer-xl.md) — universal TimeAttention over flattened multivariate patches, handling varying variate layouts.

## Related wiki pages

- [Zero-shot forecasting](zero-shot-forecasting.md)
- [Encoder-decoder T5](../architectures/encoder-decoder-t5.md)
- [Decoder-only autoregressive](../architectures/decoder-only-autoregressive.md)
