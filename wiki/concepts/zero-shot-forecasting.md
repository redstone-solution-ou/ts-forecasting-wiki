# Zero-Shot Forecasting

Zero-shot forecasting is the ability of a pretrained model to produce accurate forecasts on a new dataset, domain, or frequency without any gradient-based adaptation on target data. It is the defining capability that separates a "foundation model" from a conventional supervised forecaster, and it is what lets a single checkpoint be shipped as a drop-in API or library for arbitrary downstream series.

## Overview

For zero-shot forecasting to work, the pretraining corpus must be broad enough — in domain, frequency, dynamics, and scale — that the target distribution is effectively in-distribution or close to it. The model must also be designed to handle variable history lengths, variable horizons, and variable sampling frequencies at inference without retraining. Instance normalization (RevIN) and per-series mean-scaling are common plumbing to remove level and scale mismatches so the same parameters generalize.

Different model families realize zero-shot differently. TimesFM is trained on a 100B-point mixture of Google Trends, Wikipedia pageviews, and synthetic data, and shows that decoder-only next-patch prediction at ~200M params yields near-supervised accuracy zero-shot on Monash and other benchmarks. Chronos converts target series into its quantized vocabulary and samples — no fine-tuning needed — across 42 benchmark datasets. MOIRAI pretraining on LOTSA and evaluation on Monash and GIFT-Eval establishes zero-shot as the default. TimeGPT-1 was built around zero-shot as the product interface, and Lag-Llama was the first open decoder-only probabilistic FM to demonstrate it. TTM shows that a sub-5M-parameter MLP-Mixer can beat much larger FMs in zero-shot on many datasets, and Mamba4Cast shows the same for an SSM trained only on synthetic data. LLMTime demonstrated the surprising result that a vanilla GPT-3 / Llama-2 with numbers-as-text tokenization could match purpose-built TS models zero-shot.

## Key ideas / variants

- Variable history, horizon, and frequency handled without retraining.
- Per-series instance normalization (RevIN / mean scaling) as the bridge.
- Broad pretraining corpus across domains and frequencies.
- Zero-shot evaluation protocols anchored on Monash, GIFT-Eval, fev-bench.

## Papers that exemplify this (or use this)

- [TimesFM](../papers/timesfm.md) — zero-shot near supervised SOTA on Monash.
- [Chronos](../papers/chronos.md) — 42-dataset zero-shot benchmark.
- [TimeGPT-1](../papers/timegpt.md) — commercial zero-shot API vs. classical/ML/DL baselines.
- [Lag-Llama](../papers/lag-llama.md) — first open decoder-only probabilistic zero-shot FM.
- [LLMTime](../papers/llmtime.md) — vanilla LLM zero-shot via numbers-as-text.
- [TTM](../papers/ttm.md) — tiny MLP-Mixer beating larger FMs zero-shot.
- [MOIRAI](../papers/moirai.md) — strong zero-shot on Monash and GIFT-Eval from LOTSA pretraining.
- [Mamba4Cast](../papers/mamba4cast.md) — SSM trained on synthetic data only, zero-shot by construction.

## Related wiki pages

- [RevIN normalization](revin-normalization.md)
- [Monash Archive](../datasets-benchmarks/monash-archive.md)
- [GIFT-Eval](../datasets-benchmarks/gift-eval.md)
- [In-context learning](in-context-learning.md)
