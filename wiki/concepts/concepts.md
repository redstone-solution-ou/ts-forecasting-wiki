# Concepts

This section collects the recurring design ideas and training recipes that cut across time-series foundation model architectures. Most concern how raw numeric series are turned into something a transformer can consume (patch tokenization, value quantization, RevIN), how the model expresses uncertainty (probabilistic forecasting), how it is evaluated without task-specific fine-tuning (zero-shot, in-context learning), and how it is scaled (scaling laws, synthetic data, multi-task heads).

## Sub-pages

- [Patch tokenization](patch-tokenization.md) — contiguous windows as transformer tokens.
- [Value quantization](value-quantization.md) — mapping real values to discrete bins or VQ codes.
- [Zero-shot forecasting](zero-shot-forecasting.md) — inference on unseen series without gradient updates.
- [Probabilistic forecasting](probabilistic-forecasting.md) — predictive distributions and quantiles, not point estimates.
- [In-context learning](in-context-learning.md) — conditioning on related series or demonstrations at inference time.
- [Scaling laws](scaling-laws.md) — loss-vs-compute/data/params curves for TS.
- [RevIN normalization](revin-normalization.md) — reversible instance normalization as standard plumbing.
- [Multi-task universal](multi-task-universal.md) — one model for forecast, classify, impute, anomaly.
- [Synthetic data augmentation](synthetic-data-augmentation.md) — KernelSynth, TSMix, PFN priors, and friends.
