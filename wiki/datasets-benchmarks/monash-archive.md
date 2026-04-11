# Monash Time Series Forecasting Archive

The Monash Time Series Forecasting Archive is a curated collection of public forecasting datasets assembled at Monash University to serve as a shared benchmark for univariate and multivariate forecasting methods. It is the de facto reference corpus against which nearly every recent TS foundation model reports zero-shot numbers.

## Overview

Monash collects dozens of datasets spanning electricity, traffic, weather, retail, finance, tourism, and other domains, at frequencies ranging from minutely to yearly. Each dataset is packaged in a standardized format with fixed train/test splits and reference horizons, and the archive ships canonical evaluation scripts so that numbers are directly comparable across methods. The archive also publishes baseline scores from classical methods (ETS, ARIMA, Theta, seasonal naive) and from supervised deep models, giving zero-shot foundation models a meaningful yardstick: a model that merely matches the classical baselines on unseen data without any per-dataset training is already a non-trivial achievement.

The archive has been central to the zero-shot evaluation of modern TS foundation models. TimesFM used Monash as one of its primary zero-shot benchmarks to argue that a decoder-only ~200M-param model can approach supervised SOTA without per-dataset fine-tuning. Chronos reports Monash numbers across its T5 sizes. MOIRAI positions Monash as its main zero-shot evaluation alongside GIFT-Eval. MOMENT and Lag-Llama likewise report on Monash. The result is that Monash has become the closest thing the field has to a canonical leaderboard for zero-shot TS forecasting.

## Key ideas / variants

- Standardized formats, fixed splits, reference horizons.
- Canonical evaluation scripts for cross-paper comparability.
- Classical and supervised deep-learning baselines published alongside.
- Used primarily as a zero-shot benchmark by foundation-model papers.

## Papers that exemplify this (or use this)

- [TimesFM](../papers/timesfm.md) — zero-shot near supervised SOTA on Monash.
- [MOIRAI](../papers/moirai.md) — main zero-shot evaluation on Monash and GIFT-Eval.
- [Chronos](../papers/chronos.md) — Monash included in the 42-dataset benchmark.
- [MOMENT](../papers/moment.md) — forecast task head evaluated on Monash.
- [Lag-Llama](../papers/lag-llama.md) — open decoder-only probabilistic baseline on Monash.

## Related wiki pages

- [GIFT-Eval](gift-eval.md)
- [Zero-shot forecasting](../concepts/zero-shot-forecasting.md)
