# Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting

> **Short name:** `lag-llama` · **arXiv:** [2310.08278](https://arxiv.org/abs/2310.08278) · **PDF:** [local](../../papers/lagllama_2310.08278.pdf) · **Date:** 2023-10 · **Venue:** NeurIPS 2023 TSALM workshop

**Authors:** Kashif Rasul, Arjun Ashok, Andrew Robert Williams et al.

## Abstract
Lag-Llama is an early open, decoder-only foundation model for probabilistic time-series forecasting. It uses lagged values as covariates, outputs a Student-t predictive distribution, and includes an empirical study of neural scaling laws for time series.

## Key contributions
- First open decoder-only probabilistic TS foundation model with public weights and reproducible corpus.
- Lag feature engineering feeding the transformer with informative covariates.
- Student-t distribution head for robust probabilistic forecasts.
- Empirical neural scaling laws specific to time-series pretraining.

## Architecture at a glance
Lag-Llama is a Llama-style decoder-only transformer that ingests a target value together with a set of lagged observations selected per frequency. A Student-t output head parameterizes the next-step predictive distribution, and autoregressive rollout produces multi-step probabilistic forecasts.

## Why it matters
Lag-Llama established an open, reproducible baseline for probabilistic TS foundation models and provided one of the first published scaling-law studies for time series, motivating subsequent work on large decoder-only TS models.

## In the knowledge graph
- **Cluster:** [Decoder-only autoregressive TS-FMs](../foundation-models/taxonomy.md#cluster-1-decoder-only-autoregressive-ts-fms)
- **Architecture family:** [Decoder-only autoregressive](../architectures/decoder-only-autoregressive.md)
- **Related concepts:** [probabilistic forecasting](../concepts/probabilistic-forecasting.md), [scaling laws](../concepts/scaling-laws.md), [zero-shot forecasting](../concepts/zero-shot-forecasting.md)
- **See also:** [timesfm](./timesfm.md), [timer](./timer.md), [time-moe](./time-moe.md)
