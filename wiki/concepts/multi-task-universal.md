# Multi-Task Universal Models

Multi-task universal time-series models are single checkpoints that handle multiple downstream tasks — forecasting, classification, anomaly detection, imputation, and sometimes retrieval — without changing the backbone. The design goal is to amortize pretraining over many tasks so that practitioners can adopt one model across their whole pipeline, analogous to how a single LLM handles QA, summarization, and classification.

## Overview

Universal behavior can be obtained in several ways. The task-head approach, used by MOMENT, trains a shared encoder on a masked reconstruction objective and then attaches small task-specific heads (forecasting, classification, anomaly, imputation) on top of its representations. The encoder itself is never specialized, so one weight file serves all tasks.

The generative-unification approach, used by Timer and Timer-XL, recasts every task as generation: forecasting is predicting future tokens, imputation is filling masked tokens, and anomaly detection is scoring likelihood or reconstruction error. Under this view the same causal decoder handles everything, and heterogeneous corpora can be streamed through a single training loop via the Single-Series Sequence (S3) format. UniTS takes a similar stance with a "task-token interface": a learned token prepended to the input tells the transformer which task to perform, and a modified architecture handles variable-length inputs across 38 benchmark datasets.

TOTEM unifies via a shared VQ-VAE codebook: different domains all project into the same discrete-token space, and a single transformer operates on those tokens regardless of task. GPT4TS/OFA demonstrates that a frozen LLM, with only layer-norm and positional embeddings trained, already generalizes across six TS tasks. Chronos-2 unifies univariate, multivariate, covariate, and panel forecasting inside a single group-attention architecture.

The practical payoff is fewer models to ship, reuse of pretraining effort across task types, and better generalization to rare task variants at inference time.

## Key ideas / variants

- Shared encoder + task-specific heads (MOMENT).
- Generative unification of forecast/impute/anomaly (Timer, Timer-XL).
- Task-token interface prepended to the input (UniTS).
- Shared VQ-VAE discrete-token space across tasks (TOTEM).
- Frozen LLM backbone as a universal TS solver (GPT4TS/OFA).
- Single-model handling of univariate, multivariate, and covariate (Chronos-2).

## Papers that exemplify this (or use this)

- [MOMENT](../papers/moment.md) — shared encoder with task heads for forecast / classify / anomaly / impute.
- [Timer](../papers/timer.md) — generative unification under the S3 format.
- [Timer-XL](../papers/timer-xl.md) — universal TimeAttention handling univariate/multivariate/covariate uniformly.
- [UniTS](../papers/units.md) — task-token interface over a 38-dataset benchmark.
- [TOTEM](../papers/totem.md) — shared VQ-VAE codebook across domains and tasks.
- [GPT4TS / OFA / FPT](../papers/gpt4ts.md) — frozen LLM universal across six TS tasks.
- [Chronos-2](../papers/chronos-2.md) — group attention unifies univariate, multivariate, and panel forecasting.

## Related wiki pages

- [Masked encoder](../architectures/masked-encoder.md)
- [Decoder-only autoregressive](../architectures/decoder-only-autoregressive.md)
- [LLM reprogramming](../architectures/llm-reprogramming.md)
