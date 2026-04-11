# Timer: Generative Pre-trained Transformers Are Large Time Series Models

> **Short name:** `timer` · **arXiv:** [2402.02368](https://arxiv.org/abs/2402.02368) · **PDF:** [local](../../papers/timer_2402.02368.pdf) · **Date:** 2024-02 · **Venue:** ICML 2024

**Authors:** Yong Liu, Haoran Zhang, Chenyu Li et al. (Tsinghua)

## Abstract
Timer recasts time series as a generative pretraining problem, introducing a unified Single-Series Sequence format and training a GPT-style decoder-only transformer on roughly one billion points. The resulting model unifies forecasting, imputation, and anomaly detection under a single next-token objective.

## Key contributions
- Single-Series Sequence (S3) format converting heterogeneous TS into a unified token stream.
- Decoder-only autoregressive pretraining corpus of around 1B time points.
- Unified generative treatment of forecasting, imputation, and anomaly detection.
- Emergent few-shot transfer that can reduce fine-tuning data by up to 99% on downstream tasks.

## Architecture at a glance
Timer is a GPT-style decoder-only transformer trained with next-token prediction on patched real-valued tokens. The S3 format normalizes series into a common sequence layout so that datasets with different lengths, frequencies, and modalities can all feed the same model without architectural changes.

## Why it matters
Timer is one of the clearest demonstrations that the LLM recipe, a large decoder-only transformer trained generatively on a unified sequence format, transfers to time series and yields strong few-shot behavior on downstream TS tasks.

## In the knowledge graph
- **Cluster:** [Decoder-only autoregressive TS-FMs](../foundation-models/taxonomy.md#cluster-1-decoder-only-autoregressive-ts-fms)
- **Architecture family:** [Decoder-only autoregressive](../architectures/decoder-only-autoregressive.md)
- **Related concepts:** [patch tokenization](../concepts/patch-tokenization.md), [multi-task universal](../concepts/multi-task-universal.md), [zero-shot forecasting](../concepts/zero-shot-forecasting.md), [in-context learning](../concepts/in-context-learning.md)
- **See also:** [timer-xl](./timer-xl.md), [timesfm](./timesfm.md), [time-moe](./time-moe.md)
