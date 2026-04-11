# One Fits All: Power General Time Series Analysis by Pretrained LM (GPT4TS / FPT / OFA)

> **Short name:** `gpt4ts` · **arXiv:** [2302.11939](https://arxiv.org/abs/2302.11939) · **PDF:** [local](../../papers/gpt4ts_2302.11939.pdf) · **Date:** 2023-02 · **Venue:** NeurIPS 2023 Spotlight

**Authors:** Tian Zhou, Peisong Niu, Liang Wang et al.

## Abstract
GPT4TS, also known as FPT or OFA, demonstrates that a frozen pretrained language model can serve as a strong backbone for a wide range of time-series tasks. Self-attention and feed-forward blocks are frozen, and only layer-norm and positional-embedding parameters are fine-tuned downstream.

## Key contributions
- Frozen pretrained transformer paradigm for time series using GPT-2, BERT, or BEiT backbones.
- Fine-tuning limited to layer-norm and positional-embedding parameters.
- Universal applicability across six TS task families: forecasting, classification, anomaly detection, imputation, few-shot, and zero-shot.
- Theoretical analysis linking frozen self-attention to principal-component-like operations.

## Architecture at a glance
GPT4TS takes a pretrained transformer and passes patched time-series tokens through it largely unchanged. The bulk of the weights are frozen, and only normalization and positional-embedding parameters adapt per task. Lightweight input and output projection layers map between TS patches and the LM's residual stream.

## Why it matters
GPT4TS is the canonical early demonstration that large language models can be repurposed as generic TS backbones with minimal parameter updates. It is widely used as a reference baseline for LLM-for-TS research and motivated much of the subsequent reprogramming literature.

## In the knowledge graph
- **Cluster:** [LLM-adapted / reprogramming approaches](../foundation-models/taxonomy.md#cluster-4-llm-adapted--reprogramming-approaches)
- **Architecture family:** [LLM reprogramming](../architectures/llm-reprogramming.md)
- **Related concepts:** [patch tokenization](../concepts/patch-tokenization.md), [multi-task universal](../concepts/multi-task-universal.md), [zero-shot forecasting](../concepts/zero-shot-forecasting.md)
- **See also:** [time-llm](./time-llm.md), [llmtime](./llmtime.md)
