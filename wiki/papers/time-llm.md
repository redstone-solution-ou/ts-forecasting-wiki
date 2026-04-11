# Time-LLM: Time Series Forecasting by Reprogramming Large Language Models

> **Short name:** `time-llm` · **arXiv:** [2310.01728](https://arxiv.org/abs/2310.01728) · **PDF:** [local](../../papers/timellm_2310.01728.pdf) · **Date:** 2023-10 · **Venue:** ICLR 2024

**Authors:** Ming Jin, Shiyu Wang, Lintao Ma et al.

## Abstract
Time-LLM reprograms a frozen pretrained language model for time-series forecasting. Time-series patches are aligned with LLM word embeddings through a lightweight reprogramming layer, and domain context is injected via a Prompt-as-Prefix scheme, keeping the LLM backbone entirely frozen.

## Key contributions
- Reprogramming layer that projects TS patches into the embedding space of a frozen LLM.
- Prompt-as-Prefix mechanism for injecting natural-language context about the series.
- No language-model fine-tuning, preserving the general-purpose LLM.
- State-of-the-art few-shot and zero-shot results on eight standard benchmarks.

## Architecture at a glance
Time-LLM keeps a pretrained LLM (e.g., Llama or GPT-2) frozen and learns only a small reprogramming module that maps patched time-series tokens into a compact set of "pseudo-word" embeddings. These pseudo-words are prepended with a textual Prompt-as-Prefix describing the task and series, and a lightweight output head reads off the forecast.

## Why it matters
Time-LLM showed that substantial TS forecasting capability can be unlocked from frozen LLMs with only modest adapter modules, pointing at reprogramming as a data-efficient alternative to pretraining TS foundation models from scratch.

## In the knowledge graph
- **Cluster:** [LLM-adapted / reprogramming approaches](../foundation-models/taxonomy.md#cluster-4-llm-adapted--reprogramming-approaches)
- **Architecture family:** [LLM reprogramming](../architectures/llm-reprogramming.md)
- **Related concepts:** [patch tokenization](../concepts/patch-tokenization.md), [zero-shot forecasting](../concepts/zero-shot-forecasting.md), [in-context learning](../concepts/in-context-learning.md)
- **See also:** [gpt4ts](./gpt4ts.md), [llmtime](./llmtime.md)
