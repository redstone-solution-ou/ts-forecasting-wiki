# Large Language Models Are Zero-Shot Time Series Forecasters

> **Short name:** `llmtime` · **arXiv:** [2310.07820](https://arxiv.org/abs/2310.07820) · **PDF:** [local](../../papers/llmtime_2310.07820.pdf) · **Date:** 2023-10 · **Venue:** NeurIPS 2023

**Authors:** Nate Gruver, Marc Finzi, Shikai Qiu et al.

## Abstract
LLMTime shows that off-the-shelf large language models, such as GPT-3 and Llama-2, can forecast time series zero-shot when numerical values are encoded as carefully formatted text. A token-level distribution over digits is converted into a continuous predictive density, matching purpose-built TS models on several benchmarks.

## Key contributions
- Numbers-as-text tokenization scheme designed to avoid BPE artifacts.
- Conversion from discrete-token output distributions to continuous predictive densities.
- Demonstration that vanilla GPT-3 and Llama-2 match or exceed purpose-built TS models zero-shot.
- Observation that GPT-4 sometimes underperforms GPT-3, attributed to tokenization and RLHF artifacts.

## Architecture at a glance
LLMTime uses unmodified pretrained LLMs as forecasters. Numeric series are serialized into digit-level text with carefully chosen separators so that each number maps to a consistent set of tokens. Sampling next-token distributions and reversing the serialization yields continuous probabilistic forecasts.

## Why it matters
LLMTime provided a provocative zero-shot baseline showing that language models trained only on text already contain significant time-series knowledge, sparking investigation into why LLMs forecast and how far pure text-based approaches can go.

## In the knowledge graph
- **Cluster:** [LLM-adapted / reprogramming approaches](../foundation-models/taxonomy.md#cluster-4-llm-adapted--reprogramming-approaches)
- **Architecture family:** [LLM reprogramming](../architectures/llm-reprogramming.md)
- **Related concepts:** [value quantization](../concepts/value-quantization.md), [zero-shot forecasting](../concepts/zero-shot-forecasting.md), [probabilistic forecasting](../concepts/probabilistic-forecasting.md), [in-context learning](../concepts/in-context-learning.md)
- **See also:** [time-llm](./time-llm.md), [gpt4ts](./gpt4ts.md), [chronos](./chronos.md)
