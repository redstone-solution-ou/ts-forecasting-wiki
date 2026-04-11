# LLM Reprogramming

LLM reprogramming is the family of approaches that reuses a pretrained large language model as a time-series forecaster, either by freezing the backbone and wrapping it in lightweight TS-specific adapters, or by encoding numeric values directly as text. The motivation is that LLMs already carry strong sequence priors, so transferring them to TS sidesteps the need for TS-scale pretraining corpora.

## Overview

There are three recognizable styles. The first, represented by Time-LLM, keeps a frozen Llama backbone and inserts a "reprogramming" layer that maps patched TS embeddings into the LLM's word-embedding subspace via a learned cross-attention against a small set of text prototypes; context is injected as a "Prompt-as-Prefix" natural-language description of the series. The LLM is never updated; only the reprogramming module and the output projection train.

The second style, GPT4TS (also known as OFA or FPT), takes an even more minimal approach: it keeps a frozen GPT-2/BERT/BEiT backbone and fine-tunes only layer-norm parameters and positional embeddings, yet reaches competitive or SOTA results on six TS tasks. The paper also draws a theoretical connection between the self-attention blocks of the frozen LLM and principal component analysis, offering a partial explanation for why transfer works.

The third style, LLMTime, doesn't adapt the LLM at all. It tokenizes numbers as ASCII digits (with careful handling of sign, scale, and commas), feeds them to a vanilla GPT-3 or Llama-2, and decodes the generated digit strings back to numeric samples. A discrete-to-continuous density conversion turns the token-level distribution into a proper probabilistic forecast. Despite using no TS-specific training, it matches purpose-built TS models in the zero-shot regime.

## Key ideas / variants

- Reprogramming layer aligning TS patches with LLM word embeddings (Time-LLM).
- Prompt-as-Prefix natural-language context conditioning.
- Frozen backbone with only LN + positional embeddings trained (GPT4TS/OFA).
- Self-attention as implicit PCA interpretation.
- Numbers-as-text tokenization with discrete-to-continuous density (LLMTime).

## Papers that exemplify this (or use this)

- [Time-LLM](../papers/time-llm.md) — reprogramming layer plus Prompt-as-Prefix around a frozen Llama.
- [GPT4TS / OFA / FPT](../papers/gpt4ts.md) — frozen GPT-2/BERT/BEiT fine-tuning only LN + positional embeddings, universal across six TS tasks.
- [LLMTime](../papers/llmtime.md) — zero-shot forecasting via ASCII tokenization of numbers on vanilla GPT-3 / Llama-2.

## Related wiki pages

- [Patch tokenization](../concepts/patch-tokenization.md)
- [Value quantization](../concepts/value-quantization.md)
- [Zero-shot forecasting](../concepts/zero-shot-forecasting.md)
- [Multi-task universal](../concepts/multi-task-universal.md)
