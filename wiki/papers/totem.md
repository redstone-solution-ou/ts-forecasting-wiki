# TOTEM: TOkenized Time Series EMbeddings

> **Short name:** `totem` · **arXiv:** [2402.16412](https://arxiv.org/abs/2402.16412) · **PDF:** [local](../../papers/totem_2402.16412.pdf) · **Date:** 2024-02 · **Venue:** preprint

**Authors:** Sabera Talukder, Yisong Yue, Georgia Gkioxari (Caltech)

## Abstract
TOTEM learns a cross-domain discrete time-series codebook via a VQ-VAE and evaluates both specialist and generalist models on top of it at scale. It makes the case for discrete-token TS pretraining as a principled alternative to raw continuous patches.

## Key contributions
- VQ-VAE-based cross-domain codebook for time-series tokens.
- Systematic comparison of specialist versus generalist models trained on the shared codebook.
- Evidence that discrete tokenization scales beneficially across TS domains.

## Architecture at a glance
TOTEM trains a convolutional VQ-VAE whose codebook provides a discrete vocabulary for time-series patches. Downstream transformers then consume these tokens exactly as an NLP model would consume words, allowing cross-domain pretraining and transfer with standard sequence-model machinery.

## Why it matters
TOTEM anchors the discrete-tokenization branch of TS foundation models. By quantifying specialist-versus-generalist trade-offs with a common codebook, it motivates subsequent approaches that combine VQ-style tokenization with universal pretraining objectives.

## In the knowledge graph
- **Cluster:** [Multi-task / universal unified TS models](../foundation-models/taxonomy.md#cluster-6-multi-task--universal-unified-ts-models)
- **Architecture family:** [Masked encoder](../architectures/masked-encoder.md)
- **Related concepts:** [value quantization](../concepts/value-quantization.md), [patch tokenization](../concepts/patch-tokenization.md), [multi-task universal](../concepts/multi-task-universal.md)
- **See also:** [chronos](./chronos.md), [units](./units.md)
