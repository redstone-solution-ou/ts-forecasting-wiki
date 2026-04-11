# MOMENT: A Family of Open Time-series Foundation Models

> **Short name:** `moment` · **arXiv:** [2402.03885](https://arxiv.org/abs/2402.03885) · **PDF:** [local](../../papers/moment_2402.03885.pdf) · **Date:** 2024-10 · **Venue:** ICML 2024

**Authors:** Mononito Goswami, Konrad Szafer, Arjun Choudhry et al. (CMU Auton Lab)

## Abstract
MOMENT is an open family of T5-initialized encoder-only foundation models pretrained with masked reconstruction over the Time Series Pile, a large aggregation of public datasets. Small task heads adapt the same backbone to forecasting, classification, anomaly detection, and imputation.

## Key contributions
- Assembly of the Time Series Pile from Informer, Monash, UCR/UEA, and TSB-UAD datasets as a unified pretraining corpus.
- Encoder-only transformer with PatchTST-style patching and RevIN instance normalization.
- Masked-reconstruction pretraining objective at patch level.
- Multi-task deployment via lightweight per-task heads: forecast, classification, anomaly, imputation.
- Open release on Hugging Face in Small (40M), Base (125M), and Large (385M) sizes.

## Architecture at a glance
MOMENT starts from T5 encoder weights, reuses its transformer blocks, and operates on patched, RevIN-normalized real-valued tokens. A fraction of patches is masked and reconstructed during pretraining. At downstream time, the backbone is frozen or lightly tuned while a small task-specific head is trained on top.

## Why it matters
MOMENT provided one of the first fully open TS foundation model families with broad multi-task capability, a reproducible pretraining corpus, and competitive performance across four canonical TS tasks, acting as an accessible baseline for the open-source community.

## In the knowledge graph
- **Cluster:** [Masked-encoder / encoder-decoder TS-FMs](../foundation-models/taxonomy.md#cluster-2-masked-encoder--encoder-decoder-ts-fms)
- **Architecture family:** [Masked encoder](../architectures/masked-encoder.md)
- **Related concepts:** [patch tokenization](../concepts/patch-tokenization.md), [revin normalization](../concepts/revin-normalization.md), [multi-task universal](../concepts/multi-task-universal.md), [zero-shot forecasting](../concepts/zero-shot-forecasting.md)
- **Dataset / corpus:** [Time Series Pile](../datasets-benchmarks/time-series-pile.md)
- **See also:** [moirai](./moirai.md), [chronos](./chronos.md), [units](./units.md)
