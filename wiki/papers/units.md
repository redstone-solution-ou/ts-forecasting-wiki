# UniTS: A Unified Multi-Task Time Series Model

> **Short name:** `units` · **arXiv:** [2403.00131](https://arxiv.org/abs/2403.00131) · **PDF:** [local](../../papers/units_2403.00131.pdf) · **Date:** 2024-02 · **Venue:** NeurIPS 2024

**Authors:** Shanghua Gao, Teddy Koker, Owen Queen et al. (Harvard / MIT)

## Abstract
UniTS unifies generative and predictive time-series tasks in a single model via a task-token interface. A modified transformer accepts variable-length inputs and uses task tokens to select between forecasting, classification, anomaly detection, and imputation, supporting strong few-shot transfer across a 38-dataset benchmark.

## Key contributions
- Task-token-based interface letting a single backbone serve many TS tasks.
- Modified transformer handling variable input lengths without padding tricks.
- Unified treatment of generative and predictive tasks.
- 38-dataset benchmark demonstrating strong few-shot transfer.

## Architecture at a glance
UniTS wraps a transformer encoder with learnable task tokens that steer the model toward a particular task at inference. Patched TS inputs are concatenated with task tokens and processed jointly. Task-specific heads decode the corresponding output, and the whole system is trained multi-task across heterogeneous datasets.

## Why it matters
UniTS advances the multi-task-universal direction by showing that a single backbone and task-token prompt can match specialist models on a broad benchmark. Its design makes it easy to add new tasks and datasets without changing the architecture.

## In the knowledge graph
- **Cluster:** [Multi-task / universal unified TS models](../foundation-models/taxonomy.md#cluster-6-multi-task--universal-unified-ts-models)
- **Architecture family:** [Masked encoder](../architectures/masked-encoder.md)
- **Related concepts:** [multi-task universal](../concepts/multi-task-universal.md), [patch tokenization](../concepts/patch-tokenization.md), [zero-shot forecasting](../concepts/zero-shot-forecasting.md)
- **See also:** [moment](./moment.md), [totem](./totem.md), [timer](./timer.md)
