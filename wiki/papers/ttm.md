# Tiny Time Mixers (TTMs): Fast Pre-trained Models for Zero/Few-Shot Multivariate Forecasting

> **Short name:** `ttm` · **arXiv:** [2401.03955](https://arxiv.org/abs/2401.03955) · **PDF:** [local](../../papers/ttm_2401.03955.pdf) · **Date:** 2024-01 · **Venue:** NeurIPS 2024

**Authors:** Vijay Ekambaram, Arindam Jati, Nam H. Nguyen et al. (IBM)

## Abstract
Tiny Time Mixers are compact non-transformer foundation models based on the TSMixer (MLP-Mixer) backbone. Despite having only 1M to 5M parameters, they match or beat much larger transformer TS foundation models on zero-shot and few-shot multivariate forecasting.

## Key contributions
- Non-transformer TSMixer backbone chosen for CPU-friendly deployment.
- Adaptive patching, diverse resolution sampling, and resolution prefix tuning for multi-frequency generalization.
- Multi-level channel head and explicit exogenous-variable support.
- 1M to 5M parameter models that beat far larger TS foundation models zero/few-shot.

## Architecture at a glance
TTM stacks TSMixer blocks that alternate feature-mixing and time-mixing MLPs over patched inputs. A resolution prefix token informs the network of the sampling frequency, and channel mixing plus exogenous heads integrate multivariate and covariate information. The model is trained on heterogeneous public TS data.

## Why it matters
TTM makes the case that parameter count is not destiny for TS foundation models. By exploiting TS-specific inductive bias and careful multi-resolution training, a tiny MLP-Mixer can serve as a practical, edge-deployable foundation model, challenging the transformer-by-default trend.

## In the knowledge graph
- **Cluster:** [Lightweight / non-transformer FMs](../foundation-models/taxonomy.md#cluster-5-lightweight--non-transformer-fms)
- **Architecture family:** [Lightweight non-transformer](../architectures/lightweight-non-transformer.md)
- **Related concepts:** [patch tokenization](../concepts/patch-tokenization.md), [zero-shot forecasting](../concepts/zero-shot-forecasting.md), [multi-task universal](../concepts/multi-task-universal.md)
- **See also:** [mamba4cast](./mamba4cast.md), [moirai](./moirai.md)
