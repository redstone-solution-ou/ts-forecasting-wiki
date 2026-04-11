# Sundial: A Family of Highly Capable Time Series Foundation Models

> **Short name:** `sundial` · **arXiv:** [2502.00816](https://arxiv.org/abs/2502.00816) · **PDF:** [local](../../papers/sundial_2502.00816.pdf) · **Date:** 2025-02 · **Venue:** preprint

**Authors:** Yong Liu, Guo Qin, Xiangdong Shi et al. (Tsinghua)

## Abstract
Sundial introduces a family of time-series foundation models built around TimeFlow, a flow-matching objective that natively predicts continuous values without any discrete vocabulary. It is pretrained on TimeBench, a corpus of approximately one trillion points, and delivers probabilistic multi-horizon forecasts with fewer parameters than Time-MoE.

## Key contributions
- TimeFlow flow-matching pretraining objective tailored to continuous TS values.
- Vocabulary-free, fully continuous predictions avoiding quantization artifacts.
- TimeBench pretraining corpus of about one trillion points.
- Probabilistic multi-horizon sampling via the learned flow.
- Reported roughly 4.7% average MSE reduction relative to Time-MoE with fewer parameters.

## Architecture at a glance
Sundial pairs a decoder-style transformer with a flow-matching head that learns a velocity field over continuous value trajectories. Sampling produces diverse forecast paths, and the model is trained end-to-end on the TimeBench corpus without any value quantization.

## Why it matters
Sundial is a leading exponent of the continuous, flow-matching branch of TS foundation models. It shows that avoiding value quantization can both improve accuracy and reduce parameter requirements, opening a distinct architectural path alongside discrete-token TS models.

## In the knowledge graph
- **Cluster:** [Continuous / flow-matching FMs](../foundation-models/taxonomy.md#cluster-7-continuous--flow-matching-fms)
- **Architecture family:** [Flow matching continuous](../architectures/flow-matching-continuous.md)
- **Related concepts:** [probabilistic forecasting](../concepts/probabilistic-forecasting.md), [patch tokenization](../concepts/patch-tokenization.md), [scaling laws](../concepts/scaling-laws.md)
- **Dataset / corpus:** [TimeBench](../datasets-benchmarks/timebench.md)
- **See also:** [time-moe](./time-moe.md), [timer-xl](./timer-xl.md)
