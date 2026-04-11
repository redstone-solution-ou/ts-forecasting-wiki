# TimeBench

TimeBench is the pretraining corpus assembled for Sundial, comprising roughly one trillion time-series points across a diverse mixture of real and synthetic sources. It is the largest corpus described in any TS foundation-model paper to date and was designed to support the continuous-valued flow-matching pretraining objective that Sundial introduces.

## Overview

TimeBench's motivation is that a continuous generative objective — Sundial's TimeFlow flow-matching loss — benefits disproportionately from both scale and diversity, because the learned velocity field must cover the full range of dynamics it will be asked to generate at inference. A corpus of approximately 1T points, roughly 3x larger than Time-300B, provides that coverage. The mixture includes real series from many domains plus synthetic components, giving Sundial a broad prior over temporal dynamics.

Sundial uses TimeBench to pretrain a flow-matching model that outputs native continuous values without any vocabulary, and reports outperforming Time-MoE on several benchmarks while using fewer parameters. The paper attributes the gains both to the more expressive continuous objective and to TimeBench's scale and diversity, and positions TimeBench as the corpus of record for future flow-matching and continuous-valued TS foundation models.

Like LOTSA, the Time Series Pile, and Time-300B, TimeBench illustrates that each new-generation TS foundation model tends to ship with a new pretraining corpus of its own, reflecting the field's ongoing experimentation with data scale and composition rather than a converged standard.

## Key ideas / variants

- ~1 trillion TS points — largest known TS pretraining corpus.
- Mixture of real and synthetic sources.
- Paired with Sundial's TimeFlow flow-matching objective.
- Designed for continuous-valued generative pretraining.

## Papers that exemplify this (or use this)

- [Sundial](../papers/sundial.md) — created TimeBench and used it for flow-matching pretraining.

## Related wiki pages

- [Flow-matching continuous](../architectures/flow-matching-continuous.md)
- [Synthetic data augmentation](../concepts/synthetic-data-augmentation.md)
- [Time-300B](time-300b.md)
