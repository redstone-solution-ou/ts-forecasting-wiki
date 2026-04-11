# Time-300B

Time-300B is the pretraining corpus assembled for Time-MoE, containing approximately 300 billion time-series observations drawn from 9 domains. It is the largest TS corpus described in any TS foundation-model paper and the enabling resource for the first billion-parameter TS FM.

## Overview

Time-300B was built specifically to answer whether the scaling-law regime observed in language modeling transfers to time-series at billion-parameter scale. Earlier TS foundation models trained on anywhere from hundreds of millions to tens of billions of observations, which is comfortable for 100M-parameter models but not for the capacity of a 2.4B-parameter sparse MoE decoder. Time-300B's approximately 300B points, spanning 9 diverse domains, give Time-MoE a pretraining budget large enough to see loss continue to fall as parameter count grows.

Time-MoE uses Time-300B with a decoder-only sparse mixture-of-experts architecture, context length up to 4096 tokens, and next-patch prediction. The paper reports clean TS scaling laws as a function of both active and total parameters and establishes Time-MoE as the first billion-parameter TS foundation model (ICLR 2025 Spotlight). Because the corpus is paired so closely with Time-MoE, its primary role in the literature is as the compute-hungry corner of the data-scale axis: it is the benchmark data volume that other pretraining corpora are implicitly compared to.

Time-300B is one of four named large TS pretraining corpora alongside [LOTSA](./lotsa.md) (~27B obs), the [Time Series Pile](./time-series-pile.md) (multi-task), and TimeBench (~1T points, [Sundial](../papers/sundial.md)).

## Key ideas / variants

- ~300B observations across 9 domains.
- Paired with Time-MoE's 2.4B-parameter sparse MoE decoder.
- Enables billion-parameter TS scaling-law measurements.
- The largest-to-date observation count among named TS corpora aside from TimeBench.

## Papers that exemplify this (or use this)

- [Time-MoE](../papers/time-moe.md) — created Time-300B and used it to train the first billion-parameter TS FM.

## Related wiki pages

- [Mixture-of-experts](../architectures/mixture-of-experts.md)
- [Scaling laws](../concepts/scaling-laws.md)
- [TimeBench](timebench.md)
