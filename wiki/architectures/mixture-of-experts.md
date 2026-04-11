# Mixture-of-Experts

Mixture-of-experts (MoE) transformers replace a dense feed-forward layer with a bank of parallel expert MLPs and a lightweight router that selects the top-k experts per token. For time-series foundation models, MoE is the main route to billion-parameter scale without paying a proportional inference cost, and it provides a natural mechanism for specializing parameters across frequencies, domains, or regimes.

## Overview

In a sparse MoE block, the router reads each token embedding and emits a sparse distribution over E experts; only the top-k (typically 1 or 2) are activated, so compute scales with the active experts rather than the total. During training, a load-balancing auxiliary loss encourages uniform utilization to prevent expert collapse. The payoff is a decoupling of capacity from compute: the model has many more parameters than a dense baseline of the same FLOPs, and different experts end up attending to statistically distinct subsets of inputs.

Time-MoE applies this recipe directly to a decoder-only TS transformer, scaling to 2.4B parameters trained on the Time-300B corpus — the first billion-parameter TS foundation model — and reports clean scaling-law behavior as a function of both active and total parameters. Because the decoder operates on patch tokens rather than words, each expert effectively specializes on a type of local temporal pattern.

Moirai-MoE applies the same technique on the masked-encoder side. The original Moirai used distinct input projections per patch size as a hand-engineered form of frequency specialization; Moirai-MoE drops the frequency projections entirely, uses a single unified projection, and lets token-level sparse routing discover the specialization instead. This yields SOTA results on 39 datasets while activating fewer parameters per token than dense Moirai-Large.

## Key ideas / variants

- Sparse top-k routing with load-balancing auxiliary loss.
- Decoupling capacity (total params) from compute (active params).
- Learned frequency specialization replacing hand-engineered projections (Moirai-MoE).
- Scaling laws that hold for TS as they do for language (Time-MoE).

## Papers that exemplify this (or use this)

- [Time-MoE](../papers/time-moe.md) — 2.4B sparse decoder-only MoE, first billion-param TS FM, trained on Time-300B.
- [Moirai-MoE](../papers/moirai-moe.md) — token-level routing replaces Moirai's multi-patch-size frequency projections.

## Related wiki pages

- [Decoder-only autoregressive](decoder-only-autoregressive.md)
- [Masked encoder](masked-encoder.md)
- [Scaling laws](../concepts/scaling-laws.md)
- [Time-300B](../datasets-benchmarks/time-300b.md)
