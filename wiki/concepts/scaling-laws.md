# Scaling Laws

Scaling laws are empirical curves relating model loss to compute, parameter count, and dataset size. In language modeling they motivated the move from 100M- to 100B-parameter models; for time-series foundation models, measuring and confirming such laws has been an important validation that the "more data and parameters = better" recipe transfers from text to temporal data.

## Overview

A TS scaling-law study typically fits power laws to held-out loss across a grid of model sizes, token budgets, and sometimes active-vs-total parameters (for MoE models). The qualitative claim — that loss falls predictably as any of these inputs grow — has now been replicated on multiple TS foundation models, giving some confidence that the current generation is still in a compute-limited regime and that larger models trained on larger corpora will keep improving.

Lag-Llama was among the first open decoder-only TS FMs to report scaling-law curves, framed as empirical evidence that TS transformers exhibit loss trends analogous to LLMs. Time-MoE pushes the other end of the curve by training a 2.4B-parameter sparse mixture-of-experts decoder on the Time-300B corpus (~300B points, 9 domains), establishing it as the first billion-parameter TS FM and reporting clean scaling behavior as a function of both active and total parameters. TimesFM at ~200M params on a ~100B-point corpus also fits cleanly on a scaling curve, showing that the 100M–1B regime is sufficient to reach near-supervised zero-shot accuracy on Monash-style benchmarks. Timer reports emergent few-shot ability (roughly 99% data reduction to match supervised baselines) that is best interpreted as the TS analogue of the few-shot emergence observed in language scaling studies.

Together these results argue that the compute-data-param curve for TS is not yet saturated and that future gains will come as much from scale as from architectural innovation.

## Key ideas / variants

- Power-law fits of held-out loss vs. compute / params / tokens.
- Dense and sparse (MoE) axes: active vs. total parameters.
- Emergent few-shot capability at sufficient scale.
- Evidence the current TS FM regime is still compute-limited.

## Papers that exemplify this (or use this)

- [Lag-Llama](../papers/lag-llama.md) — early empirical TS scaling laws on an open decoder-only FM.
- [Time-MoE](../papers/time-moe.md) — 2.4B-param sparse MoE, scaling measured on Time-300B.
- [TimesFM](../papers/timesfm.md) — ~200M-param decoder fit on a ~100B-point corpus.
- [Timer](../papers/timer.md) — emergent few-shot with ~99% data reduction vs. supervised baselines.

## Related wiki pages

- [Decoder-only autoregressive](../architectures/decoder-only-autoregressive.md)
- [Mixture-of-experts](../architectures/mixture-of-experts.md)
- [Time-300B](../datasets-benchmarks/time-300b.md)
