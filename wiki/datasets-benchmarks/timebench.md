# TimeBench

TimeBench is the pretraining corpus used by the THUML [Timer](../papers/timer.md) family of time-series foundation models, comprising roughly one trillion time-series points across a diverse mixture of real and synthetic sources. It was first introduced by name in [Sundial](../papers/sundial.md) (arXiv:2502.00816, Feb 2025), and its curation and augmentation pipeline is documented in more detail in [Timer-S1](../papers/timer-s1.md) (arXiv:2603.04791, Mar 2026), which re-uses and re-curates the same corpus under the "Serial Scaling" framing. It is the largest corpus described in any TS foundation-model paper to date.

## Overview

TimeBench's motivation in Sundial was that a continuous generative objective — Sundial's TimeFlow flow-matching loss — benefits disproportionately from both scale and diversity, because the learned velocity field must cover the full range of dynamics it will be asked to generate at inference. A corpus of approximately 1T points, roughly 3x larger than Time-300B, provides that coverage. The mixture includes real series from many domains (finance, IoT, meteorology, healthcare) plus synthetic components generated with canonical signal families (linear, sinusoidal, exponential, power, impulse, step) and KernelSynth-style Gaussian-process sampling, and it incorporates public dataset snapshots from [Chronos](../papers/chronos.md) and [LOTSA](./lotsa.md).

Timer-S1 is where the curation pipeline is spelled out end-to-end: causal mean imputation for missing values, outlier removal with k-sigma and IQR thresholds on a shifting window, predictability filtering via the Augmented Dickey-Fuller test and forecastability metrics, and test-data leakage removal against the [GIFT-Eval](./gift-eval.md) benchmark. Timer-S1 also adds two augmentation primitives on top of the Sundial-era corpus — *resampling* (down-sampling plus Fourier-based interpolation to expose multiple temporal resolutions) and *value-flipping* (multiplying input/output by -1 to invert trends while preserving temporal dependencies) — and argues both contribute independently to downstream GIFT-Eval gains.

Like LOTSA, the [Time Series Pile](./time-series-pile.md), and Time-300B, TimeBench illustrates that each new-generation TS foundation model tends to ship with a new pretraining corpus of its own, reflecting the field's ongoing experimentation with data scale and composition rather than a converged standard.

## Key ideas / variants

- ~1 trillion TS points — largest known TS pretraining corpus.
- Mixture of real-world series (finance, IoT, meteorology, healthcare) and synthetic series from canonical signal families and KernelSynth.
- Curation pipeline: causal mean imputation, k-sigma/IQR outlier removal, ADF-based predictability filtering, GIFT-Eval leakage removal.
- Augmentation (introduced in Timer-S1): resampling and value-flipping.
- Paired originally with Sundial's TimeFlow flow-matching objective; later with Timer-S1's Serial-Token Prediction objective.

## Papers that exemplify this (or use this)

- [Sundial](../papers/sundial.md) — first named TimeBench and used it for flow-matching pretraining at the ~1T-point scale.
- [Timer-S1](../papers/timer-s1.md) — re-curates and augments TimeBench, documents the curation pipeline in detail, and uses it to train an 8.3B-parameter sparse-MoE decoder-only model with Serial-Token Prediction.

## Related wiki pages

- [Flow-matching continuous](../architectures/flow-matching-continuous.md)
- [Synthetic data augmentation](../concepts/synthetic-data-augmentation.md)
- [Time-300B](time-300b.md)
- [../benchmarks/rebuilding-timebench.md](../benchmarks/rebuilding-timebench.md) — practical recipe for building a TimeBench-style corpus from public sources.
