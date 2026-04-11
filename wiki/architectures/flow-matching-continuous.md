# Flow-Matching Continuous

Flow-matching continuous models learn a time-dependent vector field that transports a simple base distribution (typically standard Gaussian) onto the data distribution, and generate samples by integrating an ordinary differential equation through that field. For time-series forecasting, this provides a way to emit probabilistic, native continuous-valued predictions without ever discretizing the value axis into a vocabulary.

## Overview

The standard recipe for value-quantized TS models — Chronos is the canonical example — is to bin real values into a fixed vocabulary and train with categorical cross-entropy. This is clean and reuses language-model infrastructure, but it caps precision at the bin width, requires careful per-series scaling, and converts forecasting into a classification problem that can only sample from the chosen grid. Flow-matching objectives sidestep all of this: the model predicts a velocity at each (noisy-sample, time) pair, training reduces to a regression against the ground-truth velocity under a chosen interpolation path, and at inference one simulates the learned flow from noise to data.

Sundial applies this idea to a TS foundation model. It uses a "TimeFlow" flow-matching objective so that the model natively outputs continuous values, trains on TimeBench (approximately one trillion points), and delivers probabilistic multi-horizon forecasts by drawing multiple noise seeds and integrating the flow. The paper reports beating Time-MoE on several benchmarks despite using fewer parameters, which it attributes both to the more expressive continuous objective and to the scale and diversity of TimeBench.

The family is still young in the TS context, but it is an attractive complement to quantized and parametric approaches because it decouples the output representation from any pre-chosen distributional family.

## Key ideas / variants

- TimeFlow flow-matching objective over continuous values (Sundial).
- Native continuous outputs — no vocabulary, no dequantization step.
- Probabilistic multi-horizon forecasting via multi-seed ODE integration.
- TimeBench-scale pretraining (~1T points).

## Papers that exemplify this (or use this)

- [Sundial](../papers/sundial.md) — TimeFlow flow-matching objective, native continuous probabilistic forecasting, trained on TimeBench.

## Related wiki pages

- [Probabilistic forecasting](../concepts/probabilistic-forecasting.md)
- [Value quantization](../concepts/value-quantization.md)
- [TimeBench](../datasets-benchmarks/timebench.md)
