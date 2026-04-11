# Probabilistic Forecasting

Probabilistic forecasting produces a full predictive distribution over future values rather than a single point estimate, so that downstream decisions can account for uncertainty. For time-series foundation models, probabilistic output is increasingly the default, because it makes the model directly useful for risk-sensitive applications (inventory, pricing, capacity planning) and because the training signal — matching a distribution rather than a point — tends to be better behaved.

## Overview

TS foundation models implement probabilistic forecasting in several distinct ways. The parametric-head approach attaches a distribution layer to the transformer output and trains with the corresponding negative log-likelihood. Lag-Llama uses a Student-t head, which is heavy-tailed and well-suited to real-world series with occasional outliers. MOIRAI uses a mixture of Student-t components at every horizon step, giving it multi-modal flexibility; each forward pass yields the full parameter vector of the mixture.

The categorical-sampling approach, used by Chronos, sidesteps parametric assumptions entirely. Values are quantized into a fixed vocabulary, the model is trained with categorical cross-entropy, and at inference one draws many token trajectories from the decoder, dequantizes them, and aggregates to produce quantiles or predictive intervals. Chronos-2 instead trains a quantile decoder head that directly emits a pre-specified set of quantiles per horizon step, making the output probabilistic without sampling.

Flow-matching approaches like Sundial give a continuous probabilistic forecaster: the learned velocity field is integrated from many noise seeds to obtain samples, and quantiles are computed from the resulting cloud of trajectories. Finally, TimeGPT-1 attaches conformal prediction on top of a point-forecasting backbone to yield calibrated intervals with theoretical coverage guarantees independent of model misspecification.

## Key ideas / variants

- Parametric heads: Student-t (Lag-Llama), mixture of Student-t (MOIRAI).
- Categorical sampling over a quantized vocabulary (Chronos).
- Quantile regression heads emitting fixed quantiles per step (Chronos-2).
- Flow-matching ODE sampling for continuous distributions (Sundial).
- Conformal prediction wrapping a point forecaster (TimeGPT-1).

## Papers that exemplify this (or use this)

- [Lag-Llama](../papers/lag-llama.md) — Student-t distribution head on a decoder-only backbone.
- [MOIRAI](../papers/moirai.md) — mixture-of-Student-t head for multi-modal predictive distributions.
- [Chronos](../papers/chronos.md) — categorical sampling and dequantization for probabilistic output.
- [Chronos-2](../papers/chronos-2.md) — quantile decoder emitting fixed quantiles per horizon step.
- [Sundial](../papers/sundial.md) — flow-matching velocity field integrated from many seeds.
- [TimeGPT-1](../papers/timegpt.md) — conformal prediction intervals on top of a point forecaster.

## Related wiki pages

- [Value quantization](value-quantization.md)
- [Flow-matching continuous](../architectures/flow-matching-continuous.md)
- [Masked encoder](../architectures/masked-encoder.md)
