# RevIN Normalization

## Intuition

Reversible Instance Normalization (RevIN) is a tiny, almost boring plumbing layer: subtract the mean and divide by the standard deviation of the current input window before the model sees it, then put the mean and std back on the output. It is boring but absolutely load-bearing for modern TS foundation models, because it removes the per-series level and scale shifts that would otherwise force the transformer to waste capacity on trivial rescaling. If you forget RevIN, a foundation model trained on one domain simply cannot generalize to another domain at a different unit — and almost the entire zero-shot story collapses.

## Mechanics

Given a context window `x \in R^{B \times T \times C}` (batch, time, channels), RevIN computes per-instance, per-channel statistics and applies an affine reversal:

```
# Forward (before the model)
mu   = mean(x, dim=time, keepdim=True)                 # (B, 1, C)
var  = var(x,  dim=time, keepdim=True) + eps           # (B, 1, C)
x_n  = gamma * (x - mu) / sqrt(var) + beta             # normalized input

# ... model operates on x_n, producing y_n (forecast in normalized space) ...

# Inverse (after the model)
y    = sqrt(var) * (y_n - beta) / gamma + mu           # back on the original scale
```

`gamma, beta` are learnable affine parameters (optional; vanilla RevIN just uses `gamma=1, beta=0`). Crucially, `mu` and `var` are *instance-specific*: each example in the batch has its own statistics, computed from its own history window, and those same statistics denormalize its own forecast. Nothing is pooled across the batch.

For channel-independent architectures, RevIN is applied independently per variate: each of the `C` channels has its own `(mu_c, var_c)` and is processed as a separate sequence by the same weights. This decouples "how do I model shape" from "how do I handle unit systems," putting the latter in the normalization layer and freeing the weights to focus on the former.

In Chronos the same role is played by mean-absolute-value scaling before quantization:

```
s     = mean(|x|) + eps
x_sc  = clip(x / s, -C, C)
tokens = quantize(x_sc)
# ... decoder samples token trajectories ...
y_real = dequantize(samples) * s
```

Not reversible in the affine-std sense, but operationally equivalent: per-series scaling that cancels out in the final prediction.

## Why it works

The theoretical content is small but exact. A foundation model is trying to learn `p(y | x)` for arbitrary `(x, y)` drawn from arbitrary domains. Under a simple additive-shift model `x_d = s_d \cdot z + l_d` where `s_d, l_d` are domain-specific scale and level and `z` is the "standardized shape," the interesting structure lives entirely in `z`. Any model trained on raw `x_d` has to learn an inverse affine transform for every `(s_d, l_d)` it sees — an *infinite* family in zero-shot — which is obviously impossible. RevIN *removes* this nuisance by construction: the model only ever sees `z`, and the domain-specific affine sits outside the model where it belongs.

This connects directly to general-purpose ML intuitions. BatchNorm trades off too much information across examples to be safe for sequence modeling; LayerNorm normalizes along the feature dimension, which helps optimization but does not cancel level/scale shifts. Instance normalization (over the time axis, per sample) is the version that does exactly the right thing for TS: it handles the "unit system nuisance" and leaves everything else for the model.

The *reversible* part is not just cosmetic — it is what lets the model train on a proper loss. Without inverting the statistics at the output, the loss would be computed in normalized space and would give equal weight to a 10% error on a small series and a 10% error on a large one. With inversion, the loss lives in original units, which is typically what users care about.

## Trade-offs and failure modes

RevIN assumes *window-level stationarity*: the statistics of the future are well approximated by the statistics of the visible history. This breaks whenever there is a regime shift inside the window (a trend, a step change, a slowly drifting mean), because the model trained on standardized histories cannot easily express "the level is rising by 10% per day" in normalized space. Intermittent series with many zeros have `mean ≈ 0` and `var ≈ 0`, making the normalization unstable unless `eps` is generous; heavy-tailed series can see single outliers dominate `var` and squash all other variation.

Empirically, removing RevIN drops zero-shot accuracy by large factors and almost always makes the training loss noisier. Keeping it is basically non-negotiable, but the failure modes are real enough that several papers explore variants (subtracting a rolling mean, using robust statistics, or using learned affine `gamma, beta`).

## Design choices in the literature

- `[MOMENT](../papers/moment.md)` — explicit RevIN in the PatchTST-style input pipeline, applied per-channel before patching.
- `[MOIRAI](../papers/moirai.md)` — per-instance normalization as standard masked-encoder plumbing, used alongside the multi-patch-size projections.
- `[Chronos](../papers/chronos.md)` — mean-absolute-value scaling plays the role of RevIN before the quantized tokenizer, deliberately non-reversible at the std level because the T5 operates on discrete tokens.
- `[TimesFM](../papers/timesfm.md)` — per-window standardization before patch projection, with the statistics propagated to the output head.
- `[TTM](../papers/ttm.md)` — RevIN plus a resolution prefix-tuning token that conditions the mixer on the sampling frequency, addressing the case where the "right" normalization depends on the regime.

## Open questions

- **Is per-window stationarity too strong?** For non-stationary series, window-level `mu, var` are biased estimators; whether a learned normalization can do better is an active area.
- **Robust statistics.** Median/MAD are more robust than mean/std for heavy-tailed series, but almost no TS FM reports ablations.
- **Per-patch vs per-window normalization.** PatchTST normalized per-window; some recent work experiments with per-patch normalization, which trades coverage for locality.
- **Joint normalization across channels.** Channel-independent RevIN ignores cross-variate structure that might be informative (e.g. a dominant variate setting the reference level).
- **Interaction with the quantizer.** Chronos's mean-scaling interacts with its bin grid in non-obvious ways; a finer treatment of that interaction is open.

## Papers that exemplify this

- `[MOMENT](../papers/moment.md)` — explicit RevIN in the input pipeline of a masked encoder.
- `[MOIRAI](../papers/moirai.md)` — per-instance normalization as standard plumbing alongside multi-patch-size projections.
- `[Chronos](../papers/chronos.md)` — mean-absolute-value scaling before the quantized tokenizer; operationally plays the same role.
- `[TimesFM](../papers/timesfm.md)` — per-window standardization with propagated statistics through the decoder head.
- `[TTM](../papers/ttm.md)` — RevIN plus a resolution prefix-tuning token to handle multi-frequency regimes.

## Related wiki pages

- [Data normalization](data-normalization.md) — the broader normalization landscape this page sits within (mean-scaling, partial-window, differencing, and why RevIN won)
- [Patch tokenization](patch-tokenization.md)
- [Zero-shot forecasting](zero-shot-forecasting.md)
- [Value quantization](value-quantization.md)
- [Masked encoder](../architectures/masked-encoder.md)
- [Lightweight non-transformer](../architectures/lightweight-non-transformer.md)
