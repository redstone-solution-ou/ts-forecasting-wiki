# RevIN Normalization

Reversible Instance Normalization (RevIN) is a simple plumbing layer that standardizes each input series to zero mean and unit variance before it enters the model, and then reverses the transform on the output. It is now standard equipment in patched TS models because it removes per-series level and scale shifts that would otherwise force the transformer to waste capacity on trivial rescaling.

## Overview

Real-world time series live at wildly different scales — one sensor is in Kelvin, another in percent, another in USD — and their means and variances often drift across windows. A model trained on raw values has to learn, for every batch, to undo these shifts before it can reason about shape. RevIN sidesteps this by computing the mean and variance on the input window, subtracting and dividing at the input, and then multiplying and adding them back at the output. Because the transform is reversible and deterministic given the input window, gradients flow cleanly and the model never sees level or scale differences across series.

In practice RevIN is applied per-instance (hence "instance normalization") and per-channel, making it naturally compatible with channel-independent patching: each variate is patched, normalized, and processed as an independent sequence. The operation is cheap — a few moments — and adds minimal compute, but its effect on zero-shot transfer is large, because any pretrained model that sees normalized inputs generalizes across arbitrary unit systems.

MOMENT lists RevIN explicitly as part of its input pipeline, alongside PatchTST-style patching. Most other masked-encoder and decoder-only TS foundation models use an equivalent per-window mean-scaling or instance normalization under different names (Chronos's mean scaling plays the same role for its quantized tokenizer). As a result, RevIN is rarely a headline contribution in a given paper, but its presence or absence is decisive for cross-domain transfer.

## Key ideas / variants

- Per-instance, per-channel standardization applied before the model.
- Reversible: the same statistics denormalize the output.
- Decouples model capacity from level and scale shifts across series.
- Frequently implemented under other names (mean scaling, instance norm).

## Papers that exemplify this (or use this)

- [MOMENT](../papers/moment.md) — explicit RevIN in the PatchTST-style input pipeline.
- [MOIRAI](../papers/moirai.md) — per-instance normalization as standard plumbing in a masked encoder.
- [Chronos](../papers/chronos.md) — mean scaling plays the role of RevIN before quantization.

## Related wiki pages

- [Patch tokenization](patch-tokenization.md)
- [Masked encoder](../architectures/masked-encoder.md)
- [Zero-shot forecasting](zero-shot-forecasting.md)
