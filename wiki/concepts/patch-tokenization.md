# Patch Tokenization

Patch tokenization is the technique of grouping contiguous time-series values into fixed-length windows ("patches") and treating each window as a single transformer token via a linear or MLP projection. It is the dominant input representation in modern TS foundation models because it shortens effective sequence length, gives the self-attention layer a higher-SNR unit to reason over, and cleanly separates the numeric encoder from the sequence model.

## Overview

Naively feeding individual timestamps to a transformer produces long sequences in which each token carries very little information; quadratic attention is then dominated by short-range interactions that convolutional or recurrent baselines already capture. Patching — introduced for TS by PatchTST and now ubiquitous — projects each patch of L contiguous values into a d-dimensional embedding, effectively performing a local average-plus-linear encoding. A series of length T becomes a sequence of length T/L tokens, cutting attention cost quadratically and letting the same parameter budget cover a longer context.

Patch tokenization composes well with other design choices. Channel independence means each variate is patched separately, so the same model handles any number of channels. RevIN is applied before patching to remove instance-level mean and variance. Patches can feed into any backbone — decoder-only, masked encoder, encoder-decoder, MLP-Mixer, or SSM. Output is typically produced in patch units too: for example, TimesFM emits an output patch longer than the input patch so a single forward pass covers more of the horizon. Some models pair patching with per-frequency specialization: MOIRAI uses multiple input projections keyed by patch size (8/16/32/64/128) so that different sampling frequencies are mapped into the hidden space by dedicated parameters. TTM uses adaptive patching to adjust the patch length to the data's effective frequency, and Chronos-2 uses patch tokens as the unit over which its group-attention in-context mechanism operates.

## Key ideas / variants

- Non-overlapping patches projected by linear/MLP to transformer-token dimension.
- Channel independence: each variate patched separately.
- Output patching for multi-step emission per forward pass (TimesFM).
- Multiple patch sizes or adaptive patching for multi-frequency support.
- RevIN applied before patching as standard plumbing.

## Papers that exemplify this (or use this)

- [TimesFM](../papers/timesfm.md) — input patch vs. longer output patch for efficient decoding.
- [MOMENT](../papers/moment.md) — PatchTST-style patching plus RevIN in a masked encoder.
- [MOIRAI](../papers/moirai.md) — multi-patch-size input projection for frequency specialization.
- [Timer-XL](../papers/timer-xl.md) — multivariate next-token over flattened patches.
- [Time-LLM](../papers/time-llm.md) — TS patches reprogrammed into LLM word-embedding space.
- [TTM](../papers/ttm.md) — adaptive patching with resolution prefix tuning.
- [Chronos-2](../papers/chronos-2.md) — encoder-only patch transformer with group attention.

## Related wiki pages

- [Decoder-only autoregressive](../architectures/decoder-only-autoregressive.md)
- [Masked encoder](../architectures/masked-encoder.md)
- [RevIN normalization](revin-normalization.md)
