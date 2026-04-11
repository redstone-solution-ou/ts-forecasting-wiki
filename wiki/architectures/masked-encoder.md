# Masked Encoder

A masked encoder is a bidirectional transformer pretrained to reconstruct randomly masked patches of a time series from their surrounding context, analogous to BERT-style masked language modeling. Because attention is non-causal, every output position sees the full history and future of the (visible) input, making the family a natural fit for imputation, anomaly detection, and representation learning — and, with a lightweight forecasting head, for forecasting as well.

## Overview

The pretraining signal is a masked reconstruction loss: a fraction of the patch tokens are replaced by a learned mask embedding, and the encoder must predict the original values at those positions. This teaches the model to use both left and right context, which is a strength for non-forecasting tasks but means forecasting requires either (a) appending learned "future" mask tokens to the input and letting the encoder fill them in, or (b) attaching a dedicated forecast head on top of the representation.

Masked TS encoders typically use the PatchTST recipe: channel independence, non-overlapping patches, and instance normalization (RevIN) inside the input pipeline. This lets a single model handle arbitrary numbers of channels at inference by batching them independently. MOMENT initializes from T5 encoder weights; MOIRAI builds the encoder from scratch but introduces two TS-specific tricks — multiple input projections keyed by patch size so that different frequencies receive dedicated parameters, and "any-variate attention" that lets the same model ingest univariate, multivariate, and covariate inputs without architectural changes.

The family pairs well with probabilistic heads: MOIRAI outputs the parameters of a mixture of Student-t distributions at every horizon step, so a single forward pass yields a calibrated predictive distribution.

## Key ideas / variants

- Masked-patch reconstruction as the pretraining objective.
- Multi-patch-size input projection for frequency specialization (MOIRAI).
- Any-variate attention for unified univariate/multivariate inputs.
- Task heads (forecast / classify / anomaly / impute) reusing the same encoder (MOMENT).
- Sparse MoE variant that replaces frequency projections with token-level routing (Moirai-MoE).

## Papers that exemplify this (or use this)

- [MOMENT](../papers/moment.md) — T5-initialized encoder with PatchTST patching and RevIN, multi-task via task heads.
- [MOIRAI](../papers/moirai.md) — masked encoder with multi-patch-size projections and mixture-of-Student-t head.
- [Moirai-MoE](../papers/moirai-moe.md) — replaces frequency projections with sparse token-level routing.

## Related wiki pages

- [Patch tokenization](../concepts/patch-tokenization.md)
- [RevIN normalization](../concepts/revin-normalization.md)
- [Multi-task universal](../concepts/multi-task-universal.md)
- [Mixture-of-experts](mixture-of-experts.md)
