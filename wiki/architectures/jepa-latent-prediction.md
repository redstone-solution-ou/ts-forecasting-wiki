# JEPA / Latent-Target Prediction

## Intuition

A JEPA-style time-series model is an encoder-plus-predictor pair trained so the predictor can recover the *embedding* of a masked or future portion of the input under a slowly-moving target encoder, instead of the input itself. The architecture family bundles four parts: a tokenizer (CNN or linear projection over patches), an online encoder (typically a transformer), a predictor (also a transformer), and a target encoder maintained as an exponential moving average of the online encoder. The training loss is computed in the latent space, and a stop-gradient on the target encoder is what stops the loop from collapsing to a trivial constant. This family contrasts with [masked-encoder](masked-encoder.md) models, which reconstruct values in input space, and with [decoder-only autoregressive](decoder-only-autoregressive.md) models, which forecast values directly.

## Mechanics

End-to-end forward pass for the canonical JEPA-for-TS recipe (TS-JEPA-style):

```
# Inputs: x in R^T (univariate window), P = patch length
patches            = unfold(x, size=P, stride=P)            # (N, P)
tokens             = CNN_tokenizer(patches) + pos_emb       # (N, D)
mask_ids           = sample_uniform(N, ratio=0.7)           # 70% masked is typical
visible, hidden    = split(tokens, mask_ids)

# Online branch (gradients flow)
z_visible          = E_theta(visible)                       # transformer encoder
z_pred_hidden      = P_beta(z_visible)                      # transformer predictor

# Target branch (no gradients)
with stop_gradient:
    z_target_hidden = E_xi(hidden)                          # EMA copy of E_theta

# Loss: latent-space distance over masked positions only
L = mean_i_in_mask( || z_pred_hidden[i] - z_target_hidden[i] ||_1 )

# Bookkeeping: EMA update *after* gradient step
E_xi.params = m * E_xi.params + (1 - m) * E_theta.params    # m ~ 0.998
```

Three TS-specific extensions in the current literature:

- **[LaT-PFN](../papers/lat-pfn.md)** keeps the JEPA loop but adds a **PFN context branch**: the predictor is a transformer with cross-attention from the held-out series prompt to a learned average-pooled summary of *related synthetic series*. Decoder is separate, trained with stop-gradient on the latent prediction. Embedder is a dilated MobileNet1D (CNN), not a transformer.
- **[MTS-JEPA](../papers/mts-jepa.md)** keeps the loop and adds two pieces. First, **multi-resolution targets**: the EMA branch encodes both a fine patched view and a coarse downsampled-window view, and the online branch is supervised against both. Second, a **soft codebook bottleneck** `Q` that maps continuous features to simplex distributions over `K` learnable prototypes; the loss becomes KL alignment between simplex distributions, with bounded convex-hull geometry as a non-collapse certificate.
- **[TS-JEPA](../papers/ts-jepa.md)** is the minimal recipe: tokenizer + encoder + predictor + EMA, L1 loss, no decoder, no codebook. The cleanest reference for ablating "is JEPA alone competitive on TS?"

## Why it works

The latent-space loss is bounded above by what the target encoder retained. When a series contains heavy-tailed noise or confounding factors, the input-space MSE used by [masked encoders](masked-encoder.md) penalizes the model for not reconstructing the noise; the JEPA loss does not — the target encoder has already discarded it. Empirically this shows up as JEPA outperforming MAE on classification benchmarks where the input is noisy ([TS-JEPA](../papers/ts-jepa.md) Table 1: FordA 91.5% vs MAE 85.1%, ECG5000 89.5% vs MAE 91.6% — close on clean signals, JEPA wins on noisy ones).

Decoupling prediction from generation also frees the encoder to learn predictability-optimized features. A JEPA-trained encoder will keep "this window is in a 'rising trend' regime" and drop "this specific high-frequency wiggle is at sample t=4173." The MTS-JEPA paper makes this explicit by using a discrete codebook to crystallize the regime structure, and demonstrates that a small subset of codebook entries activate disproportionately on anomalous windows (paper Figure 4) — the latent space has learned regime prototypes the loss never asked for.

The EMA + stop-gradient bookkeeping is what keeps the loop trainable. Without it, the network's loss function admits a trivial fixed point: collapse all encodings to a single constant. The EMA gives the predictor a slowly-moving target that the online encoder cannot trivially match by collapsing. MTS-JEPA additionally argues that even with EMA, JEPA can fail silently on continuous time-series, and adds the codebook as a geometric constraint that enforces a strictly positive lower bound on batch variance (paper Appendix A.3).

## Trade-offs and failure modes

The single largest failure mode is **representation collapse**: encoder learns the constant mapping, loss drops monotonically, downstream metrics flatline. EMA + stop-gradient is the standard recipe. [MTS-JEPA](../papers/mts-jepa.md) demonstrates that on continuous TS this is sometimes insufficient and the soft codebook is needed as backup.

The second is the **training-evaluation gap**. JEPA gives a representation, not a forecast. Forecasting evaluation requires either an attached decoder ([LaT-PFN](../papers/lat-pfn.md), [MTS-JEPA](../papers/mts-jepa.md)) or downstream finetuning of a regression head ([TS-JEPA](../papers/ts-jepa.md) for short-term, autoregressive rollout for long-term). This is unlike the [decoder-only autoregressive](decoder-only-autoregressive.md) family, where forecasting is the training objective directly.

The third is **benchmark mismatch**. The 2024 TS-FM cohort ([TimesFM](../papers/timesfm.md), [Chronos](../papers/chronos.md), [MOIRAI](../papers/moirai.md), [Time-MoE](../papers/time-moe.md), [Sundial](../papers/sundial.md)) is benchmarked on Monash, GIFT-Eval, Chronos Benchmark II — none of which the JEPA-for-TS papers report. This makes head-to-head comparison with the dominant TS-FM lineage indirect, and is the cleanest gap to close in 2026.

A more subtle issue is that JEPA representations are *predictability-optimized* but not necessarily *forecast-optimized*. [TS-JEPA](../papers/ts-jepa.md) loses to autoregressive on short-term forecasting (next-patch MSE) on all three of its forecasting datasets — this is the regime forecasting-trained TS-FMs optimize for directly, and JEPA gives it up by construction.

## Siblings and design space

Compared to [masked-encoder](masked-encoder.md), JEPA replaces input-space reconstruction with latent-space prediction; the architectural skeleton (tokenizer, transformer, masking, position encoding) is identical, only the target and the loss change. Compared to [decoder-only autoregressive](decoder-only-autoregressive.md), JEPA gives up the direct forecasting objective for a representation that transfers better to noisy classification and anomaly detection, at the cost of a benchmark-comparison gap on Monash / GIFT-Eval. Compared to [contrastive representation learning](../concepts/contrastive-representation-learning.md) (CPC, TS2Vec, TNC, T-Loss), JEPA does not need negative samples — the EMA target encoder takes their place — and the loss is bounded by what the target retained rather than by `log N` (the number of negatives). All three families (masked-encoder, JEPA, contrastive) are non-forecasting pretraining objectives that *can* serve forecasting downstream; the right choice depends on signal-to-noise ratio and how much the downstream task cares about high-frequency detail.

JEPA also overlaps with the [value quantization](../concepts/value-quantization.md) cluster via [MTS-JEPA](../papers/mts-jepa.md)'s codebook — the soft codebook is structurally a VQ-VAE-style discretization, but its role is collapse prevention and regime detection rather than tokenization.

## Design choices in the literature

- **[LaT-PFN](../papers/lat-pfn.md)** (Verdenius et al., 2024-05) — JEPA + PFN. Embedder is dilated MobileNet1D (CNN). Predictor is a PFN transformer with 2D attention over examples × sequence and cross-attention to a learned average-pooled context summary. Separate decoder over 100 categorical bins, trained with stop-gradient. System-identification regularization head regresses synthetic-prior simulation parameters. Trained exclusively on synthetic data with a context-aware triple-sampling prior. 24 hours on a single A10G per seed.
- **[TS-JEPA](../papers/ts-jepa.md)** (Ennadir, Golkar, Sarra, 2025-09) — minimal four-component JEPA. 1D-CNN tokenizer, transformer encoder + predictor (2 heads, dim 128), EMA target encoder (m=0.998). 70% mask ratio, L1 latent loss. Frozen-encoder + small head for evaluation. The cleanest reference for the ablation "JEPA alone vs MAE alone vs contrastive alone vs autoregressive alone, all at matched encoder capacity." Single V100.
- **[MTS-JEPA](../papers/mts-jepa.md)** (He et al., 2026-02) — JEPA + multi-resolution + soft codebook. Online encoder is a residual CNN tokenizer + transformer; EMA target encodes both fine (P=5 patches of length L=20) and coarse (P=1 downsampled patch). Soft codebook of K prototypes with temperature-scaled cosine similarity, EMA-updated. Dual-branch predictor: fine (transformer) + coarse (cross-attention with learnable query). Auxiliary decoder with RevIN-normalized reconstruction. Loss = `L_pred + L_code + λ_r L_rec` with `L_pred = λ_f (L_KL_fine + γ L_MSE_fine) + λ_c L_KL_coarse`.

## Open questions

- **Scaling.** No JEPA-for-TS paper has trained a checkpoint anywhere near the [Time-MoE](../papers/time-moe.md) (2.4B activated) or [Timer-S1](../papers/timer-s1.md) (8.3B sparse) brackets. Whether the recipe scales is an open question, and it gates whether JEPA can overtake forecasting-trained TS-FMs on the dominant benchmarks.
- **Optimal asymmetry between encoder and predictor.** I-JEPA uses a small predictor relative to the encoder; TS adaptations use roughly symmetric architectures. The right asymmetry for time series is unexplored.
- **Codebook necessity.** [MTS-JEPA](../papers/mts-jepa.md) shows the codebook is essential on the four anomaly benchmarks; [TS-JEPA](../papers/ts-jepa.md) does not need it on its five classification + three forecasting datasets. Is the codebook necessary for stability on continuous TS in general, or specific to the anomaly-prediction regime where regime structure is genuinely discrete?
- **Latent loss norm.** TS-JEPA picks L1 over L2 empirically; no analysis. MTS-JEPA picks KL on the simplex. Which is the right default?
- **Probabilistic outputs.** Adding a Chronos-style tokenized cross-entropy head to a JEPA-pretrained backbone is unexplored.
- **Cross-variate JEPA targets.** All three current papers use channel-independent targets; cross-variate latent prediction is unstudied.

## Related wiki pages

- [Joint-embedding predictive architecture concept](../concepts/joint-embedding-predictive-architecture.md) — the principle, decoupled from architecture details.
- [Masked encoder](masked-encoder.md) — the input-space sibling. JEPA is "masked encoder, but reconstruct in latent space."
- [Decoder-only autoregressive](decoder-only-autoregressive.md) — the other dominant TS-FM family.
- [Contrastive representation learning](../concepts/contrastive-representation-learning.md) — the third non-reconstructive self-supervision family; JEPA's nearest cousin.
- [Patch tokenization](../concepts/patch-tokenization.md) — input pipeline shared with masked-encoder.
- [Value quantization](../concepts/value-quantization.md) — relevant to the [MTS-JEPA](../papers/mts-jepa.md) soft codebook.
- [Lightweight non-transformer](lightweight-non-transformer.md) — relevant to [LaT-PFN](../papers/lat-pfn.md)'s dilated MobileNet1D embedder.
