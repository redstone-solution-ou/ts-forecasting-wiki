# Joint-Embedding Predictive Architecture (JEPA)

A self-supervised pretraining recipe in which the model predicts the *latent* embedding of a masked or future portion of the input — produced by a target encoder — instead of reconstructing the input itself. JEPA explicitly separates the prediction objective from a generative decoder, so the encoder is forced to discard input-space detail that is unpredictable or irrelevant.

## Intuition

A masked auto-encoder asks: "given the visible patches, regenerate the masked pixels / values exactly." That objective rewards the encoder for keeping every fluctuation that could help the decoder match the input — including high-frequency noise and confounding factors. JEPA replaces "reconstruct the input" with "predict the *embedding* of the input under a target encoder." The target embedding is itself a learned, lossy compression: it has already discarded the unpredictable detail. So the predictor's job is to match a representation that lives at the right abstraction level, and the loss on noise terms is bounded by what the target encoder retained — which, in a healthy training run, is very little.

This shift was formalized by LeCun's 2022 vision document and operationalized by I-JEPA (image), V-JEPA (video), and A-JEPA (audio). The 2024–2026 time-series adaptations — [LaT-PFN](../papers/lat-pfn.md), [TS-JEPA](../papers/ts-jepa.md), [MTS-JEPA](../papers/mts-jepa.md) — port the same recipe to 1-D temporal sequences. The promise is consistent across modalities: latent prediction beats input-space reconstruction on downstream tasks where the input is noisy and where the goal is "high-level state," not "every sample."

## Mechanics

The canonical setup has four components plus a bookkeeping rule.

```
# Online (gradient-trained) branch
z_visible = E_θ(x_visible)                # online encoder
z_predicted_masked = P_β(z_visible)        # predictor — forecasts the masked latents

# Target branch (gradient-free)
with stop_gradient:
    z_target_masked = E_ξ(x_masked)        # target encoder = EMA of E_θ

# Loss is in latent space, not input space
L = || z_predicted_masked - z_target_masked || _{p}    # p = 1 (TS-JEPA), 2 (I-JEPA), KL on simplex (MTS-JEPA)

# Bookkeeping: EMA update prevents collapse
E_ξ ← m * E_ξ + (1 - m) * E_θ              # m ≈ 0.998 (TS-JEPA, MTS-JEPA)
```

The four pieces and the bookkeeping rule are not optional. Without `stop_gradient` and the EMA bookkeeping, the network has a trivial degenerate solution: collapse all inputs to a single constant vector, drive the loss to zero, learn nothing. The EMA + stop-gradient combination is what makes the "predict your own latents" loop train at all — see Bootstrap Your Own Latent (BYOL) and DINO for the broader self-distillation lineage. The `p = 1` choice in TS-JEPA is empirically more stable on noisy series than `p = 2`; MTS-JEPA goes further and replaces the metric with KL divergence over a simplex of codebook prototypes.

JEPA is *agnostic* to the encoder backbone (TS-JEPA and MTS-JEPA use transformers; LaT-PFN's embedder is a dilated MobileNet1D), to the masking strategy (uniform random in TS-JEPA, multi-resolution split in MTS-JEPA, contextual in LaT-PFN), and to whether there is a separate decoder for downstream reconstruction (LaT-PFN has one with stop-gradient; TS-JEPA does not).

## Why it works

Three forces. First, **target compression**: because the target encoder has itself discarded irrelevant detail, the loss is bounded above by what the target retained. Noise is not penalized; signal is. Second, **decoupled gradients**: the target encoder is updated by EMA, not by gradient descent, so the target is a slowly-moving fixed point. The predictor chases it without the loop instability that comes from co-training a target with its own predictor. Third, **information geometry**: predicting an embedding instead of an input means the loss surface lives in a much lower-dimensional, smoother space; the optimizer no longer wastes capacity on high-entropy input distributions like raw 16-bit waveform samples.

A subtle but important corollary: JEPA representations are *predictability-optimized*, not reconstruction-optimized. They will keep features that help anticipate the future and drop features that are stochastic but visually salient. For time series with strong seasonal structure plus heavy idiosyncratic noise (financial micro-structure, electricity grid telemetry, biomedical signals), this is exactly the right inductive bias.

## Trade-offs and failure modes

- **Representation collapse is the canonical failure mode.** Without EMA + stop-gradient (or some surrogate constraint like a codebook bottleneck — see [MTS-JEPA](../papers/mts-jepa.md)), the optimizer drives the encoder to a constant. Collapse is silent: the loss looks fine, the embeddings look dead.
- **No pixel / value-level outputs for free.** JEPA gives you a representation, not a forecast or a reconstruction. To use it for forecasting you need a decoder (LaT-PFN, MTS-JEPA) or downstream finetuning of a regression head (TS-JEPA).
- **Hyperparameter sensitivity.** EMA decay `m`, mask ratio, predictor capacity relative to encoder capacity, and the loss norm all matter. The literature has not converged on robust defaults: I-JEPA uses `m ∈ [0.996, 1.0]` schedules; TS-JEPA uses `m = 0.998`; MTS-JEPA EMA-updates both encoder and codebook.
- **Asymmetric capacity is part of the recipe.** I-JEPA uses a small predictor relative to the encoder; on TS, both TS-JEPA and MTS-JEPA use roughly symmetric architectures, and the right asymmetry is unexplored.
- **Loss does not directly measure downstream quality.** A JEPA model can show monotonically decreasing pretraining loss while downstream classification stagnates — the encoder may be learning to predict targets that don't separate the labels you care about.
- **Hard to compare against forecasting-trained TS-FMs.** [TimesFM](../papers/timesfm.md) / [Chronos](../papers/chronos.md) optimize a forecasting loss directly; their numbers on GIFT-Eval / Monash are not directly comparable to TS-JEPA's frozen-encoder + small-head numbers on FordA / ECG5000. The benchmark mismatch is real and has not been bridged.

## Design choices in the literature

- **[LaT-PFN](../papers/lat-pfn.md)** — combines JEPA with the Prior-data Fitted Network framework. Embedder is a dilated MobileNet1D (CNN, not transformer); predictor is a PFN transformer with cross-attention from a held-out prompt to a learned average-pooled context summary. Decoder is trained separately with stop-gradient. Adds a system-identification regularization head that regresses synthetic-prior simulation parameters.
- **[TS-JEPA](../papers/ts-jepa.md)** — the cleanest single-objective JEPA-for-TS baseline. Tokenizer (1D-CNN), encoder (transformer), predictor (transformer), EMA target encoder. 70% mask ratio, L1 loss, AdamW. No decoder — frozen encoder + small downstream head for evaluation.
- **[MTS-JEPA](../papers/mts-jepa.md)** — adds two extensions to the TS-JEPA template: (a) **multi-resolution targets** (the EMA branch produces both a fine patch-sequence target and a single coarse downsampled-window target; the online branch sees only fine but is supervised against both), and (b) a **soft codebook bottleneck** that maps continuous features to a simplex distribution over `K` prototypes. The codebook is the most distinctive design choice: it gives an analytical non-collapse bound and turns the prediction loss into a KL alignment between simplex distributions.
- Outside time series, the prototype JEPAs are I-JEPA (Assran et al., CVPR 2023, image patches), V-JEPA (Bardes et al., 2024, video tubes), and A-JEPA (audio mel-spectrogram patches). The TS adaptations inherit their EMA + stop-gradient bookkeeping verbatim.

## Open questions

- Does JEPA pretraining scale to billion-parameter TS-FMs the way [Time-MoE](../papers/time-moe.md) and [Timer-S1](../papers/timer-s1.md) do? No paper in the current cohort runs the test.
- Is the codebook bottleneck of [MTS-JEPA](../papers/mts-jepa.md) a robust collapse-prevention recipe, or is it specific to anomaly prediction's discrete-regime structure? Replicating on forecasting and classification would settle this.
- What is the right asymmetry between encoder and predictor capacity for time series? I-JEPA uses small predictors; TS adaptations have not searched.
- Can JEPA latent prediction be combined with a tokenized cross-entropy decoder (Chronos-style) to get a probabilistic forecasting head while keeping the latent-space pretraining advantage?
- For multivariate panels, should JEPA targets be per-channel or joint across channels? [MTS-JEPA](../papers/mts-jepa.md) uses channel-independent input formulation; cross-variate JEPA targets are unexplored.
- Why does L1 work better than L2 for the latent-space loss in [TS-JEPA](../papers/ts-jepa.md)? Empirical observation in the paper, no analysis.

## Papers that exemplify this

- **[LaT-PFN](../papers/lat-pfn.md)** — JEPA + PFN, dilated-CNN embedder, separate decoder, system-identification regularization, normalized abstract time axis.
- **[TS-JEPA](../papers/ts-jepa.md)** — pure four-component JEPA (tokenizer + encoder + predictor + EMA target), L1 latent loss, frozen-encoder evaluation. The cleanest reference for "is JEPA alone competitive on TS?"
- **[MTS-JEPA](../papers/mts-jepa.md)** — JEPA + multi-resolution predictive objective + soft codebook bottleneck, anomaly-prediction focus, KL loss on the simplex, analytical non-collapse bound.

## Related wiki pages

- [JEPA / latent-space prediction architecture](../architectures/jepa-latent-prediction.md) — the architecture-family hub for Cluster 8.
- [Contrastive representation learning](contrastive-representation-learning.md) — the *other* non-reconstructive self-supervision family; CPC and TS2Vec are the primary contrast points to JEPA.
- [Value quantization](value-quantization.md) — overlaps with the codebook bottleneck in [MTS-JEPA](../papers/mts-jepa.md).
- [In-context learning](in-context-learning.md) — the PFN side of [LaT-PFN](../papers/lat-pfn.md).
- [Multi-task universal](multi-task-universal.md) — the downstream-task setting JEPA representations are typically evaluated under.
- [Synthetic data augmentation](synthetic-data-augmentation.md) — JEPA pretraining on synthetic series ([LaT-PFN](../papers/lat-pfn.md)) is one of the active threads.
- [Masked-encoder architecture](../architectures/masked-encoder.md) — the input-space analog JEPA contrasts with.
