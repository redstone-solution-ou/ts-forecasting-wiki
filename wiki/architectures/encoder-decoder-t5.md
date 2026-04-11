# Encoder-Decoder (T5-style)

## Intuition

An encoder-decoder T5-style TS foundation model treats forecasting as a sequence-to-sequence problem: the encoder consumes a history window and produces a contextual representation, and a decoder autoregressively emits the forecast horizon while cross-attending to the encoder output. The attraction is pragmatic: unmodified T5 code, unmodified tokenizer interface, unmodified training loop. Chronos in particular makes the famous design choice to leave the T5 *completely untouched* and change only the tokenizer — which means you get decades of LLM engineering (checkpoints, HuggingFace integration, distributed training, sampling strategies) in exchange for writing a quantizer.

## Mechanics

Forward pass of a Chronos-style encoder-decoder TS FM:

```
# Encoder side: tokenize history
s          = mean(|x_hist|) + eps                   # per-series scale
x_tilde    = clip(x_hist / s, -C, C)
bin_ids    = quantize_uniform(x_tilde, V)           # integer tokens in 0..V-1
ctx_tokens = embed(bin_ids)                         # (N_ctx, D)
enc_out    = T5Encoder(ctx_tokens)                   # (N_ctx, D)

# Decoder side: autoregressive horizon generation
dec_tokens = [<BOS>]
for step in 1..N_horizon:
    z          = T5Decoder(dec_tokens, cross_attn_kv=enc_out)
    logits     = z[-1] @ W_vocab                     # (V,)
    next_bin   = sample(softmax(logits))
    dec_tokens = append(dec_tokens, next_bin)

# Dequantize and rescale
y_hat_tokens = dec_tokens[1:]                        # drop BOS
y_hat_real   = bin_centers[y_hat_tokens] * s         # back on the original scale
```

At inference, `M` independent trajectories are sampled in parallel; the empirical distribution across trajectories yields quantiles or prediction intervals. The whole pipeline is *architecturally identical* to T5 applied to text — only the vocabulary semantics differ.

`[Chronos-2](../papers/chronos-2.md)` is an important twist: it drops the T5 decoder entirely in favor of a **120M encoder-only** patch-based transformer augmented with a **group-attention** mask that lets multiple related series (panel siblings, exogenous covariates) attend to each other in a single forward pass. A dedicated **quantile decoder head** emits a fixed set of quantile levels per horizon step, trained with pinball loss:

```
# Group attention: panel of S related series, each with N patches
tokens = [patch_embed(series_s) for s in 1..S]   # (S*N, D)
mask   = within_series_causal OR same_time_across_series
z      = Transformer(tokens, mask)
for step in 1..L, alpha in {0.1, 0.5, 0.9, ...}:
    q_alpha[step] = QuantileHead(z_target[step], alpha)
```

This keeps the *spirit* of encoder-decoder (clean history-to-horizon decoupling, one-shot probabilistic output) while removing both the discrete vocabulary and the autoregressive decoder.

## Why it works

The encoder-decoder split is conceptually clean: the encoder is allowed to spend as much capacity as it wants on understanding the history, and the decoder focuses on generating the horizon conditioned on that understanding via cross-attention. For Chronos, this split lets an unmodified T5 carry almost all of the heavy lifting — the quantizer is just a translator between the real line and T5's input space. Categorical cross-entropy over the quantized vocabulary gives the model a fully probabilistic training signal and fully probabilistic inference, for free, without attaching any parametric head.

The approach also inherits T5's scaling behavior: Chronos trains at 20M, 60M, 200M, and 710M and observes monotonic improvements with size, giving the cleanest published dense scaling curve for an encoder-decoder TS FM. Because the T5 architecture is untouched, every optimization, compiler trick, and distributed-training pattern from language modeling transfers.

Chronos-2's encoder-only refinement takes the position that the decoder was the wrong place to spend parameters: with group attention providing in-context learning at the encoder level and a quantile head handling the output, you do not need autoregressive decoding at all. This gives you direct multi-horizon probabilistic output in one forward pass — no sampling, no dequantization, and no precision ceiling.

## Trade-offs and failure modes

Chronos's approach inherits the trade-offs of value quantization: precision is capped at the bin width, mean-scaling is fragile for intermittent or near-zero series, and the fixed vocabulary must be set at pretraining time. Categorical sampling also costs `M×` the inference compute of a deterministic decoder, because probabilistic output requires `M` independent rollouts. The autoregressive decoder is subject to the same compounding-error problem as any decoder-only model at long horizons.

Chronos-2's encoder-only design sidesteps most of these issues but loses the clean "language-model drop-in" story: you cannot reuse T5 checkpoints, you need bespoke attention masks, and the quantile head replaces the universal sampling interface with a fixed set of levels — you can no longer answer arbitrary distributional queries without interpolation.

Both approaches are fundamentally bound by *encoder capacity*: the encoder's output is the only channel through which the decoder or head can learn about the history, so the encoder has to be expressive enough to summarize long contexts. At very long histories (thousands of tokens), the encoder self-attention cost becomes a bottleneck.

## Siblings and design space

Compared to `[Decoder-only autoregressive](decoder-only-autoregressive.md)`, encoder-decoder is philosophically cleaner (history and horizon are separated) but costs more parameters for equal capacity because the encoder and decoder are separate stacks. Compared to `[Masked encoder](masked-encoder.md)`, Chronos's T5 is fully autoregressive at the decoder, which matches training and inference exactly at the cost of bidirectional imputation capability; Chronos-2's encoder-only variant is closer to a masked encoder with a direct quantile head.

## Design choices in the literature

- `[Chronos](../papers/chronos.md)` — unmodified T5 at 20M/60M/200M/710M sizes, mean-scaling plus uniform quantization as the tokenizer, categorical cross-entropy training on 42 datasets, probabilistic inference via categorical sampling and dequantization. KernelSynth and TSMix augmentation in the pretraining mix.
- `[Chronos-2](../papers/chronos-2.md)` — 120M encoder-only patch transformer with group attention for in-context learning across panels, multivariate, and covariate inputs; quantile decoder head for direct multi-horizon probabilistic output. Heavy use of synthetic data to construct sibling panels for ICL training.
- `[TimeGPT-1](../papers/timegpt.md)` — earliest commercial TS foundation model, closed-source encoder-decoder transformer exposed via API, probabilistic via conformal prediction on top of a point forecaster.

## Open questions

- **Is encoder-decoder worth the parameter overhead?** Decoder-only FMs ([TimesFM](../papers/timesfm.md), [Timer](../papers/timer.md)) close the performance gap with simpler architectures and lower FLOPs.
- **Quantile head vs sampling.** Chronos-2 argues quantile regression is cleaner than categorical sampling; the [CRPS](../evaluation/metrics.md#21-crps--continuous-ranked-probability-score) comparison at matched compute has not been published cleanly.
- **Group attention and scale.** Can group attention extend to hundreds of siblings, or does the `O((SN)^2)` cost limit it to small panels?
- **Fine-tuning recipes.** Chronos's 710M variant is still small by LLM standards; whether fine-tuning it to domain data is worthwhile vs zero-shot is under-studied.
- **Continuous-output encoder-decoder.** What happens if you replace Chronos's quantizer with a flow-matching head — encoder-decoder continuous TS models are almost unexplored.

## Related wiki pages

- [Value quantization](../concepts/value-quantization.md)
- [Probabilistic forecasting](../concepts/probabilistic-forecasting.md)
- [In-context learning](../concepts/in-context-learning.md)
- [Synthetic data augmentation](../concepts/synthetic-data-augmentation.md)
- [Zero-shot forecasting](../concepts/zero-shot-forecasting.md)
- [Decoder-only autoregressive](decoder-only-autoregressive.md)
- [Masked encoder](masked-encoder.md)
