# Masked Encoder

## Intuition

A masked encoder is a bidirectional transformer pretrained to reconstruct randomly masked patches of a time series from their visible neighbors — BERT for numbers. Because attention is non-causal, every position sees the full visible context to the left *and* to the right, which gives the family a natural fit for imputation, anomaly detection, and classification. With a lightweight forecasting head tacked on top (or by appending "future" mask tokens and letting the encoder fill them in), the same backbone also produces forecasts — usually with a direct multi-horizon output instead of autoregressive rollout, which sidesteps compounding error.

## Mechanics

End-to-end forward pass for a typical masked TS encoder:

```
# Inputs: x in R^{C x T} (C channels, T timesteps), P = patch length, D = hidden
x_norm, stats = RevIN_forward(x)                       # per-instance, per-channel
# channel independence: each of C channels processed separately
x_flat        = x_norm.reshape(B*C, T)
patches       = unfold(x_flat, size=P, stride=P)       # (B*C, N, P)
tokens        = patches @ W_in + b_in + pos_emb        # (B*C, N, D)
# masking: replace a random m% of tokens with a learned [MASK] embedding
mask_ids      = sample_random_positions(N, ratio=m)
tokens[mask_ids] = mask_emb
# bidirectional attention stack (no causal mask)
for layer in 1..L:
    tokens = tokens + MHA(LN(tokens))                   # full attention
    tokens = tokens + FFN(LN(tokens))
# pretraining loss: reconstruct the masked patches
recon         = tokens[mask_ids] @ W_out + b_out        # (|mask|, P)
loss          = MSE(recon, original_patches[mask_ids])
```

For forecasting, the standard trick is to append `L / P` future mask tokens to the end of the input sequence and read out the predictions at those positions after the final layer:

```
future_toks = [mask_emb] * (L / P)
full_seq    = concatenate(context_tokens, future_toks)
z           = Transformer(full_seq)
y_hat       = forecast_head(z[-(L/P):])                 # (L/P, P)
```

`[MOIRAI](../papers/moirai.md)` goes further and attaches a *mixture-of-Student-t* head at the future positions, yielding a calibrated probabilistic forecast in one forward pass.

Two MOIRAI-specific tricks deserve mention. First, **multi-patch-size projections**: instead of one `W_in`, MOIRAI keeps several projections keyed to patch sizes (e.g. 8/16/32/64/128); the tokenizer picks the right one based on the series sampling frequency. Second, **any-variate attention**: by keeping track of a variate index per token, the same architecture handles univariate, multivariate, and covariate inputs without per-layout retraining.

## Why it works

The masked reconstruction objective teaches the model `p(masked | visible)` — a fully bidirectional conditional — whereas causal next-patch prediction only teaches `p(future | past)`. For non-forecasting tasks (imputation is literally the training objective; anomaly detection can score reconstruction error; classification pools `z` and attaches a head) bidirectional context is strictly more informative, which is why masked encoders dominate the multi-task story.

For forecasting, the trade is more subtle. A masked encoder that processes "history plus future mask tokens" in one shot emits the entire horizon simultaneously, which *eliminates* the autoregressive compounding that plagues decoder-only models at long horizons. The cost is that it has to learn how to fill in future positions specifically, which requires training-time masking patterns that include trailing masks (not just random masks). MOIRAI's training procedure deliberately varies the mask pattern so the model can handle both imputation and forecasting from the same weights.

Channel independence combines naturally with masked encoding: each channel's patches are processed as a separate sequence, so the model learns a single shared shape-modeling prior that applies to any variate regardless of unit system. This is what enables a single MOIRAI checkpoint to handle any number of variates at inference.

## Trade-offs and failure modes

The main drawback is the *training-inference mismatch* that bidirectional pretraining introduces for forecasting. During pretraining, the model sees randomly masked positions surrounded by visible context on both sides; during forecasting, the future is *entirely masked* with no right-side visible context. Training recipes partially compensate by deliberately masking long trailing blocks, but the mismatch can still hurt long-horizon accuracy relative to decoder-only models that were trained exactly for rollout.

Masked encoders are also less sample-efficient per token than decoder-only models: only a fraction (typically 15–40%) of positions contribute to the loss at each step, whereas a decoder-only model scores every position. This is partly why MOIRAI needs [LOTSA](../datasets-benchmarks/lotsa.md) (~27B observations) to train competitively.

Channel independence itself is a trade-off. It is the reason for the architecture's flexibility, but it discards cross-variate structure — a dominant variate's information cannot directly inform another variate's forecast unless something like MOIRAI's any-variate attention is layered on top.

## Siblings and design space

Compared to `[Decoder-only autoregressive](decoder-only-autoregressive.md)`, masked encoders are stronger on imputation, anomaly detection, and classification, but weaker at pure long-horizon forecasting unless augmented with trailing-mask training. Compared to `[Encoder-decoder T5](encoder-decoder-t5.md)`, masked encoders lack a dedicated decoder so they read out predictions directly from encoder positions; this is simpler but gives up some clean history/horizon decoupling. Compared to `[Mixture-of-experts](mixture-of-experts.md)` variants like Moirai-MoE, vanilla masked encoders are dense and cap out at ~400M parameters before diminishing returns.

## Design choices in the literature

- `[MOMENT](../papers/moment.md)` — T5-initialized encoder weights + PatchTST patching + RevIN, masked reconstruction pretraining on the [Time Series Pile](../datasets-benchmarks/time-series-pile.md), task-specific heads for forecast/classify/anomaly/impute. Three sizes (40M/125M/385M) demonstrating a dense masked-encoder scaling curve.
- `[MOIRAI](../papers/moirai.md)` — built from scratch on LOTSA (~27B observations), introduces multi-patch-size projections for frequency specialization and any-variate attention for unified univariate/multivariate/covariate inputs; mixture-of-Student-t head for calibrated probabilistic output.
- `[Moirai-MoE](../papers/moirai-moe.md)` — drops the frequency-keyed multi-projection in favor of a *single* projection plus sparse token-level routing, arguing that frequency specialization is better discovered than engineered. Reports SOTA on 39 datasets while activating fewer parameters per token than dense Moirai-Large.

## Open questions

- **Is trailing-mask training enough to close the forecasting gap with decoder-only?** The head-to-head evidence is mixed and highly data-dependent.
- **Optimal masking ratio for forecasting-first masked encoders.** BERT uses 15%; PatchTST uses higher ratios; no systematic study for TS FMs.
- **Any-variate attention scaling.** How does the same mechanism behave with hundreds or thousands of variates? Most experiments top out at ~tens.
- **Cross-variate information flow.** Channel independence is safe but leaves information on the table; any-variate and Moirai-MoE start to address this but without clean ablations.
- **Bidirectional pretraining vs causal pretraining for zero-shot transfer.** MOIRAI and [TimesFM](../papers/timesfm.md) are close on benchmarks; which prior is better is domain-dependent.

## Related wiki pages

- [Patch tokenization](../concepts/patch-tokenization.md)
- [RevIN normalization](../concepts/revin-normalization.md)
- [Probabilistic forecasting](../concepts/probabilistic-forecasting.md)
- [Multi-task universal](../concepts/multi-task-universal.md)
- [In-context learning](../concepts/in-context-learning.md)
- [Decoder-only autoregressive](decoder-only-autoregressive.md)
- [Mixture-of-experts](mixture-of-experts.md)
- [Encoder-decoder T5](encoder-decoder-t5.md)
