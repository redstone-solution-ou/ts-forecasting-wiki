# Decoder-Only Autoregressive

## Intuition

A decoder-only autoregressive TS foundation model is a GPT-style causal transformer whose "tokens" are numeric patches (or quantized bins) rather than subwords. The model factorizes the joint distribution of a series as a product of left-to-right conditionals, and training is pure next-token prediction. The appeal is twofold: first, it gives you the *entire* LLM engineering stack for free — KV-cache inference, rotary position embeddings, flash attention, sparse MoE — and second, the training objective is the most data-efficient one we know (every token position contributes a loss), which is exactly what you want when pretraining on tens of billions of points.

## Mechanics

End-to-end forward pass for a typical decoder-only patched TS FM:

```
# Inputs: x in R^T (one series), P_in = input patch, P_out = output patch, D = hidden
x_norm, stats = RevIN_forward(x)                      # per-instance standardization
patches       = unfold(x_norm, size=P_in, stride=P_in) # (N, P_in), N = T / P_in
tokens        = patches @ W_in + b_in + pos_emb       # (N, D)
# causal self-attention stack
for layer in 1..L:
    tokens = tokens + CausalMHA(LN(tokens))            # mask: no peek forward
    tokens = tokens + FFN(LN(tokens))                  # dense or MoE
# per-position output: decode a patch of future values
y_patch       = tokens @ W_out + b_out                # (N, P_out)
# training: token at position i predicts the next P_out values
loss          = MSE(y_patch[:-1], next_patches)        # or NLL under a parametric head
```

Key moves:

- **Patch-level autoregression.** Each token represents `P_in` consecutive timestamps; each output predicts the next `P_out` timestamps. Standard LLMs have `P_in = P_out = 1` (one subword in, one out); TimesFM uses `P_in = 32, P_out = 128` so one token emits four patches of future values. This shortens the rollout by the factor `P_out / P_in`, reducing exposure bias and wall-clock.
- **Causal mask.** Position `i` attends only to positions `<= i`. KV-cache at inference makes rollout `O(N)` per new step rather than `O(N^2)`.
- **Output head.** Three choices: (a) linear regression head + MSE, (b) parametric distribution head (Student-t, mixture) + NLL, or (c) categorical over a quantized vocabulary + CE. [Chronos](../papers/chronos.md) is the exception that uses encoder-decoder with (c); most decoder-only models use (a) or (b).

Rollout for an `L`-step horizon:

```
tokens = encode(history)
for step in 1..ceil(L / P_out):
    z_new   = decoder_step(tokens)           # uses KV-cache, O(N) per step
    patch   = decode_head(z_new)             # (P_out,)
    tokens  = append(tokens, embed(patch))
```

## Why it works

The factorization `p(x_1, ..., x_T) = \prod_t p(x_t | x_{<t})` is exact, so training one conditional at a time captures the full joint. This gives three advantages over direct multi-horizon regression. First, the training signal is dense: every position contributes to the loss, so an `N`-token sequence gives `N` supervision pairs, which is `N×` more data-efficient than a single history→horizon pair. Second, the model can emit *arbitrary* horizons by rolling out; horizon is not baked into the architecture. Third, the resulting model supports in-context use naturally — just concatenate context and let the causal mask do the rest — which is why Timer-XL's multivariate flattening works at all.

The causal mask also provides a free implicit regularizer: the model cannot cheat by peeking at future values, so it has to learn the true conditional. Compared to masked-encoder reconstruction, which lets the model see bidirectional context, the causal setup is harder but transfers more cleanly to forecasting because the training and inference distributions match exactly.

Patch-level output (`P_out > P_in`) is a subtle but important refinement. Classical 1-step-ahead decoders suffer from exposure bias: training sees teacher-forced history, inference sees model-generated history, and errors compound. Longer output patches reduce the number of rollout steps and therefore reduce compounding, at the cost of coarser granularity within each emitted block.

## Trade-offs and failure modes

Autoregressive rollout has a compounding-error problem that is sharpest at long horizons: an error at step 10 shifts the input distribution seen at step 11, and the divergence can grow exponentially. Parametric heads with wide distributions help (the next-step uncertainty absorbs some drift), but cannot eliminate it. Direct multi-horizon heads (masked encoder with all horizon tokens at once, or flow-matching) avoid this but lose the dense training signal.

Causal attention also throws away future context at training time — in contrast, a masked encoder learns from bidirectional context, which is more sample-efficient for non-forecasting tasks like imputation. Decoder-only models are therefore often worse at imputation and classification than equivalent-size masked encoders, even if they are better at forecasting.

Another failure mode is *locked input granularity*: once you fix `P_in`, changing the user's sampling frequency at inference requires either resampling the series or having multiple projection heads keyed by frequency ([MOIRAI](../papers/moirai.md)'s trick) or routing ([Moirai-MoE](../papers/moirai-moe.md)'s). Pure decoder-only TS FMs with one projection struggle on off-training frequencies.

## Siblings and design space

Compared to `[Masked encoder](masked-encoder.md)`, decoder-only is better for forecasting rollout but worse for bidirectional tasks. Compared to `[Encoder-decoder T5](encoder-decoder-t5.md)`, decoder-only is simpler (no encoder cross-attention, smaller KV footprint) but loses the clean history-horizon decoupling; Chronos's encoder-decoder plus quantization is a different tradeoff in the same neighborhood. Compared to `[Lightweight non-transformer](lightweight-non-transformer.md)`, decoder-only is heavier but scales cleanly with MoE; [TTM](../papers/ttm.md) argues the opposite, that small non-transformer models can substitute.

## Design choices in the literature

- `[TimesFM](../papers/timesfm.md)` — ~200M decoder-only patched transformer with `P_in = 32`, `P_out = 128`, trained on a ~100B-point mixture of Google Trends, Wiki Pageviews, and synthetic data. The asymmetric output patch is the headline mechanical choice.
- `[Timer](../papers/timer.md)` — GPT-style decoder trained on a ~1B-point corpus under the Single-Series Sequence (S3) unified format, with emergent few-shot behavior at scale.
- `[Timer-XL](../papers/timer-xl.md)` — extends Timer to multivariate and covariate settings by flattening panels and using universal TimeAttention with 2D position embeddings, preserving causality along the time axis.
- `[Lag-Llama](../papers/lag-llama.md)` — first open decoder-only probabilistic TS FM, injects lag features as explicit covariates into the causal stream and attaches a Student-t head.
- `[Time-MoE](../papers/time-moe.md)` — 2.4B sparse MoE decoder-only trained on [Time-300B](../datasets-benchmarks/time-300b.md), scaling active and total parameters independently; the first billion-param TS FM and the main evidence that TS decoder-only scales like LLMs.
- `[Sundial](../papers/sundial.md)` — decoder-only backbone with a TimeFlow flow-matching head, continuous-valued probabilistic output.

## Open questions

- **Optimal input/output patch asymmetry.** TimesFM picked 32/128, but the right ratio is data-dependent and under-explored.
- **Does rollout compounding fundamentally limit long-horizon accuracy?** Direct multi-horizon heads ([Chronos-2](../papers/chronos-2.md), Sundial) bypass it; decoder-only rollout with sampling has to simulate many trajectories to recover calibration.
- **MoE inside a decoder-only TS FM.** Time-MoE shows scaling works, but routing strategies specific to temporal data have not been systematically explored.
- **Long-context tractability.** 4k–16k tokens is the current ceiling; longer contexts need SSM or linear-attention alternatives.
- **Native covariate support.** Most decoder-only TS FMs are univariate; Lag-Llama and Timer-XL start to address this but neither is the clear winner.

## Related wiki pages

- [Patch tokenization](../concepts/patch-tokenization.md)
- [Probabilistic forecasting](../concepts/probabilistic-forecasting.md)
- [Scaling laws](../concepts/scaling-laws.md)
- [Zero-shot forecasting](../concepts/zero-shot-forecasting.md)
- [Mixture-of-experts](mixture-of-experts.md)
- [Masked encoder](masked-encoder.md)
- [Encoder-decoder T5](encoder-decoder-t5.md)
