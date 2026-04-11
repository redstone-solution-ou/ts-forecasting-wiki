# LLM Reprogramming

## Intuition

LLM reprogramming is the family of approaches that takes a pretrained large language model — GPT-2, Llama, BERT, BEiT — and turns it into a time-series forecaster by wrapping it in lightweight TS-specific adapters, or by encoding numeric values directly as text. The motivation is simple: LLMs already carry strong sequence priors learned from trillions of text tokens, and time-series foundation models are starved for data. If the LLM's learned "how sequences flow" transfers to numeric sequences, you can skip expensive TS pretraining and piggyback on a frozen backbone. The empirical finding — that this *works*, sometimes surprisingly well — is one of the more unexpected results in the field.

## Mechanics

There are three recognizable strategies, differing in how much of the LLM they touch.

**1. Reprogramming layer (Time-LLM).** Keep a frozen Llama backbone. Patch the input series and project patches into the LLM's word-embedding subspace via a learned cross-attention against a small set of *text prototypes*:

```
# TS side
patches  = patch_and_embed(x)                 # (N, D_ts)
# Text prototype bank: a learned subset of the LLM's vocab embeddings
V_proto  = learned_select(LLM_word_embeddings, k=1000)  # (K, D_llm)
# Cross-attention: each patch queries text prototypes, retrieves a mixture
Q        = patches @ W_q                      # (N, D_llm)
K, V     = V_proto @ W_k, V_proto @ W_v
patch_in_word_space = softmax(Q @ K.T / sqrt(D)) @ V   # (N, D_llm)

# Prompt-as-Prefix: prepend a natural-language description
prefix   = tokenize("This is an hourly electricity load series with weekly seasonality.")
seq      = concat(LLM_embed(prefix), patch_in_word_space)
z        = FrozenLLM(seq)                     # LLM unchanged
y_hat    = forecast_head(z[-L:])
```

Only the cross-attention, the prompt, and the forecast head are trained. The billions of Llama parameters stay frozen, so gradient descent touches a few million parameters at most.

**2. Frozen backbone + minimal fine-tune (GPT4TS / OFA / FPT).** Even smaller footprint: take GPT-2 (or BERT or BEiT), freeze almost everything, and fine-tune *only* the layer-norm parameters and positional embeddings:

```
for param in LLM.parameters():
    param.requires_grad = False
for layer in LLM.layers:
    layer.ln1.requires_grad_(True)
    layer.ln2.requires_grad_(True)
LLM.position_embeddings.requires_grad_(True)
# plus a fresh input projection and output head
```

That is it — no adapter, no prompt, just LN-and-pos-emb tuning. Yet across six TS tasks the result is competitive or SOTA.

**3. Numbers-as-text (LLMTime).** Do not adapt the LLM at all. Serialize numbers as ASCII digit strings with explicit handling of sign, decimal point, and digit grouping:

```
12345.67 -> "1 2 3 4 5 . 6 7"
x_series -> " , ".join(format_digits(v) for v in x_series)
```

Feed the resulting string to a frozen GPT-3 or Llama-2, let it generate more digits, and parse them back. A discrete-to-continuous conversion turns the LLM's per-token distribution over digit positions into a proper continuous density over numeric outcomes.

## Why it works

The deepest question in reprogramming is *why an LLM trained on text says anything useful about a numeric series at all*. Two complementary explanations are currently on offer.

The first is **sequence-modeling generality**: an LLM trained on trillions of tokens has learned a powerful prior over "how sequences evolve" — trends, autocorrelations, local dependencies — that is not actually tied to the identity of the tokens. Whether a token is the word "spring" or the embedding of a numeric patch, the transformer's attention circuits can still track its relationship to earlier tokens. The GPT4TS paper draws a concrete connection between self-attention blocks and principal component analysis, arguing that each attention layer performs a kind of data-dependent PCA on its inputs, which is a perfectly reasonable operation for TS even though it was learned on text.

The second is **embedding-space alignment**: LLM word embeddings span a rich, high-dimensional geometry. Time-LLM's reprogramming layer does not require *any specific word* to correspond to "trend"; it only needs the cross-attention to find a mixture of prototype embeddings that puts each TS patch in a well-conditioned region of the LLM's input space. Once there, the frozen attention stack processes it as usual.

LLMTime is the strongest empirical signal: zero adapters, zero TS training, and yet competitive zero-shot accuracy. This suggests the LLM's next-token prediction circuit *already understands numbers as sequences* at a certain level, even without any TS-specific alignment — plausibly because the pretraining corpus itself contains a lot of numeric data (tables, stock prices, scientific text).

## Trade-offs and failure modes

Reprogramming is parameter-efficient but comes with strong limits. Time-LLM's cross-attention bandwidth — the dimensionality of the alignment layer — is a bottleneck: if it is too small, TS patches collapse into a few prototypes and the LLM loses the shape information. The size of the LLM backbone also matters; GPT4TS's arguments break down on very small LLMs (say GPT-2 small) because the attention circuits are not expressive enough to repurpose.

LLMTime is conceptually elegant but expensive at token budget: every number costs 3–10 tokens depending on precision, and long histories saturate the LLM's context window fast. It also inherits the LLM's BPE idiosyncrasies — a tokenizer that splits "12345" differently from "12346" can hurt calibration.

The bigger limitation is ceiling: reprogrammed models generally do not beat purpose-built TS FMs at the frontier. Time-LLM, GPT4TS, and LLMTime are competitive on standard benchmarks but [TimesFM](../papers/timesfm.md), [MOIRAI](../papers/moirai.md), and [Chronos](../papers/chronos.md) typically win on leaderboard position. The argument for reprogramming is more about parameter efficiency and data efficiency than peak accuracy.

## Siblings and design space

Compared to `[Decoder-only autoregressive](decoder-only-autoregressive.md)` built from scratch, reprogramming is much cheaper to train but has less headroom. Compared to `[Masked encoder](masked-encoder.md)` TS FMs, reprogramming does not require any TS-specific pretraining data.

## Design choices in the literature

- `[Time-LLM](../papers/time-llm.md)` — reprogramming layer aligning patched TS embeddings with LLM word embeddings via cross-attention to text prototypes, plus a Prompt-as-Prefix natural-language context. Frozen Llama backbone; only the reprogramming module and forecast head train. ICLR 2024.
- `[GPT4TS / OFA / FPT](../papers/gpt4ts.md)` — "One Fits All": frozen GPT-2/BERT/BEiT with only layer-norm and positional embeddings trained, reaching competitive or SOTA across six TS tasks (forecast, classification, anomaly, imputation, short-term, long-term). Argues self-attention as implicit PCA as a theoretical justification. NeurIPS 2023.
- `[LLMTime](../papers/llmtime.md)` — numbers-as-text ASCII tokenization on vanilla GPT-3 / Llama-2, with careful handling of sign, scale, and digit grouping. A discrete-to-continuous density conversion yields calibrated probabilistic forecasts. Zero adapters, zero TS-specific training. NeurIPS 2023.

## Open questions

- **How does reprogramming scale with LLM size?** GPT4TS hints that bigger frozen LLMs transfer better, but no systematic sweep exists.
- **What, exactly, transfers from text?** The PCA argument is partial. Is it next-token prediction circuitry, long-range attention patterns, or learned positional inductive biases?
- **Can reprogramming beat purpose-built TS FMs with *enough* alignment training?** Time-LLM's ceiling is unclear.
- **Covariate and panel reprogramming.** Current recipes are univariate; how to ingest multivariate or exogenous inputs through an LLM backbone is open.
- **Tokenization of numbers for LLMs.** LLMTime's BPE-digit hack works but is not optimal; a learned numeric tokenizer on top of a frozen LLM is untried.

## Related wiki pages

- [Value quantization](../concepts/value-quantization.md)
- [Patch tokenization](../concepts/patch-tokenization.md)
- [Zero-shot forecasting](../concepts/zero-shot-forecasting.md)
- [Multi-task universal](../concepts/multi-task-universal.md)
- [Decoder-only autoregressive](decoder-only-autoregressive.md)
- [Encoder-decoder T5](encoder-decoder-t5.md)
