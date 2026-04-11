# Mixture-of-Experts

## Intuition

A mixture-of-experts (MoE) transformer replaces each dense feed-forward layer with a bank of `E` parallel expert MLPs plus a small router that picks the top-`k` experts per token. Only the selected experts run, so compute scales with the *active* experts rather than the total — you can have 10× the parameters of a dense model at the same FLOP budget. For time-series foundation models, MoE is the main route to billion-parameter scale without paying a billion-parameter inference cost, and it provides a natural mechanism for specializing parameters across frequencies, domains, or temporal regimes without hand-engineering.

## Mechanics

A sparse MoE layer replaces the FFN inside a transformer block:

```
# token x in R^D, E experts, top-k = 1 or 2
logits   = x @ W_router                                 # (E,)
top_k    = argmax_k(logits, k=2)                        # indices of chosen experts
weights  = softmax(logits[top_k])                       # (k,) re-normalized
out      = sum(weights[i] * FFN_i(x) for i in top_k)    # only k FFNs run
```

Training adds a load-balancing auxiliary loss to prevent collapse onto a few hot experts:

```
L_balance = E * sum_i (fraction_tokens_routed_to_i * mean_gate_prob_i)
L_total   = L_task + alpha * L_balance
```

The router is typically a single linear layer (tiny compared to an expert), and the experts themselves are standard FFNs — just many of them. Total parameters scale with `E`; active parameters per token scale with `k`. Time-MoE-2.4B activates roughly the dense-model FLOPs of a ~700M model per token but stores 2.4B parameters across experts.

Putting this inside a TS decoder looks like:

```
for layer in 1..L:
    x = x + CausalMHA(LN(x))     # attention is dense (shared across tokens)
    x = x + MoE_FFN(LN(x))        # sparse: k experts out of E per token
```

Attention is usually left dense because sparse attention tends to hurt quality; the MoE lives only in the FFN layers where the bulk of parameters sit anyway.

## Why it works

MoE decouples *capacity* from *compute*. A dense model spends every FLOP on every token; a sparse MoE model spends FLOPs only on the subset of experts that each token actually needs. If the data has latent structure — some tokens need "high-frequency expert," others need "trend expert" — the router discovers the split and routes accordingly. Over the training run, different experts specialize on statistically distinct subsets of the input distribution, and the effective model becomes a *soft mixture of specialist sub-models*.

For TS, this directly addresses the *heterogeneity problem*. A real corpus mixes minute-level stock ticks, hourly energy loads, daily retail, monthly macro, and yearly demographics — dynamics that a dense model has to compress into one shared FFN. An MoE can let different experts specialize on different regimes without being told which is which, which is exactly the hand-engineering that MOIRAI did with per-patch-size projections. Moirai-MoE's headline argument is precisely this: *let routing discover the specialization* rather than keying it on patch size.

From an optimization angle, sparse routing also provides a mild form of regularization: each expert only sees a subset of data, so over-specialization is limited by the load-balancing loss. The resulting models are often *more* sample-efficient than dense at matched FLOPs, which is part of why Time-MoE's scaling curves are favorable.

## Trade-offs and failure modes

The canonical failure is **expert collapse**: without strong load balancing, a few experts absorb most of the traffic and the rest are untrained. The fix is the auxiliary loss plus sometimes capacity limits (an expert can only accept up to `C` tokens per batch; overflow is dropped or sent elsewhere). Getting the capacity factor right is finicky; too low and you drop tokens, too high and load balancing degrades.

MoE also introduces **memory and communication overhead**: the total parameters must still fit in RAM, and distributed training needs all-to-all communication to route tokens to the right expert shard. This makes MoE training significantly more complex than dense. Inference is simpler (experts just live in memory) but still has higher RAM requirements than a dense model at the same FLOPs.

A more subtle issue is **routing instability during training**: early in training, the router is random, experts are random, and the feedback loop between routing and expert specialization is noisy. Warm-up schedules and careful learning-rate tuning are usually needed.

Finally, routing interpretability is genuinely limited. The hope is that "expert 5 handles high-frequency" but in practice expert identity is mostly entangled; ablations like "remove experts 5 and 6" rarely have clean effects.

## Siblings and design space

Inside `[Decoder-only autoregressive](decoder-only-autoregressive.md)` MoE is the direct route to billion-parameter TS FMs (Time-MoE). Inside `[Masked encoder](masked-encoder.md)` MoE replaces hand-engineered frequency projections with learned routing (Moirai-MoE). Compared to `[Lightweight non-transformer](lightweight-non-transformer.md)`, MoE is the opposite philosophy: pay for capacity, keep compute low via sparsity, rather than shrinking everything.

## Design choices in the literature

- `[Time-MoE](../papers/time-moe.md)` — 2.4B total parameters as a sparse decoder-only MoE trained on the [Time-300B](../datasets-benchmarks/time-300b.md) corpus (~300B points, 9 domains). The *first billion-parameter TS FM*, and the cleanest published evidence that TS obeys LLM-style scaling on both active and total parameter axes. MoE lives in the FFN blocks of a causal patched transformer.
- `[Moirai-MoE](../papers/moirai-moe.md)` — applies sparse routing inside the masked-encoder `[MOIRAI](../papers/moirai.md)` architecture. The key ablation is that dropping the hand-engineered multi-patch-size projections and replacing them with a *single* projection plus MoE routing recovers — and in several datasets exceeds — the specialized model's accuracy. The paper frames this as strong evidence that frequency-based specialization is insufficient and that routed specialization is a strict superset.

## Open questions

- **Routing strategies specific to TS.** Language MoE uses token-level top-k; TS might benefit from patch-level or window-level routing, where siblings of the same time window share an expert. Untested.
- **What do the experts actually specialize on?** Anecdotally frequency or regime, but no clean decomposition exists.
- **Optimal `E` and `k` for TS.** Time-MoE uses specific values; whether smaller or larger configurations Pareto-dominate at matched compute is open.
- **MoE + [in-context learning](../concepts/in-context-learning.md).** Routing a sibling series's token to a different expert than the target's token could break group attention in subtle ways; this has not been studied.
- **Hardware efficiency.** MoE all-to-all communication is the bottleneck at scale; shard-friendly expert layouts specific to TS workloads are a practical open question.

## Related wiki pages

- [Scaling laws](../concepts/scaling-laws.md)
- [Patch tokenization](../concepts/patch-tokenization.md)
- [Multi-task universal](../concepts/multi-task-universal.md)
- [Decoder-only autoregressive](decoder-only-autoregressive.md)
- [Masked encoder](masked-encoder.md)
- [Lightweight non-transformer](lightweight-non-transformer.md)
