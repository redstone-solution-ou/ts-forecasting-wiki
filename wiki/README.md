# Knowledge Wiki — Time-Series Forecasting Foundation Models

This wiki is a hand-curated knowledge graph about time-series forecasting,
with a strong focus on the recent wave of *foundation models* (TS-FMs)
that learn a single pretrained model transferable across many datasets,
frequencies, and tasks. It is organized general-to-specific: high-level
orientation pages sit at the top, architecture families and cross-cutting
concepts in the middle, and individual research papers form the leaves.
Each page is short, links forward and backward, and ends with a list of
related nodes so that readers can move through the graph rather than
reading it linearly.

The wiki pairs with the `papers/` directory at the repo root, which holds
the actual PDFs. A typical traversal starts here, narrows down through
`foundation-models/taxonomy.md` to a cluster, clicks into an architecture
or concept page for context, and finally opens the corresponding leaf in
`papers/` to read the paper itself. Cross-links are always relative paths,
so the wiki is fully browsable on GitHub or offline.

## Reading paths

- **15-minute overview.** Start with
  [foundations/time-series-forecasting.md](foundations/time-series-forecasting.md),
  then read [foundation-models/README.md](foundation-models/README.md), then
  skim [foundation-models/taxonomy.md](foundation-models/taxonomy.md).
- **Compare architectures.** Read
  [foundation-models/taxonomy.md](foundation-models/taxonomy.md) for the
  cluster map, then walk through
  [architectures/README.md](architectures/README.md) and the individual
  architecture pages it links to.
- **Pick a model for a use case.** Start from
  [foundation-models/taxonomy.md](foundation-models/taxonomy.md) to find
  the right cluster, consult the cross-cutting pages under
  [concepts/README.md](concepts/README.md) (zero-shot, probabilistic,
  patch tokenization), then open the relevant leaves in
  [papers/README.md](papers/README.md).

## Top-level sections

- [foundations/](foundations/README.md) — the classical and deep-learning
  precursors that set the stage for TS-FMs.
- [foundation-models/](foundation-models/README.md) — the TS-FM paradigm,
  its brief history, and the seven-cluster taxonomy.
- [architectures/](architectures/README.md) — architecture families:
  decoder-only autoregressive, masked encoder, encoder-decoder, MoE,
  LLM reprogramming, lightweight non-transformer, continuous / flow
  matching.
- [concepts/](concepts/README.md) — cross-cutting technical ideas: patch
  tokenization, value quantization, zero-shot forecasting, probabilistic
  forecasting, in-context learning, scaling laws, RevIN, multi-task
  universal models, synthetic data.
- [datasets-benchmarks/](datasets-benchmarks/README.md) — the corpora and
  evaluation suites that TS-FMs are trained and judged on (Monash, LOTSA,
  Time-Series Pile, Time-300B, TimeBench, GIFT-Eval).
- [papers/](papers/README.md) — the 19 leaf pages, one per paper.

## Knowledge-graph sketch

```
ts-forecasting-wiki
|
+-- foundations/
|   +-- time-series-forecasting
|   +-- classical-methods
|   +-- deep-learning-era
|
+-- foundation-models/
|   +-- taxonomy
|       +-- Cluster 1 decoder-only AR ---> TimesFM, Timer, Timer-XL,
|       |                                  Lag-Llama, TimeGPT
|       +-- Cluster 2 masked / enc-dec --> Chronos, Chronos-2, MOMENT, MOIRAI
|       +-- Cluster 3 mixture-of-experts-> Time-MoE, Moirai-MoE
|       +-- Cluster 4 LLM reprogramming -> Time-LLM, GPT4TS, LLMTime
|       +-- Cluster 5 lightweight / SSM -> TTM, Mamba4Cast
|       +-- Cluster 6 multi-task unified-> UniTS, TOTEM (+ MOMENT, Timer-XL,
|       |                                  Chronos-2 secondary)
|       +-- Cluster 7 continuous / flow -> Sundial
|
+-- architectures/   (one page per family above)
+-- concepts/        (patching, VQ, zero-shot, CRPS, ICL, scaling, RevIN, ...)
+-- datasets-benchmarks/ (Monash, LOTSA, TS-Pile, Time-300B, TimeBench, GIFT)
+-- papers/          (19 leaves)
```

## Related wiki pages

- [foundation-models/taxonomy.md](foundation-models/taxonomy.md)
- [papers/README.md](papers/README.md)
