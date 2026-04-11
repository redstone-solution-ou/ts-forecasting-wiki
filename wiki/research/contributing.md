# Contributing

This wiki is meant to be a living research base. There are two ways to
contribute: add to the wiki itself, or use the wiki as a launchpad for
research you then push back into the field. Both are welcome; this
page covers the mechanics of the first and the etiquette of the
second.

## Contributing to the wiki

The wiki is a directed graph of short pages. Adding a new paper is the
most common contribution and involves four mechanical steps.

1. **Add a leaf in `papers/`.** Create `wiki/papers/<slug>.md` using
   the existing files as templates. One-paragraph summary, key
   architectural choices, pretraining details, claimed results, and a
   "Related wiki pages" footer with at least two cross-links.
2. **Slot the paper into the taxonomy.** Update
   [../foundation-models/taxonomy.md](../foundation-models/taxonomy.md)
   so the new paper appears in the right cluster, and update the
   cluster's page under `architectures/` if the paper introduces a
   new mechanism.
3. **Update the cross-cutting concept pages.** If the paper uses or
   advances a concept already described in `concepts/` (patch
   tokenization, [value quantization](../concepts/value-quantization.md), [in-context learning](../concepts/in-context-learning.md), scaling laws,
   probabilistic forecasting, [RevIN](../concepts/revin-normalization.md), synthetic data), add a sentence
   or paragraph there with a link.
4. **Update the research pages.** Add a row to
   [comparison-matrix.md](comparison-matrix.md), a line in
   [reproducibility.md](reproducibility.md), and — if the paper
   closes or opens a frontier question — a sentence in
   [open-problems.md](open-problems.md). Consider whether the
   [reading-roadmap.md](reading-roadmap.md) needs reordering.

The PR should be one commit per concern when possible: "add paper
leaf", "slot into taxonomy", "update research pages". Reviewers can
then land the leaf even if the taxonomy edit needs more discussion.

## Style guide

- **No emojis.** Anywhere.
- **No fabricated numbers.** If a paper does not state a metric, say
  so ("not disclosed", "—"). Do not paraphrase a number from memory.
- **Every non-obvious claim is cited.** At minimum link to the paper
  leaf; for stronger claims link directly to the PDF or the arXiv ID.
- **Relative links only.** The wiki must be browsable offline and on
  GitHub, so always use relative paths like `../papers/chronos.md`.
- **Short pages.** Each page should be readable in two to five
  minutes. Longer material belongs in a dedicated research page, not
  in a leaf.
- **End every page with "Related wiki pages".** At least two outbound
  links, so the graph stays navigable.
- **Prose, not bullet salad.** Use bullets where they genuinely help;
  write paragraphs where a comparison needs a sentence.
- **Name papers by their canonical name and slug** ([TimesFM](../papers/timesfm.md), [Chronos](../papers/chronos.md),
  [MOIRAI](../papers/moirai.md), [Time-MoE](../papers/time-moe.md)) so the cross-links are predictable.

## Contributing to the field

If you want to do research rather than curation, start from
[open-problems.md](open-problems.md) — it is explicitly the "topics
waiting for a contributor" page. A few pieces of practical advice that
are harder to find elsewhere:

- **Talk to the authors.** The TS-FM community is small enough that
  every major paper has responsive authors. An email that says "I am
  trying to reproduce figure 4 and am getting X, does that match
  anything you saw internally?" will usually get an answer. That alone
  is worth more than a month of guessing.
- **Upload weights.** If you train something, publish the checkpoint
  on HuggingFace with a model card. The field has a strong norm of
  open weights (see [reproducibility.md](reproducibility.md)) and
  closed releases are read with suspicion.
- **Benchmark on GIFT-Eval, not the legacy long-horizon suite.** The
  ETT / Weather / Traffic / ILI datasets have been overused; GIFT-Eval
  is the cleanest protocol right now. See
  [../datasets-benchmarks/gift-eval.md](../datasets-benchmarks/gift-eval.md).
- **Report probabilistic metrics.** [CRPS](../evaluation/metrics.md#21-crps--continuous-ranked-probability-score) and [WQL](../evaluation/metrics.md#23-wql--weighted-quantile-loss) are the community
  norm; a point-only report is increasingly seen as incomplete.
- **Cite ablations, not only headline numbers.** An ablation table
  that isolates what actually mattered in your method is more
  persuasive than a bold row on a leaderboard.
- **Document training cost.** Most TS-FM papers under-report this.
  Publishing GPU-hours and a rough dollar estimate is a small act that
  makes the whole field more reproducible.

## Related wiki pages

- [open-problems.md](open-problems.md)
- [reproducibility.md](reproducibility.md)
- [../foundation-models/taxonomy.md](../foundation-models/taxonomy.md)
