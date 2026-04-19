# Rebuilding a TimeBench-style Corpus from Public Sources

This page is a practical recipe for assembling a **TimeBench-equivalent
pretraining corpus** using only publicly released data, starting from
the `GIFT-Eval Pretrain` subset of [LOTSA](../datasets-benchmarks/lotsa.md).
[TimeBench](../datasets-benchmarks/timebench.md) itself — the ~1 trillion-point
corpus used by [Sundial](../papers/sundial.md) and [Timer-S1](../papers/timer-s1.md) —
has not been re-released in auditable form, but both its ingredient
list and the [Timer-S1](../papers/timer-s1.md) curation pipeline are
documented in enough detail that an independent rebuild is possible.

The reader has a target GIFT-Eval evaluation, already owns `GIFT-Eval Pretrain`,
and wants the missing pieces plus the filters to apply before training.

## 1. TL;DR — the finished mix

| Layer | Source | Approx. size (obs) | Leakage status | Action |
|---|---|---|---|---|
| Core (have it) | `GIFT-Eval Pretrain` (LOTSA subset) | 230B | clean | use as-is |
| Real add-on | [Time-300B](../datasets-benchmarks/time-300b.md) | ~300B | unaudited | dedupe + GIFT-Eval scrub |
| Real add-on | Chronos public corpus (28 datasets) | tens of B | high (Monash overlap) | dedupe + aggressive scrub |
| Real add-on (optional) | [Time Series Pile](../datasets-benchmarks/time-series-pile.md) | smaller, multi-task | high (Monash) | skip for pure forecasting |
| Synthetic | KernelSynth (GP) | scale to taste | zero by construction | generate |
| Synthetic | TSMix (convex mixup) | scale to taste | inherits from inputs | generate after scrubbing inputs |
| Synthetic | Canonical signal families | scale to taste | zero by construction | generate |

Expected real-data total after scrub/dedup: ~450–550B observations.
Remaining gap to reach the 1T TimeBench headline is filled with
synthetic generation.

## 2. Starting point — `GIFT-Eval Pretrain`

`GIFT-Eval Pretrain` is the LOTSA-derived, leakage-filtered subset used
by [Moirai 2.0](../papers/moirai-2.md). It contains 230B observations
and is already scrubbed against the [GIFT-Eval](../datasets-benchmarks/gift-eval.md)
test splits (see [lotsa.md § Related / derived corpora](../datasets-benchmarks/lotsa.md)).
Distributed via Salesforce's `uni2ts` tooling alongside `Salesforce/lotsa_data`.

Because it is pre-scrubbed, **no additional GIFT-Eval filtering is
required on this layer** — but every subsequent addition must pass
through the full pipeline in section 6.

## 3. Layer 1 — Time-300B

[Time-300B](../datasets-benchmarks/time-300b.md) is [Time-MoE](../papers/time-moe.md)'s
corpus, ~300B observations across 9 domains, released alongside
Time-MoE's weights. It is the single biggest public addition.

Caveats:

- **No published GIFT-Eval leakage audit.** Time-MoE was assembled
  before GIFT-Eval and is not part of the Timer-S1 scrub
  documentation. You must apply the scrub yourself.
- **Skewed domain mix.** Per [training-a-small-model.md § 3](training-a-small-model.md),
  "Nature" is ~90% of observations, so the effective diversity is
  lower than the 300B headline.
- **Overlap with LOTSA is likely.** Both corpora ingest Electricity,
  Traffic, Weather and parts of Monash. A dataset-level dedup pass
  against `GIFT-Eval Pretrain` is mandatory before counting
  observations.

Expected clean yield after dedup + scrub: **on the order of 200–280B
observations**, depending on how aggressive the fuzzy dedup is.

## 4. Layer 2 — Chronos public corpus snapshot

[Chronos](../papers/chronos.md)'s pretraining corpus is 28 public
source datasets, redistributed via the HuggingFace dataset card at
`amazon-science/chronos-forecasting` (see [papers/chronos.md § Reproducibility](../papers/chronos.md)).
[TimeBench](../datasets-benchmarks/timebench.md) explicitly reuses
"public dataset snapshots from Chronos"
(see [datasets-benchmarks/timebench.md § Overview](../datasets-benchmarks/timebench.md)).

This is the **highest-leakage ingredient** in the rebuild:
[Moirai-MoE](../papers/moirai-moe.md) Figure 3 asterisks Chronos on
Monash for precisely this reason
([evaluation/protocols.md § 5](../evaluation/protocols.md)), and the
Chronos v1 paper itself acknowledges that Benchmark II datasets
overlap non-trivially with training via [TSMix](../concepts/synthetic-data-augmentation.md).

Expected clean yield after dedup + scrub: **single-digit billions to
low tens of billions of observations**. Much of the raw content is
Monash-adjacent and disappears in the GIFT-Eval leakage pass.

## 5. Layer 3 — Synthetic generators

Synthetic generation is the only knob that produces arbitrarily many
clean points. Three named recipes plus an optional fourth are
documented in [concepts/synthetic-data-augmentation.md](../concepts/synthetic-data-augmentation.md):

- **KernelSynth (Chronos).** Gaussian-process samples with random
  kernel compositions. Cheap, license-free, **zero leakage by
  construction**. Scale to whatever point count you need. Original
  Chronos used 1M series / ~1.02B points (see [lotsa.md § Moirai-2 derivative mixture](../datasets-benchmarks/lotsa.md)).
- **TSMix (Chronos).** Convex Dirichlet-weighted mixtures of real
  series. Inherits leakage from its inputs: only apply **after** the
  GIFT-Eval scrub of layers 1–2. Chronos used 10M mixed series as a
  default.
- **Canonical signal families.** The primitives named in the Sundial
  / Timer-S1 TimeBench description: linear, sinusoidal, exponential,
  power, impulse, step (see [datasets-benchmarks/timebench.md § Overview](../datasets-benchmarks/timebench.md)).
  Exact parameterization is not published — you match the spirit, not
  the bits.
- **Optional: ForecastPFN priors.** Used by [Mamba4Cast](../papers/mamba4cast.md)
  (70% GP, 30% ForecastPFN). Adds a second synthetic prior family if
  you want more diversity than KernelSynth alone.

Use synthetic generation to fill the gap between your cleaned real
total and the 1T target.

## 6. The mandatory curation pipeline

Apply the full [Timer-S1](../papers/timer-s1.md) pipeline to **every
real-data layer you add** (layers 3 and 4; not needed on
`GIFT-Eval Pretrain` which is pre-scrubbed). Source:
[training-a-small-model.md § The Timer-S1 curation pipeline as a recipe](training-a-small-model.md)
and [papers/timer-s1.md § Key contributions](../papers/timer-s1.md).

1. **Causal mean imputation** for missing values. No future-into-past
   leakage through the imputer.
2. **Outlier removal** via k-σ or IQR on shifting windows.
3. **ADF predictability filter.** Series that fail the Augmented
   Dickey-Fuller test are discarded or down-weighted. The argument is
   that unpredictable-noise series degrade pretraining signal.
4. **Eval-leakage scrub.** Explicit series-level dedup against the
   GIFT-Eval test splits *and* against Chronos Benchmark II. Do this
   with dataset-name match first, then a fuzzy-hash or nearest-neighbor
   check on series windows to catch renamed redistributions.
5. **Resampling augmentation.** Downsample plus Fourier interpolation
   to expose the model to multiple temporal resolutions from a single
   source.
6. **Value-flipping augmentation.** Multiply by −1 to invert trends
   while preserving dependencies. **Do not apply to strictly
   non-negative series** (counts, prices, utility demand).

Run steps 1–4 on layers 3 and 4 before counting observations. Run
steps 5–6 as a final augmentation pass across the whole corpus.

## 7. Size budget worksheet

Cumulative observations at each stage, assuming typical dedup /
scrub overheads:

| Stage | Cumulative real obs | Synthetic budget to add | Notes |
|---|---|---|---|
| After `GIFT-Eval Pretrain` | 230B | 0 | already clean |
| + Time-300B (post-scrub) | ~450–510B | 0 | dedup vs LOTSA required |
| + Chronos corpus (post-scrub) | ~460–530B | 0 | most is Monash-adjacent → dropped |
| + KernelSynth | same | ~100–300B | generate to any target |
| + TSMix | same | ~100–300B | inputs must be already scrubbed |
| + Canonical signal families | same | ~50–200B | cheap, composable |
| **Target** | — | — | **~1T points** to match TimeBench |

The synthetic layers are where you close the gap. This is also why
TimeBench's per-source weighting is "not published in auditable form"
(see [papers/timer-s1.md § Limitations and open critiques](../papers/timer-s1.md)):
the 1T headline is not necessarily 1T of *real* observations.

## 8. What you cannot faithfully reproduce

Be explicit about the limits in any paper that builds on this recipe:

- **TimeBench's exact source mixture and per-domain weights** are not
  published (see [papers/timer-s1.md § Limitations and open critiques](../papers/timer-s1.md)).
  Your rebuild will match the *composition* of TimeBench but not the
  exact sampling ratios.
- **The CloudOps telemetry** used in the Moirai-2 mixture
  (~2.15M series / ~1.48B daily observations; see
  [lotsa.md § Related / derived corpora](../datasets-benchmarks/lotsa.md))
  is proprietary and unavailable.
- **Timer-S1's exact ADF threshold, k-σ multiplier, IQR bounds, and
  fuzzy-dedup threshold** are not stated numerically in the paper.
  You set them and document them.
- **Any paper published against your rebuild should label results as
  "trained on a TimeBench-style corpus," not "trained on TimeBench,"**
  and ship the full source list, weights, and filter parameters
  alongside the model.

## 9. Why this is worth doing

From [benchmarks/training-a-small-model.md § 6 Rolling your own corpus](training-a-small-model.md):
*"Data curation is the contribution of [Timer-S1](../papers/timer-s1.md)
precisely because it is hard."* A transparent, reproducible rebuild
is itself a contribution — it is the one thing Timer-S1 does not
deliver, and the one thing the field most needs for trustworthy
zero-shot claims on [GIFT-Eval](../datasets-benchmarks/gift-eval.md).

## Related wiki pages

- [wikipedia-pageviews-leakage.md](wikipedia-pageviews-leakage.md) — concrete audit of Wikipedia data in GIFT-Eval (test has none; Pretrain has three datasets), with scrub procedure if you add raw Wikimedia pageviews on top.
- [datasets-benchmarks/timebench.md](../datasets-benchmarks/timebench.md) — the target corpus description.
- [datasets-benchmarks/lotsa.md](../datasets-benchmarks/lotsa.md) — base corpus and `GIFT-Eval Pretrain` provenance.
- [datasets-benchmarks/time-300b.md](../datasets-benchmarks/time-300b.md) — biggest public add-on.
- [datasets-benchmarks/time-series-pile.md](../datasets-benchmarks/time-series-pile.md) — optional multi-task layer.
- [datasets-benchmarks/gift-eval.md](../datasets-benchmarks/gift-eval.md) — the evaluation benchmark everything is scrubbed against.
- [concepts/synthetic-data-augmentation.md](../concepts/synthetic-data-augmentation.md) — KernelSynth, TSMix, ForecastPFN recipes.
- [papers/timer-s1.md](../papers/timer-s1.md) — the curation pipeline's canonical source.
- [papers/sundial.md](../papers/sundial.md) — the paper that introduced the TimeBench name.
- [benchmarks/training-a-small-model.md](training-a-small-model.md) — companion guide on architecture, head, and compute choices.
- [evaluation/protocols.md](../evaluation/protocols.md) — known leakage cases and leakage-free cross-paper benchmarks.
- [evaluation/comparability-checklist.md](../evaluation/comparability-checklist.md) — the seven-item checklist a reviewer will apply to your results.
