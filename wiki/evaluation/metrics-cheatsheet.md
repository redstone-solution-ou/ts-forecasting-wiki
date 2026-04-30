# Metrics Cheatsheet — what every leaderboard column actually means

This is the **read-this-first** companion to [metrics.md](metrics.md).
The flagship page has the rigorous formulas, citations, and edge-case
catalog. This page covers the same metrics in plain English, with one
worked numerical example, and is organized so that the vocabulary you
see on a leaderboard cell — `GM-MASE`, `WQL skill score`, `GM-CRPS`,
`fev-bench SQL` — decomposes cleanly into the four operations under
the hood.

If a number in your evaluation pipeline does not match a published
reference, the diagnosis is almost always a mismatch in one of those
four operations. This page lays them out so the mismatch is easy to
spot.

## 1. The four operations under every leaderboard cell

Every TS-FM leaderboard cell is one combination of these four steps,
applied in order:

1. **A per-point error** — how wrong the forecast is at one timestep.
   `|y_t − ŷ_t|`, `(y_t − ŷ_t)²`, the pinball loss at quantile `τ`,
   or the squared distance between predicted CDF and ideal step.
2. **A per-series score** — average those errors across the horizon
   for one (dataset, configuration) pair. `MAE`, `MAPE`, raw `WQL`,
   raw `CRPS`. Native units of `y` for MAE/CRPS, dimensionless % for
   MAPE.
3. **A per-series normalization vs Seasonal Naive** — divide the
   per-series score by Seasonal Naive's score on the same series.
   This is what makes the number scale-free and comparable across
   datasets. `MASE` already bakes this in (the denominator is
   Seasonal Naive's in-sample MAE); for `MAPE`, `CRPS`, `WQL` you do
   it manually as `MAPE_model / MAPE_SeasonalNaive` per series.
4. **A cross-dataset aggregate** — geometric mean across the `D`
   datasets/configs: `G = (Π_d r_d)^{1/D}`. This is the leaderboard
   number, often called `GM-MASE`, `GM-MAPE`, `GM-CRPS`, `GM-WQL`.
5. **(Optional) Skill-score wrapper** — `S = 100 × (1 − G)`. Same
   number, sign-flipped, expressed as a percentage. `30 skill score
   on GIFT-Eval MASE` ⇔ `GM-MASE = 0.70`.

If a paper reports `0.728 GM-MASE` and another reports `27.2 MASE
skill score` and a third reports `0.728 MASE` (no GM, no skill, just
"MASE"), the first and second are identical, the third is on a
different series-level scale, and only the first two are leaderboard
numbers.

## 2. Point metrics — what each one literally measures

### MAE — Mean Absolute Error

Average of `|y_t − ŷ_t|` over the horizon. Same units as `y`. Useful
for a single-dataset report, useless for ranking models across many
datasets at different scales — a 12-car MAE on Traffic is great, the
same MAE on Exchange Rate is catastrophic. **Step 2 only**, never
appears on a cross-dataset leaderboard directly.

### MAPE — Mean Absolute Percentage Error

`(1/n) Σ |y_t − ŷ_t| / |y_t|`. Dimensionless. Scale-free by
construction — that's the whole point. Two failure modes you must
know about:

- **Blowup at zeros.** When `y_t ≈ 0` the per-point ratio explodes.
  On intermittent-demand series (retail with stockouts, healthcare
  events, count data with rare values) MAPE can be literally
  unbounded. A single zero observation can dominate the score.
- **Asymmetry.** A 50%-low forecast and a 50%-high forecast on the
  same `y` produce different absolute errors but symmetric MAPE.
  MAPE systematically under-penalizes under-forecasting.

Modern TS-FM papers have largely retired MAPE in favor of MASE for
exactly these reasons. The exception is the [GIFT-Eval paper
itself](../papers/aksu-gift-eval.md) which still reports MAPE.

### MASE — Mean Absolute Scaled Error  ⭐

Forecast MAE divided by **Seasonal Naive's in-sample MAE**:

```
MASE = MAE_model / MAE_SeasonalNaive_in-sample
```

Seasonal Naive predicts each point equal to the same point one
season ago: `ŷ_t = y_{t−m}` where `m = 7` for daily, 24 for hourly,
12 for monthly, etc. The in-sample MAE is computed on the *training*
portion of the series; that single scalar becomes the denominator
for every horizon position when scoring the test forecast.

Three properties make this the modern standard:

- **Scale-free.** Both numerator and denominator have the same units.
- **Stable.** The denominator is computed once on training data,
  doesn't depend on test values, doesn't blow up.
- **Semantically grounded.** `MASE = 1` means *exactly as good as
  Seasonal Naive*. `MASE = 0.5` means *twice as good as Seasonal
  Naive*. `MASE > 1` means *you lost to a forecaster that just
  repeated last season*.

Bar to clear on a real leaderboard: `MASE < 1`. Frontier numbers on
GIFT-Eval are in the **0.65–0.80** band (Sundial 0.673, TimesFM
0.680, Timer-S1 0.693). The single biggest pitfall: **wrong `m`
silently destroys the metric.** Always use the seasonal period the
benchmark publishes per dataset; see
[seasonality-and-baselines.md](seasonality-and-baselines.md).

## 3. Probabilistic metrics — what each one literally measures

A probabilistic forecast commits to an entire distribution `F̂_t`,
not just a single number. We score the distribution.

### CRPS — Continuous Ranked Probability Score

Geometric intuition: imagine the predicted CDF `F̂(z)` plotted as a
curve from 0 to 1. The truth is `y`, so the *ideal* CDF is a step
function at `y` (jumps from 0 to 1 at `z = y`). CRPS is the **area
between your predicted CDF and the ideal step**.

Two important properties:

- **Same units as `y`.** So a CRPS of 3.7 on Traffic is commensurate
  with an MAE of 4.2 on Traffic. You can compare a probabilistic
  model against a deterministic model on the same scale.
- **Reduces to MAE for a degenerate distribution.** If your
  "distribution" is a Dirac delta at `ŷ`, CRPS becomes `|y − ŷ|`. So
  you cannot be penalized for being deterministic — you just don't
  earn the calibration credit a sharp probabilistic model can earn.

CRPS is the headline probabilistic metric in MOIRAI, Moirai-MoE,
Sundial, Timer-S1, and the [GIFT-Eval paper](../papers/aksu-gift-eval.md).

### WQL — Weighted Quantile Loss

What you actually compute when your model outputs a finite quantile
grid (typically `τ ∈ {0.1, 0.2, ..., 0.9}`) instead of a continuous
distribution.

Mechanics: average pinball loss across the quantile grid, divided by
total target volume so high-volume series don't dominate cross-series
aggregates. The `gluonts` implementation calls this
`mean_weighted_sum_quantile_loss`:

```
WQL = (1/K) * Σ_τ (Σ_t pinball_τ(y_t, q̂_τ,t)) / Σ_t |y_t|
```

WQL is numerically very close to CRPS on the same series — they're
both proper scores measuring distribution-vs-truth distance. Used as
the headline probabilistic metric on GIFT-Eval and Chronos Benchmark
II.

**Implementation gotcha (this is the one that bites everyone).** The
GIFT-Eval paper's CRPS-from-quantiles approximation in Appendix C
includes a **factor of 2**:

```
CRPS_GIFT-Eval ≈ 2 * gluonts.mean_weighted_sum_quantile_loss
```

So if you compute `mean_weighted_sum_quantile_loss` directly with
gluonts and compare against a GIFT-Eval paper number, your value is
half of theirs. The factor of 2 is the standard
"CRPS-from-quantiles" identity (see [metrics.md §2.3](metrics.md#23-wql--weighted-quantile-loss));
papers vary on whether they include it explicitly.

### SQL — Scaled Quantile Loss

fev-bench's variant of WQL: instead of normalizing by `Σ |y_t|`,
normalize by **Seasonal Naive's WQL** on the same series. So SQL
plays the same role for probabilistic forecasts that MASE plays for
point forecasts: scale-free *and* anchored to a meaningful baseline.

This is why fev-bench SQL skill scores and GIFT-Eval WQL skill scores
are not directly comparable — same underlying loss, different
denominator.

### Pinball loss

The atomic ingredient of WQL/SQL: the strictly-proper score for a
single quantile `τ`. You don't see "pinball loss" directly on a
leaderboard but it shows up as a training objective in Chronos-2
(9 quantiles), Timer-S1 (9 quantiles), Moirai 2.0 (9 quantiles).

## 4. Aggregation: per-series → leaderboard

GM-anything = **geometric mean across `D` datasets/configurations of
(per-dataset metric ÷ same metric for Seasonal Naive)**:

```
GM-CRPS = (Π_d (CRPS_d / CRPS_SeasonalNaive,d))^{1/D}
GM-WQL  = (Π_d (WQL_d  / WQL_SeasonalNaive,d ))^{1/D}
GM-MAPE = (Π_d (MAPE_d / MAPE_SeasonalNaive,d))^{1/D}
GM-MASE = (Π_d MASE_d)^{1/D}
```

Note the asymmetry on the last line: MASE already has Seasonal Naive
in its denominator, so the geometric mean across datasets is enough
— no extra division needed. The other three metrics need explicit
per-dataset Seasonal-Naive normalization before the GM.

### Why geometric mean and not arithmetic?

Two reasons:

- **Ratios should be averaged geometrically.** A model that is 2×
  better on dataset A and 2× worse on dataset B has GM-ratio = 1
  (correct: same on average), but arithmetic mean = 1.25
  (artificially worse).
- **Robust to a single hard dataset.** Geometric mean is bounded
  above by the smallest factor; one catastrophic outlier dataset
  can't blow up the score the way it can with an arithmetic mean.

This is also why **the per-dataset Seasonal-Naive normalization in
step 3 is essential**. Without it, the geometric mean across raw
MAPEs is dominated by datasets with near-zero values; the geometric
mean across raw CRPSs is dominated by high-volume datasets. The
ratio against Seasonal Naive removes both.

## 5. Skill score: a sign-flipped percentage

Once you have GM-MASE / GM-CRPS / GM-WQL / GM-SQL, the skill score
is a presentation wrapper:

```
S = 100 × (1 − G)
```

So `G = 0.70` ⇔ `S = 30`. Interpretation:

- `S > 0` ⇔ better than Seasonal Naive. Good.
- `S = 0` ⇔ ties Seasonal Naive.
- `S < 0` ⇔ worse than Seasonal Naive. Surprisingly common for
  badly-calibrated full-shot transformers.

**The conversion is exact** (`G = 1 − S/100`). The "approximate"
only enters when comparing two papers reporting on different
snapshots of the same benchmark — the leaderboard's dataset list
and config set drift over time. See
[methodology-caveats.md §2](../benchmarks/methodology-caveats.md)
for the convention map.

## 6. Worked numerical example

Series truth (one season, `m = 6`): `y = [10, 12, 14, 13, 11, 9]`.
Suppose Seasonal Naive's MAE on the in-sample portion of this series
is `MAE_SN_in = 2.0`.

**Forecast 1** (sharp point predictor): `ŷ = [9, 12, 14, 14, 11, 10]`.

| Quantity     | Per-point        | Aggregate                           |
| ------------ | ---------------- | ----------------------------------- |
| Errors       | `[1,0,0,1,0,1]`  | MAE = 0.5                           |
| MAPE terms   | `[0.10, 0, 0, 0.077, 0, 0.111]` | MAPE = 4.8%          |
| MASE         | —                | `MAE / MAE_SN_in = 0.5 / 2.0 = 0.25`|

**Forecast 2** (Seasonal Naive itself): `ŷ = [8, 10, 12, 11, 9, 7]`
(shifted by one season from in-sample).

| Quantity | Aggregate |
| -------- | --------- |
| MAE      | 2.0       |
| MASE     | 1.0 (by construction) |

So Forecast 1's MASE is 0.25 — meaning it is 4× better than Seasonal
Naive on this series. If that held across all 97 GIFT-Eval configs,
GM-MASE = 0.25, skill score = 75 — a frontier-leaderboard number.

For probabilistic versions, replace `|y_t − ŷ_t|` with `CRPS(F̂_t,
y_t)` and continue the same operations.

## 7. Diagnosing a unit mismatch in your own pipeline

When a number from your evaluation script doesn't match a published
reference on the same benchmark, walk the four steps:

1. **Per-point error.** Pinball, squared, or absolute? `gluonts`
   defaults differ from `evalml` defaults differ from custom code.
2. **Per-series score.** Did you average across the horizon, sum,
   take the median? Did you apply the factor-of-2 in
   CRPS-from-quantiles?
3. **Per-series normalization.** Did you divide by Seasonal Naive's
   score on the same series? If you skipped this step, your number
   is on a totally different scale than any published leaderboard.
4. **Cross-dataset aggregate.** Geometric mean (correct) or
   arithmetic mean (also seen, but won't match published numbers)?

The single most common bug: **forgetting step 3.** A pipeline that
computes raw CRPS and arithmetic-means across configs gives a number
that has *no relationship* to a published `GM-CRPS` value. Both
numbers can be valid measurements of forecast quality on the same
data — they just aren't the same statistic.

## Related wiki pages
- [metrics.md](metrics.md) — full formulas, edge cases, citations.
- [seasonality-and-baselines.md](seasonality-and-baselines.md) — how
  to pick `m` (the seasonal period that determines the Seasonal-Naive
  denominator).
- [probabilistic-evaluation.md](probabilistic-evaluation.md) — CRPS
  decomposition, calibration vs sharpness.
- [comparability-checklist.md](comparability-checklist.md) — the
  seven-question checklist for "can I compare these two cells?"
- [../benchmarks/methodology-caveats.md](../benchmarks/methodology-caveats.md)
  — skill-score / GM-relative / WQL / SQL conversion identities.
- [../benchmarks/leaderboard.md](../benchmarks/leaderboard.md) — the
  actual leaderboard tables this page teaches you to read.
