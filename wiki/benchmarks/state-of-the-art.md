# State of the Art — Which Model Wins Where

This page interprets the numbers in
[leaderboard.md](leaderboard.md). It partitions the TS-FM space by
forecasting *regime* — the shape of the task a user actually faces —
because no single model is SOTA everywhere. Every claim here points
back to a specific cell on the leaderboard page.

## Zero-shot probabilistic forecasting (univariate, no covariates)

As of late 2025, the narrow "zero-shot univariate probabilistic"
leaderboard is tightly packed at the top. On GIFT-Eval, four models —
Chronos-2 (51.4), TimesFM-2.5 (51.0), TiRex (50.2) and Toto-1.0
(48.6) — are within 3 skill-score points of each other under WQL
(Chronos-2, Table 4). On the same suite, Chronos-2 claims the top
spot but the pairwise confidence intervals it publishes for fev-bench
(Chronos-2, Figure 2b) show its win over TiRex and TimesFM-2.5 is
statistically significant (2–3 skill-score points, non-overlapping
95% CIs), whereas the gap between TiRex and TimesFM-2.5 is not.

The practical takeaway for *just* zero-shot univariate CRPS/WQL is
that Chronos-2, TimesFM-2.5 and TiRex are effectively a three-way
tie, with Chronos-2 slightly ahead on aggregate skill score and
clearly ahead on short-context tasks (Chronos Benchmark II WQL
skill score 46.6 vs 42.4/41.7, Chronos-2 Table 5).

## Zero-shot forecasting with covariates

This is the single regime with a clear separation. On the
fev-bench covariates subset (42 tasks), Chronos-2 reaches 47.0 SQL
skill score; TabPFN-TS, the only other model with native
known-covariate support, reaches 40.0; every other pretrained
model — including TiRex, TimesFM-2.5, Moirai-2.0, Toto-1.0, Sundial —
ignores the covariates and sits in a tight band between 28.0 and
39.9 (Chronos-2, Figure 4b). On domain case studies with real
dynamic covariates (European energy prices, Rossmann retail sales),
the gap widens further: 51.3 vs 46.5 SQL on the energy covariates
subset and 48.6 vs 44.3 WQL on retail (Chronos-2, Figure 5). If the
production task has covariates, Chronos-2 is the only current public
TS-FM where that information is actually used at inference time.

## Fine-tuned / long-horizon point forecasting (LTSF)

The classic "long-sequence forecasting" benchmarks (ETT, Weather,
ECL, Traffic, averaged over horizons {96, 192, 336, 720}) reward
different models than the zero-shot probabilistic suites. On ETTh1
and ETTh2 zero-shot MSE, Sundial-Large posts 0.395 / 0.334 and
Time-MoE-Large 0.394 / 0.405 (Sundial Table 1; Time-MoE Table 3),
which are essentially at parity with the best full-shot baselines
of the 2023 era (PatchTST ~0.41, iTransformer ~0.45 per Time-MoE
Table 4). TTM-A (5M parameters) matches them at 0.400 / 0.333 (TTM
Table 1) despite being 80-500x smaller. MOIRAI-Large, by contrast,
is noticeably worse on ETTh1 (0.510 MSE, MOIRAI Table 6) — a
consistent weak point for that model family.

On this regime, the differentiator is not zero-shot prowess but
whether the model's training mix included ETT-like data. See
[methodology-caveats.md](methodology-caveats.md) — LTSF results for
TimesFM and Chronos need an asterisk because their pretraining
corpora contain datasets overlapping with the ETT / Weather / ECL
families.

## Multivariate forecasting

On the fev-bench multivariate subset (26 tasks), Chronos-2 again
leads (57.9 SQL skill score) but the margin is small: Toto-1.0
57.6, TimesFM-2.5 57.1, TiRex 56.1 (Chronos-2 Figure 4a). The
Chronos-2 paper explicitly notes that "univariate channel-independent
models often perform on par with multivariate ones" on this subset
(Chronos-2, Section 5.2), echoing Nie et al.'s channel-independence
finding for PatchTST. Unless a dataset has short history and strong
instantaneous cross-series dependence, native multivariate support
(Chronos-2, Toto-1.0) is a small effect — much smaller than native
covariate support.

## Short-context / short-history tasks

Chronos Benchmark II is the best proxy because most of its 27 tasks
have fewer than 300 observations. Here the top-4 ordering is
Chronos-2 (46.6), TimesFM-2.5 (42.4), Toto-1.0 (41.9), TiRex (41.7)
under WQL skill score (Chronos-2, Table 5). Chronos-2 opens a ~4
skill-score-point lead here — considerably larger than on GIFT-Eval —
which the authors attribute to its in-context learning (ICL)
mechanism: with little history per series, the model leans on cross-
series information from the batch (Chronos-2, Section 5.2).

## Efficiency-constrained / CPU / on-device

TTM dominates this axis. At 1M (TTM-Base) to 5M (TTM-Advanced)
parameters it reports ZS MSE comparable or better than Moirai,
TimesFM, and Chronos on the ETT/Weather/ECL suite while running in
0.01s/batch on CPU versus ~2,350s/batch for Chronos-Base
(TTM, Tables 1 and 3 — a 240,000x CPU speedup). Time-MoE-Base is
the second option: 50M activated parameters, explicitly designed
for CPU inference (Time-MoE, Section 3.2.3). The Chronos-2 small
variant (28M) trails the base 120M model by only ~1 skill-score point
on GIFT-Eval with ~2x faster inference (Chronos-2, Section 5.4), and
is probably the most accurate sub-50M choice.

## Current SOTA recommendations by use case

These are opinions informed by the leaderboard, not paper-independent
ground truth. Always validate on your own holdout.

- **Zero-shot univariate point or probabilistic forecasting, arbitrary
  series, accuracy first.** Chronos-2 (120M) for the best
  aggregate numbers across all three major suites (fev-bench 47.3,
  GIFT-Eval WQL 51.4, Chronos-II WQL 46.6; Chronos-2 Tables 3–5).
  TimesFM-2.5 is an almost-identical runner-up within 1-5 skill-score
  points on every suite.
- **Zero-shot with covariates (known exogenous features).** Chronos-2
  is effectively unrivalled. TabPFN-TS is the only other pretrained
  model with real covariate support; all other TS-FMs silently drop
  them.
- **Long-horizon LTSF suite, fine-tuning allowed.** Sundial-Large or
  Time-MoE-Large for best aggregate ETT/ECL/Weather numbers
  (Sundial Table 1). If compute is constrained, TTM-A at 5M params
  matches them (TTM Table 1).
- **Short-history / few observations per series.** Chronos-2 by the
  largest margin on Chronos Benchmark II, with TimesFM-2.5 second.
- **CPU deployment / edge / extreme latency budget.** TTM (1–5M) by
  240,000x on CPU vs Chronos-Base (TTM Table 3). Time-MoE-Base is the
  mid-point at 50M activated parameters.
- **Long-context (thousands of timesteps).** Time-MoE (trained at
  4096 tokens, Time-MoE Section 3.2.3), Timer-XL, or Chronos-2-2K /
  Chronos-2 with the 8192-step post-training (Chronos-2 Section 5.4).

The "no single winner" part of the story is what makes
[methodology-caveats.md](methodology-caveats.md) necessary: whichever
paper you read last will, by construction, show its own model on top.

## Related wiki pages

- [leaderboard.md](leaderboard.md)
- [methodology-caveats.md](methodology-caveats.md)
- [efficiency-and-cost.md](efficiency-and-cost.md)
- [../foundation-models/taxonomy.md](../foundation-models/taxonomy.md)
