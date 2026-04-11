# Time-Series Forecasting

Time-series forecasting is the task of predicting future values of one or
more numerical sequences given their past observations and, optionally,
exogenous covariates (calendar features, prices, weather, text, control
signals). Formally, given a history
`y_{1:T}` and covariates `x_{1:T+H}` the forecaster produces an estimate
of `y_{T+1:T+H}` over a horizon of length `H`. This simple framing hides
a wide design space, and most of the disagreement between methods in the
literature can be traced back to which axis of that space they privilege.

## Axes of variation

- **Univariate vs. multivariate.** A single series (one store's daily
  sales) or many correlated series (every SKU in a warehouse, every node
  in a power grid). Multivariate problems may be treated
  *channel-independently* — one model applied per series, as PatchTST
  popularized — or with explicit cross-channel attention, as [MOIRAI](../papers/moirai.md) and
  [Chronos-2](../papers/chronos-2.md) do.
- **Point vs. probabilistic.** A point forecast returns a single number
  per future step; a probabilistic forecast returns a distribution or a
  sample path. Modern TS-FMs are predominantly probabilistic, either by
  emitting distributional parameters, quantiles, or autoregressive
  samples over a discretized vocabulary.
- **Short vs. long horizon.** Short-horizon tasks (one step, a handful of
  steps) are dominated by local structure and are easy targets for
  classical methods. Long-horizon tasks (96, 336, 720, 1000+ steps on
  the standard ETTh/ETTm/Weather/Traffic suites) punish error
  accumulation and reward architectures that can attend to long
  contexts.
- **Regular vs. irregular sampling.** Most benchmarks assume a fixed
  sampling rate; many real problems (clinical vitals, financial ticks)
  do not, motivating continuous-time models.
- **Stationary vs. non-stationary.** Trends, seasonalities, regime shifts
  and distribution drift all break the i.i.d. assumption and are a major
  reason techniques like RevIN (see
  [../concepts/revin-normalization.md](../concepts/revin-normalization.md))
  matter.

## Loss functions

Classical papers report **MSE** and **MAE** for point forecasts and
**MAPE**, **sMAPE**, or **[MASE](../evaluation/metrics.md#17-mase--mean-absolute-scaled-error)** when series live on very different
scales. Probabilistic evaluation uses the **Continuous Ranked
Probability Score ([CRPS](../evaluation/metrics.md#21-crps--continuous-ranked-probability-score))** or its quantile-loss approximation, and —
for discrete-token models such as [Chronos](../papers/chronos.md) — plain **negative
log-likelihood** on the predicted distribution. Benchmarks like [GIFT-Eval](../datasets-benchmarks/gift-eval.md)
and [TimeBench](../datasets-benchmarks/timebench.md) aggregate several of these across dozens of datasets so
that model comparisons are not dominated by any single series.

## Canonical applications

- Retail and supply-chain demand (Walmart, M-competitions).
- Energy: electricity load, wind and solar generation, building HVAC.
- Finance: volatility, price, risk-factor forecasting.
- Healthcare: vitals in the ICU, disease-incidence nowcasting.
- Infrastructure monitoring: traffic flow, network telemetry, anomaly
  detection.

Each application stresses a different subset of the axes above, which is
exactly why a *foundation* model — trained once, then reused
zero-shot — is such an attractive target.

## Where to read next

For the pre-2023 landscape, continue to
[classical-methods.md](classical-methods.md) and then
[deep-learning-era.md](deep-learning-era.md). For the TS-FM paradigm
itself, jump to [../foundation-models/foundation-models.md](../foundation-models/foundation-models.md).

## Related wiki pages

- [classical-methods.md](classical-methods.md)
- [deep-learning-era.md](deep-learning-era.md)
- [../foundation-models/foundation-models.md](../foundation-models/foundation-models.md)
