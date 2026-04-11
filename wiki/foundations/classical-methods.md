# Classical Methods

Before deep learning, the time-series literature had already produced a
remarkably effective toolbox of statistical models. These methods remain
strong baselines — often humbling — for modern forecasters, and the
design choices of today's foundation models can only be understood
against the backdrop of what they do and do not do well.

## A brief tour

- **Exponential smoothing (ETS).** A family of recursive updates that
  track level, trend, and seasonal components with exponentially
  decaying weights. Holt, Holt-Winters, and the
  error-trend-seasonality (ETS) state-space formulation of Hyndman and
  colleagues are the canonical references. Cheap, interpretable, and
  robust on short or noisy series.
- **ARIMA / SARIMA.** Autoregressive integrated moving average models
  (and their seasonal extension) fit linear dynamics in the differenced
  series. They come with a well-developed identification procedure
  (Box-Jenkins) and closed-form confidence intervals under Gaussian
  noise. SARIMA extends the idea to seasonal lags.
- **Theta method.** A simple decomposition-plus-smoothing approach that
  famously won the M3 competition and remained competitive in later
  contests. Its strength is that it is nearly parameter-free.
- **Prophet.** Facebook's 2017 decomposable model (piecewise-linear
  trend + Fourier seasonality + holiday effects) aimed at business
  analysts. Easy to configure, tolerant of missing data, but limited in
  expressive power.
- **State-space and Kalman filtering.** A general linear-Gaussian
  framework that subsumes ETS and ARIMA. Useful when covariates, latent
  components, or online updating matter; the backbone of much of
  engineering practice.
- **TBATS.** Trigonometric seasonality + Box-Cox + ARMA errors + trend
  + seasonal components, designed to handle multiple and non-integer
  seasonalities (for example intra-day and intra-week together).

## What they do well

Classical methods are interpretable, need little data, and are
well-calibrated on stationary or weakly non-stationary processes. On
the short, clean, single-series problems that dominate many industrial
applications they remain extremely hard to beat — a fact reinforced by
the **M4** and **M5** competitions, where ensembles of classical
methods placed near the top of the leaderboard well into the deep
learning era.

## Where they struggle

- **Multivariate structure.** Most classical methods either ignore
  cross-series information or require bespoke vector extensions (VAR,
  VECM) that do not scale.
- **Long horizons.** Error compounds and seasonalities drift; a method
  tuned for the next step is rarely optimal 720 steps out.
- **Cross-series transfer.** Each series typically needs its own fit.
  There is no equivalent of pretraining, so a million-series catalog
  means a million fits.
- **Cold start.** With no history, ETS and ARIMA have nothing to work
  with; modern TS-FMs promise meaningful zero-shot forecasts from a
  pretrained prior instead.

## The Monash Forecasting Archive as empirical bedrock

The [Monash Forecasting Archive](../datasets-benchmarks/monash-archive.md)
is the single most important collection of classical-method baselines.
It bundles dozens of public datasets with consistent splits and reports
ETS, ARIMA, Theta, TBATS, and Prophet numbers for each, making it the
reference yardstick that later deep-learning and foundation-model papers
cite when they claim to beat "classical baselines."

## Where to read next

Continue to [deep-learning-era.md](deep-learning-era.md) for the neural
forecasting lineage that leads into TS-FMs.

## Related wiki pages

- [../datasets-benchmarks/monash-archive.md](../datasets-benchmarks/monash-archive.md)
- [deep-learning-era.md](deep-learning-era.md)
- [time-series-forecasting.md](time-series-forecasting.md)
