# Flow-Matching Continuous

## Intuition

Flow-matching continuous models learn a time-dependent velocity field that transports a simple base distribution (usually standard Gaussian noise) onto the data distribution. Sampling is an ODE integration from noise to data — a continuous-time cousin of diffusion. For time-series forecasting, this gives you probabilistic, native continuous-valued predictions without ever discretizing the value axis into a vocabulary and without committing to a specific parametric distributional family (Gaussian, Student-t, mixture). It is the most recent probabilistic output family to enter TS foundation models, and it sits squarely opposite value quantization in the design space.

## Mechanics

The flow-matching objective trains a neural network `v_theta(y, t, context)` to predict the *velocity* that pushes a sample at interpolation time `t` toward the data manifold. Given a ground-truth target `y_1` and noise `y_0 ~ N(0, I)`, the standard linear interpolation path is:

```
y_t = (1 - t) * y_0 + t * y_1            # t in [0, 1]
v_target = y_1 - y_0                      # constant-velocity path
```

Training is a regression against the true velocity:

```
for step in 1..N:
    y_1, context = sample_data(history, target)
    y_0          = randn_like(y_1)
    t            = uniform(0, 1)
    y_t          = (1 - t) * y_0 + t * y_1
    v_pred       = v_theta(y_t, t, context)
    loss         = || v_pred - (y_1 - y_0) ||^2
```

At inference, sample `y_0 ~ N(0, I)` and integrate the learned ODE from `t=0` to `t=1`:

```
y = randn_like(target_shape)
for step in 1..K:                         # K Euler (or Heun) steps
    t   = step / K
    dt  = 1 / K
    y  += v_theta(y, t, context) * dt
y_sample = y                              # one trajectory from the predictive distribution
```

For probabilistic forecasting, draw `M` independent noise seeds, integrate `M` trajectories, and form the empirical distribution from the resulting cloud:

```
samples = [integrate(noise_i, context) for noise_i in randn(M, horizon)]
quantiles = empirical_quantiles(samples, q=[0.1, 0.5, 0.9, ...])
```

The `context` here is the TS foundation model's encoding of the history — typically produced by a standard patched transformer or SSM backbone — and the velocity network `v_theta` is a conditional head on top of that encoding.

## Why it works

Flow matching inherits two deep advantages from continuous normalizing flows. First, **no distributional commitment**: the learned transport can represent any smooth distribution, so multimodal, skewed, and heavy-tailed predictive distributions are handled automatically without picking a family. This sidesteps the single-Gaussian-vs-mixture-vs-Student-t debate that dominates parametric-head design. Second, **no precision ceiling**: outputs live on the continuous real line, not on a discrete bin grid, so there is no equivalent of Chronos's vocabulary cutoff.

The intellectual difference from diffusion is that flow matching regresses against a *known* velocity (derived from the chosen interpolation path) rather than a score function that must be estimated via stochastic differential equations. This makes the training objective a clean supervised regression — stable, easy to tune, no noise-schedule finagling — while preserving the expressiveness of score-based generative modeling.

From an information-theoretic angle, the velocity field is a *transport map* from noise to data, and the training loss is the squared error along that path. Because the path is fixed (linear interpolation), the task reduces to learning a vector field on a bounded time interval — an ordinary regression problem whose landscape is much easier to optimize than GAN or VAE objectives. This is part of why flow-matching models trained at scale, like Sundial, can outperform billion-parameter sparse MoE alternatives with fewer parameters.

## Trade-offs and failure modes

The main practical cost is **inference compute**: generating a single trajectory requires `K` forward passes through `v_theta` (typical `K` is 10–100), and generating probabilistic output requires `M` trajectories. Total cost is `K * M` forward passes per forecast, which can be 100–1000× more than a deterministic single-pass decoder. Various tricks (distillation, larger `dt`, learned schedulers) mitigate this but add complexity.

Flow matching is also sensitive to the choice of path. Linear interpolation is the common default but not always optimal; a poorly chosen path can make the velocity field sharp near `t=0` or `t=1`, which hurts both training stability and sampling accuracy. Calibration across the `t` axis is another subtle issue — if training is dominated by easy values of `t`, the learned field may underfit the hard regions.

Finally, debugging flow-matching models is harder than debugging point forecasters. A point forecaster that is off by 10% has an obvious diagnosis; a flow-matching model whose quantile intervals are miscalibrated could have a problem in the velocity network, the path, the integration scheme, or the number of samples, and isolating them requires dedicated tooling.

## Siblings and design space

Compared to `[Value quantization](../concepts/value-quantization.md)` + categorical sampling (Chronos), flow matching removes the precision ceiling and the vocabulary, at the cost of an ODE integration at inference. Compared to parametric heads on `[Masked encoder](masked-encoder.md)` (MOIRAI's mixture of Student-t), flow matching removes the distributional commitment at the cost of non-closed-form density. Compared to `[Decoder-only autoregressive](decoder-only-autoregressive.md)` rollout, flow matching emits the full horizon in one integration, avoiding autoregressive compounding error — but paying for `K` integration steps instead.

## Design choices in the literature

- `[Sundial](../papers/sundial.md)` — the reference TS foundation model in this family. Uses a **TimeFlow** flow-matching objective so the model outputs native continuous values, trains on **TimeBench** (~1T points) combining real and synthetic components, and delivers probabilistic multi-horizon forecasts by drawing multiple noise seeds and integrating the learned flow. The paper reports outperforming Time-MoE on several benchmarks despite using fewer parameters, attributed both to the more expressive continuous objective and to the scale and diversity of TimeBench. Tsinghua, 2502.00816.

## Open questions

- **Inference-compute efficiency.** Can distillation or flow-straightening reduce the `K * M` cost to match a single-pass decoder without sacrificing calibration?
- **Path design.** Linear interpolation is not obviously optimal for TS; alternatives like variance-preserving paths are untested.
- **Conditional architecture.** What is the right backbone to produce the `context` vector that feeds `v_theta` — transformer, SSM, MLP-Mixer? Sundial picks one choice; the design space is large.
- **Multimodal horizons.** Does flow matching actually capture multimodal next-step distributions better than a mixture head in practice? Clean ablations are missing.
- **Scaling laws for flow matching.** Sundial is one data point; whether the same Chinchilla-style curves hold for flow-matching TS FMs is an open question.
- **Long horizons.** How does single-shot flow integration compare to autoregressive rollout at horizons of 1000+ steps?

## Related wiki pages

- [Probabilistic forecasting](../concepts/probabilistic-forecasting.md)
- [Value quantization](../concepts/value-quantization.md)
- [Scaling laws](../concepts/scaling-laws.md)
- [Synthetic data augmentation](../concepts/synthetic-data-augmentation.md)
- [Decoder-only autoregressive](decoder-only-autoregressive.md)
- [Masked encoder](masked-encoder.md)
- [Lightweight non-transformer](lightweight-non-transformer.md)
