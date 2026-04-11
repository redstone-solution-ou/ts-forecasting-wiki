# Multi-Task Universal Models

## Intuition

A multi-task universal time-series model is a single pretrained checkpoint that handles forecasting, classification, anomaly detection, imputation, and sometimes retrieval — without changing the backbone for each task. The goal is exactly the same as universal LLMs in NLP: amortize pretraining across many objectives so that a practitioner only has to ship, monitor, and update one model across an entire pipeline. It is the last mile between "foundation model that forecasts well" and "foundation model that replaces the toolbox."

## Mechanics

Four recognizable recipes have been tried. They differ in *where* the task-switch lives.

**1. Shared encoder plus task-specific heads.** Pretrain a bidirectional encoder with masked reconstruction, then attach small task heads on top of its representations:

```
z = Encoder(patches)                   # (B, N, D) shared backbone
y_forecast = ForecastHead(z)           # MLP -> (B, horizon)
y_class    = ClsHead(mean_pool(z))     # MLP -> (B, num_classes)
y_anomaly  = ReconHead(z) vs patches   # reconstruction error per step
y_impute   = ReconHead(z)[masked]      # fill masked positions
```

The encoder never specializes; each head is a few-layer MLP that can be swapped or re-initialized per task. `[MOMENT](../papers/moment.md)` is the canonical example, with 40M/125M/385M-parameter T5-initialized encoders and four task heads.

**2. Generative unification.** Cast every task as token generation and train one causal decoder to do all of them. Forecasting is "predict future tokens"; imputation is "fill masked tokens in place"; anomaly detection is "score token likelihood or reconstruction error":

```
# Timer's S3 (Single-Series Sequence) format
stream = flatten_any_series_into_token_sequence(s)
loss   = -log p_theta(stream[1:] | stream[:-1])
```

Because everything is in token space, the training loop is uniform across datasets and tasks.

**3. Task-token interface.** Prepend a learned embedding that names the task, let the transformer condition on it:

```
task_tok = embed(task_id)              # (D,)
input    = [task_tok, patches...]
z        = Transformer(input)
y        = head(z[1:])
```

`[UniTS](../papers/units.md)` takes this route, handling 38 datasets with variable-length inputs through a modified transformer that reads the task token.

**4. Shared discrete-token space.** `[TOTEM](../papers/totem.md)` trains a cross-domain VQ-VAE codebook over patches; a single transformer operates on those tokens regardless of which task is being asked, and the specialist-vs-generalist axis (per-domain vs shared codebook) is explicitly studied.

## Why it works

The intellectual trick is that forecasting, imputation, and anomaly detection are not actually different problems at the representation level — they all require a model that *knows what future values look like given past ones*. Masked reconstruction trains exactly this conditional, and whether you use it for imputation (by construction), for anomaly detection (score residuals), for forecasting (append future mask tokens), or for classification (pool and project) is a matter of which side of the conditional you read out. One encoder carrying a strong conditional can power all four downstream tasks.

Generative unification goes further: it argues that the conditional itself is the only thing worth pretraining, and everything else is a decode-time reinterpretation. This is philosophically clean but bets on the same data distribution being good for every task, which is sometimes false — anomaly detection, for instance, benefits from seeing many "normal" regimes, whereas classification benefits from seeing many labeled *classes*.

The payoff in practice is operational: one weight file, one training pipeline, one inference stack. For small teams this is an enormous simplification, and it is why multi-task FMs tend to beat specialist models once you account for the engineering overhead.

## Trade-offs and failure modes

Shared encoders pay a *capacity tax*: each task gets a fraction of the encoder's representation budget, and a dedicated forecaster can usually beat a universal one at pure forecasting given matched compute. Task-head designs also require labelled data for the non-reconstructive tasks (classification, named-anomaly), which is not always abundant.

Generative unification can be fragile at long horizons (rollout error) and at tasks that are not naturally autoregressive, such as global classification. Task-token interfaces can suffer *task interference*: training on too many heterogeneous tasks hurts any given one unless task weights are carefully balanced. TOTEM's shared VQ-VAE is limited by the codebook size — a single global codebook cannot capture every domain's patch distribution equally well, which is exactly why the paper studies the specialist-vs-generalist split.

A subtle failure mode is *evaluation protocol drift*: multi-task benchmarks (UniTS's 38 datasets, MOMENT's pile) use different splits and metrics than forecasting-only benchmarks ([Monash](../datasets-benchmarks/monash-archive.md), [GIFT-Eval](../datasets-benchmarks/gift-eval.md)), so scores from multi-task and forecasting-only models are not directly comparable.

## Design choices in the literature

- `[MOMENT](../papers/moment.md)` — T5-init encoder with PatchTST patching and [RevIN](./revin-normalization.md), masked reconstruction pretraining, plus forecast/classify/anomaly/impute heads at 40M/125M/385M sizes.
- `[Timer](../papers/timer.md)` — generative unification via the Single-Series Sequence (S3) format, causal decoder handling all tasks as next-token prediction.
- `[Timer-XL](../papers/timer-xl.md)` — universal TimeAttention extends the generative-unification story to multivariate and covariate tasks with a single decoder-only architecture.
- `[UniTS](../papers/units.md)` — task-token interface over a modified transformer, variable-length inputs across 38 datasets, mixing generative and predictive tasks under one model.
- `[TOTEM](../papers/totem.md)` — shared VQ-VAE codebook across domains, with explicit specialist-vs-generalist ablations.
- `[GPT4TS / OFA / FPT](../papers/gpt4ts.md)` — a frozen LLM (GPT-2/BERT/BEiT) with only layer-norm and positional embeddings trained, acting as a universal TS solver across six tasks.
- `[Chronos-2](../papers/chronos-2.md)` — unifies univariate, multivariate, covariate, and panel forecasting in one group-attention architecture (a task-axis unification rather than a task-type unification).

## Open questions

- **Specialist vs universal trade-off.** How much accuracy do you lose by making one model do everything vs training task-specific checkpoints? No systematic study at matched compute exists.
- **Task weighting.** How should the pretraining loss balance forecasting, classification, and anomaly signals? Current recipes use ad-hoc uniform weighting.
- **Zero-shot on *new* tasks.** Can a universal model handle a task type it was not pretrained on (e.g. changepoint detection) via head-only training, or does it need full retraining?
- **Cross-task interference.** Which tasks help each other and which ones conflict? There is almost no evidence beyond anecdote.
- **Benchmarks.** The multi-task evaluation story is fragmented. A cleanly leak-audited multi-task benchmark is still missing.

## Papers that exemplify this

- `[MOMENT](../papers/moment.md)` — canonical shared-encoder-plus-heads design; four task heads over a T5-init masked encoder.
- `[Timer](../papers/timer.md)` — generative unification via the S3 format on a causal decoder.
- `[Timer-XL](../papers/timer-xl.md)` — extends generative unification to multivariate and covariate with universal TimeAttention.
- `[UniTS](../papers/units.md)` — task-token interface on variable-length inputs across 38 datasets.
- `[TOTEM](../papers/totem.md)` — shared VQ-VAE codebook across domains, specialist-vs-generalist comparison.
- `[GPT4TS / OFA / FPT](../papers/gpt4ts.md)` — frozen-LLM universal solver across six TS tasks.
- `[Chronos-2](../papers/chronos-2.md)` — one architecture for univariate, multivariate, covariate, and panel forecasting via group attention.
- `[TSPulse](../papers/tspulse.md)` — a fifth recipe: instead of one head per task, pretrain three disentangled embedding *views* (TimeE / FFTE / register-token RegE) from a dual-space masked reconstruction, then pick the view per task via loss reweighting and post-hoc fusers (TSLens for classification, Multi-Head Triangulation for anomaly). Task specialisation by representation view, not by head. Explicitly not a forecaster.

- [In-context learning](in-context-learning.md)
- [Zero-shot forecasting](zero-shot-forecasting.md)
- [Value quantization](value-quantization.md)
- [Masked encoder](../architectures/masked-encoder.md)
- [Decoder-only autoregressive](../architectures/decoder-only-autoregressive.md)
- [LLM reprogramming](../architectures/llm-reprogramming.md)
