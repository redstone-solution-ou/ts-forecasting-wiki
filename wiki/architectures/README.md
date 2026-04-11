# Architectures

Time-series foundation models have converged on a small set of architecture families, most borrowed from language modeling and adapted to the continuous, multi-frequency, multivariate nature of temporal data. The main axes of variation are the training objective (autoregressive next-token vs. masked reconstruction vs. flow matching), how numeric values are represented (raw patches, quantized bins, VQ codes, or text), and whether a general-purpose LLM backbone is reused or a model is trained from scratch on time series.

## Sub-pages

- [Decoder-only autoregressive](decoder-only-autoregressive.md) — GPT-style next-patch prediction (TimesFM, Timer, Lag-Llama, Time-MoE).
- [Masked encoder](masked-encoder.md) — BERT-style masked-patch reconstruction (MOMENT, MOIRAI).
- [Encoder-decoder T5](encoder-decoder-t5.md) — seq2seq on tokenized values (Chronos, Chronos-2).
- [Mixture-of-experts](mixture-of-experts.md) — sparse routing for scale (Time-MoE, Moirai-MoE).
- [LLM reprogramming](llm-reprogramming.md) — frozen or aligned LLM backbones (Time-LLM, GPT4TS, LLMTime).
- [Lightweight non-transformer](lightweight-non-transformer.md) — MLP-Mixer and SSM alternatives (TTM, Mamba4Cast).
- [Flow-matching continuous](flow-matching-continuous.md) — continuous generative objectives without vocabularies (Sundial).
