# Time Series Pile

The Time Series Pile is the pretraining corpus assembled for MOMENT by the CMU Auton Lab. It aggregates multiple public time-series datasets into a single unified format so that a masked-encoder foundation model can be trained across forecasting, classification, anomaly, and imputation data in one pass.

## Overview

MOMENT's design — a T5-initialized encoder with PatchTST patching, [RevIN](../concepts/revin-normalization.md), and masked-patch reconstruction — needs a corpus that is (a) large enough to justify foundation-model pretraining and (b) heterogeneous enough to cover the multiple downstream tasks MOMENT is meant to handle via task heads. The Time Series Pile satisfies both requirements by pooling public TS datasets that were previously used in isolation for forecasting, classification, or anomaly detection, normalizing their formats, and delivering a unified stream suitable for masked pretraining.

The corpus is one of the key artifacts of the MOMENT release (alongside the three model sizes, Small 40M / Base 125M / Large 385M, and their HuggingFace checkpoints). Because MOMENT is open and the Time Series Pile is described explicitly, the corpus serves both as pretraining data for the released models and as a reproducible baseline for later work that wants to isolate architectural contributions from data contributions.

Compared to LOTSA and [Time-300B](./time-300b.md), the Time Series Pile's emphasis is breadth of task type rather than raw token count: LOTSA and Time-300B target forecasting-style pretraining with very large observation counts, while the Time Series Pile mixes in classification and anomaly datasets to support MOMENT's universal task-head interface.

## Key ideas / variants

- Unified format over multiple public TS datasets.
- Coverage across forecasting, classification, anomaly, imputation tasks.
- Published alongside open MOMENT HuggingFace checkpoints.
- Complements raw-volume corpora with task-type diversity.

## Papers that exemplify this (or use this)

- [MOMENT](../papers/moment.md) — created the Time Series Pile as MOMENT's pretraining corpus.

## Related wiki pages

- [Masked encoder](../architectures/masked-encoder.md)
- [Multi-task universal](../concepts/multi-task-universal.md)
- [LOTSA](lotsa.md)
