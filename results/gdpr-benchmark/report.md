# GDPR retrieval and reranking benchmark

Corpus `data/real-gdpr/chunks.jsonl`, gold set `gdpr-multilingual-source-aligned-v1` (label source: **benchmark**, not human), 27 answerable queries in EN/DE/FR, k=10, rerank candidate depth=30.

| Run | Hit@k | Recall@k | MRR | nDCG@k | nDCG EN | nDCG DE | nDCG FR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 1.000 | 0.793 | 0.877 | **0.719** | 0.756 | 0.694 | 0.708 | 0 | 3 |
| bm25+rerank | 1.000 | 0.798 | 1.000 | **0.795** | 0.812 | 0.794 | 0.780 | 3492 | 4150 |
| dense:multilingual-e5-small | 1.000 | 0.743 | 0.907 | **0.704** | 0.740 | 0.685 | 0.687 | 12 | 13 |
| dense:multilingual-e5-small+rerank | 1.000 | 0.794 | 0.981 | **0.761** | 0.808 | 0.767 | 0.710 | 1713 | 1861 |
| hybrid:multilingual-e5-small | 1.000 | 0.759 | 0.957 | **0.748** | 0.785 | 0.725 | 0.734 | 12 | 13 |
| hybrid:multilingual-e5-small+rerank | 1.000 | 0.807 | 1.000 | **0.790** | 0.826 | 0.781 | 0.762 | 1648 | 1869 |
| dense:multilingual-e5-base | 1.000 | 0.841 | 1.000 | **0.805** | 0.856 | 0.778 | 0.780 | 26 | 31 |
| dense:multilingual-e5-base+rerank | 1.000 | 0.778 | 0.957 | **0.723** | 0.753 | 0.714 | 0.703 | 1721 | 1890 |
| hybrid:multilingual-e5-base | 1.000 | 0.824 | 0.981 | **0.789** | 0.821 | 0.782 | 0.763 | 26 | 32 |
| hybrid:multilingual-e5-base+rerank | 1.000 | 0.816 | 0.981 | **0.781** | 0.816 | 0.760 | 0.766 | 1651 | 1866 |

## Reranking effect on nDCG@10 (paired bootstrap, 10,000 resamples)

| First stage | Δ nDCG | 95% CI | improved / worse queries |
|---|---|---|---|
| bm25 | +0.076 | [+0.004, +0.152] | 12 / 8 |
| dense:multilingual-e5-small | +0.058 | [-0.011, +0.127] | 15 / 10 |
| hybrid:multilingual-e5-small | +0.042 | [-0.022, +0.106] | 12 / 9 |
| dense:multilingual-e5-base | -0.081 | [-0.146, -0.016] | 8 / 15 |
| hybrid:multilingual-e5-base | -0.008 | [-0.067, +0.047] | 11 / 11 |

## Pinned models

- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` @ `1427fd652930e4ba29e8149678df786c240d8825`
- `intfloat/multilingual-e5-small` @ `614241f622f53c4eeff9890bdc4f31cfecc418b3`
- `intfloat/multilingual-e5-base` @ `d128750597153bb5987e10b1c3493a34e5a4502a`

Environment: Python 3.10.12, Linux-6.8.0-138-generic-aarch64-with-glibc2.35, device `cpu`. Latency is wall-clock per query on this machine, including query encoding.

Labels are deterministic article-level source alignment. A CI that crosses zero means the reranker effect is not distinguishable from noise at this sample size.
