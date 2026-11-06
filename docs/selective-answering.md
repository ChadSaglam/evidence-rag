# Selective answering

`SelectiveAnswerer` makes a deterministic answer-or-abstain decision. It does not generate an answer and does not estimate a calibrated probability of correctness.

## Gates

1. No hits: abstain with `no_evidence`.
2. Top score below `min_top_score`: abstain with `low_retrieval_score`.
3. Citations required but no report supplied: abstain with `citation_verification_required`.
4. Citation report not fully supported: abstain with `citation_verification_failed`.
5. All enabled gates pass: return `answer`.

Retrieval scores are not comparable across BM25, dense cosine, RRF, and cross-encoders. Every threshold must therefore be calibrated separately on a labeled development set and reported with the exact retriever configuration. Do not transfer a threshold between methods.

## CLI

```bash
evidence-rag selective \
  --response artifacts/retrieval.json \
  --verification artifacts/verification.json \
  --min-top-score 0.5
```

`--allow-unverified` disables the citation gate for pre-generation evidence checks. It must not be presented as answer verification. The decision records the threshold, top score, top chunk ID, citation status, action, and reason.

## Threshold calibration

Calibrate the pre-generation evidence gate on the reviewed development batch:

```bash
evidence-rag calibrate-selective \
  --gold-set artifacts/human-gold-batch-01.json \
  --chunks artifacts/chunks.jsonl \
  --method bm25 \
  --max-risk 0 \
  --output artifacts/selective-calibration-bm25.json
```

The calibrator retrieves top 1 for every gold query, including unanswerable queries. An answer decision is correct only if that chunk has a positive gold judgment. A missing or unjudged top chunk is therefore an incorrect answer decision. This is an evidence-selection proxy, not generated-answer correctness or semantic entailment.

For each unique observed score, and one score immediately above the maximum, the report computes:

- `coverage`: answered queries divided by all queries.
- `selective_accuracy`: correct answer decisions divided by answered queries.
- `wrong_answer_rate`: incorrect answer decisions divided by answered queries.
- `decision_accuracy`: correct answers plus correct abstentions, divided by all queries.

Zero-answered accuracy and risk are defined as 0. The selector maximizes coverage among points satisfying `wrong_answer_rate <= max_risk`, then maximizes decision accuracy, then chooses the lowest threshold. Candidate order and all tie-breaking are deterministic.

The citation gate is not part of this pre-generation calibration and remains required for final answering unless explicitly disabled. Calibrate separately for every retriever, decomposition, and reranking configuration. Evaluate the frozen threshold on a separate test set; development calibration metrics are not test performance.

## Reviewed batch 01 diagnostic

With BM25 and `max_risk=0`, the selected threshold is `2.6132743039632724`, coverage is 0, wrong-answer risk is 0, and decision accuracy is 0.8. Every candidate that answers at least one query has empirical wrong-answer risk above 0, so the zero-risk constraint selects abstention for all five queries. This is a development diagnostic showing inadequate top-1 selective evidence, not a production-ready threshold or a claim about the full top-4 ranking.

## Frozen dense policy

The dense E5 threshold `0.822904370645214` is frozen in `data/selective-policies/dense-e5-small-batch-01.json`. The policy records the exact model revision, corpus and calibration artifact hashes, citation requirement, and development metrics. Its development coverage is 0.2 with zero empirical wrong-answer risk.

`data/annotation-batch-02/` contains 20 completed human test decisions. The frozen threshold was evaluated unchanged:

```bash
evidence-rag evaluate-selective \
  --policy data/selective-policies/dense-e5-small-batch-01.json \
  --gold-set artifacts/human-gold-batch-02.json \
  --chunks artifacts/chunks.jsonl \
  --output artifacts/selective-evaluation-batch-02-dense.json
```

The command validates the policy contract, evaluation batch identity, and corpus SHA-256. Dense evaluation pins the recorded Hugging Face model revision. Its report contains one fixed-threshold metrics object, not calibration candidates. The metrics still describe top-1 evidence selection rather than generated-answer or citation correctness.

Batch-02 test results are 0.4 coverage, 1.0 selective accuracy over two answered queries, 0.0 empirical wrong-answer risk, and 0.6 decision accuracy. The test gold set has no all-zero unanswerable query because one marginal invoice-background label makes the VAT-rate query answerable under the current export rule. See `artifacts/selective-evaluation-batch-02-dense.md` for per-query interpretation and limitations.

Do not select a new threshold from batch 02; any retuning requires a new development split and a new untouched test split.
