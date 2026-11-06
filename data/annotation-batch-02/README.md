# Annotation batch 02 — independent test

Status: **20/20 human decisions collected and frozen-policy evaluation completed**. Original labels are preserved; the submitted file is archived at `artifacts/annotations-batch-02-submitted.txt`.

This package contains five new test queries and four rank-blind candidates per query, for 20 required decisions. It uses the same frozen synthetic multilingual smoke corpus as development batch 01 but has distinct query IDs and query text. This is query-level independence, not a new-domain or large-corpus benchmark.

## Independence rules

1. Read `../../docs/annotation-guidelines.md` before starting.
2. Do not inspect batch-01 labels, retrieval ranks, calibration reports, generated answers, or the frozen threshold while annotating.
3. For every candidate in `annotations.json`, set `relevance` to 0, 1, 2, or 3 and set `annotator` to the human adjudicator's stable identifier.
4. Judge only whether the chunk answers the query. Candidate order is ascending `chunk_id` and carries no relevance signal.
5. Use `notes` for disagreement or adjudication context. Keep independent raw annotations in a separate log; this file holds final adjudicated decisions.
6. Leave no candidate partially filled. The intended fifth query has no obvious answer in this snapshot, but annotators must decide independently from the text.

## Validate progress

From the repository root:

```bash
evidence-rag annotation-summary \
  --batch data/annotation-batch-02/annotations.json
```

The current summary shows 20 annotated pairs and 0 pending pairs. Export requires all decisions to remain complete and the corpus SHA-256 to match. One marginal invoice-background label makes the VAT-rate query answerable under the current any-positive-label rule; see the evaluation report for this limitation:

```bash
evidence-rag annotation-export \
  --batch data/annotation-batch-02/annotations.json \
  --chunks artifacts/chunks.jsonl \
  --output artifacts/human-gold-batch-02.json

evidence-rag validate-gold --gold-set artifacts/human-gold-batch-02.json
```

After export, evaluate the frozen dense policy without selecting a new threshold on batch 02:

```bash
evidence-rag evaluate-selective \
  --policy data/selective-policies/dense-e5-small-batch-01.json \
  --gold-set artifacts/human-gold-batch-02.json \
  --chunks artifacts/chunks.jsonl \
  --output artifacts/selective-evaluation-batch-02-dense.json
```

The evaluator checks batch identity and corpus SHA-256 and pins the recorded model revision. Batch-02 labels must not be used for threshold tuning.

## Recreate the pending package

```bash
evidence-rag annotation-create \
  --queries data/annotation-batch-02/queries.json \
  --chunks artifacts/chunks.jsonl \
  --output data/annotation-batch-02/annotations.json \
  --batch-id batch-02 \
  --name human-evidence-batch-02-test \
  --corpus-name synthetic-multilingual-smoke-corpus-v1 \
  --split test
```

Recreating overwrites decisions. Back up completed annotations and independent logs first.
