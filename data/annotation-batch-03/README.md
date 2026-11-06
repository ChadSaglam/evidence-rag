# Annotation batch 03: two independent annotators on the real GDPR corpus

Status: **pending human annotation (0/50 pairs per annotator)**.

This batch is how the project gets inter-annotator agreement. It has seven queries (EN/DE/FR, one deliberately unanswerable) over `data/real-gdpr/chunks.jsonl`, with 50 pooled query–chunk pairs. Each candidate came from the union of the BM25 top-5 and the `multilingual-e5-base` top-5, the usual pooling method when a corpus is too large to label exhaustively. Candidates are listed in `chunk_id` order, so no rank signal reaches the annotator. `pool.json` records which systems contributed each candidate.

Two people label the same 50 pairs separately. Agreement is computed from their files, disagreements are adjudicated, and only the adjudicated labels become gold.

## Rules for both annotators

1. Read `../../docs/annotation-guidelines.md` (relevance scale 0–3).
2. Do not talk about specific pairs with the other annotator until both files are submitted.
3. Do not look at retrieval reports, `data/real-gdpr/gold_set.json`, or the other copy.
4. Judge only whether the chunk text helps answer the query, not whether you agree with the law.

## Step by step (no terminal needed)

1. Open `annotate.html` in this folder in Chrome or Safari (double-click it).
2. Enter your annotator ID, e.g. `annotator-a-cs`.
3. Click **Choose file**. Annotator A picks `annotations-annotator-a.json`; annotator B picks `annotations-annotator-b.json`.
4. Label each pair by clicking 0–3 or pressing the `0`–`3` keys. Your progress is saved in the browser while you work.
5. When all 50 pairs are labelled, click **Download finished file** and put it back in this folder under the same name.

## After both files are back

```bash
evidence-rag annotation-agreement \
  --primary data/annotation-batch-03/annotations-annotator-a.json \
  --secondary data/annotation-batch-03/annotations-annotator-b.json \
  --provenance human \
  --output artifacts/batch-03-agreement.json
```

The report gives exact agreement, Cohen's κ, quadratic-weighted κ, and every disagreement. Record a resolution for each disagreement in `adjudication-log.json`, using the same format as batch 01, then run the command again with `--adjudications data/annotation-batch-03/adjudication-log.json` until it reports `complete: true`.

The adjudicated labels can then be compared with the source-aligned benchmark labels in `data/real-gdpr/gold_set.json`, which shows how far automated article-level alignment is from human relevance judgments.

## Rebuild the package

```bash
python scripts/build_annotation_batch_03.py
```

The script pins the dense model revision, so the pool rebuilds identically.
