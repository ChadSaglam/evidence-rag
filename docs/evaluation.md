# Retrieval evaluation

The evaluation runner supports graded relevance from 0 through 3 and macro-averages metrics over answerable gold queries. Unanswerable queries are counted separately and excluded from retrieval metric means.

## Metrics at cutoff k

- **Hit@k:** whether at least one positive-relevance chunk appears in the top k.
- **Recall@k:** unique retrieved positive chunks divided by all positive gold chunks.
- **Precision@k:** retrieved positive chunks divided by k, including when fewer than k hits are returned.
- **Reciprocal rank:** inverse rank of the first positive chunk, or zero.
- **nDCG@k:** graded DCG divided by ideal DCG, using gain `2^relevance - 1` and logarithmic rank discount.

Duplicate retrieved chunk IDs are rejected. Queries without positive judgments are excluded by the runner; calling the metric function directly with no positive judgment raises an error.

## Run the synthetic smoke evaluation

```bash
evidence-rag validate-gold --gold-set data/sample/gold_set.json
evidence-rag evaluate \
  --gold-set data/sample/gold_set.json \
  --chunks artifacts/chunks.jsonl \
  --method bm25 \
  --k 4
```

The sample labels are synthetic and only demonstrate reproducibility. They are not a human benchmark or evidence of general retrieval quality. The same command accepts `dense`, `hybrid`, `--decompose`, and `--rerank` when the corresponding optional models are installed.

For a human evaluation, follow `docs/annotation-guidelines.md`, freeze the corpus and gold set, and report the label source, retriever configuration, cutoff, per-query metrics, and skipped unanswerable count.
