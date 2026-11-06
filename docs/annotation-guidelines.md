# Gold-set annotation guidelines

A human-labeled gold set requires annotations from identifiable human annotators. The files under `data/sample/` are synthetic software fixtures and must never be relabeled as human work.

## Query requirements

- Preserve the original query text, language, and stable query ID.
- Include answerable and unanswerable queries in separate evaluation slices.
- An answerable query must have at least one positively labeled chunk.
- An unanswerable query must have no positively labeled chunk in the corpus snapshot.
- Record corpus and chunk versions elsewhere in the experiment manifest before comparing reports.

## Relevance scale

| Value | Label | Guideline |
|---|---|---|
| 0 | Irrelevant | Does not help answer the query |
| 1 | Marginal | Useful background, but does not directly answer the central request |
| 2 | Relevant | Answers an important part of the query or supports a necessary claim |
| 3 | Essential | Directly answers the central request or is indispensable evidence |

Judge relevance to the query, not writing quality or agreement with a generated answer. Each query-chunk pair receives one adjudicated label. Keep disagreements in a separate annotation log rather than silently overwriting them.

## Human workflow

1. Train annotators on example pairs that are not part of the reported test set.
2. Annotate independently without showing retrieval ranks or model answers.
3. Record an annotator identifier and optional note for every judgment.
4. Adjudicate disagreements and validate the resulting `GoldSet` artifact.
5. Freeze a development set for threshold calibration and a separate test set for reporting.

## First development batch

Start with `data/annotation-batch-01/README.md` and the completed `annotations.json` in that directory. It contains five queries and 20 exhaustive query-chunk decisions over the small synthetic corpus snapshot. Candidates are ordered by chunk ID, not retrieval rank.

```bash
evidence-rag annotation-summary \
  --batch data/annotation-batch-01/annotations.json
```

Complete every `relevance` and `annotator` pair before export. The batch stores the exact corpus SHA-256 and development split. Export derives answerability from the final labels and rejects incomplete decisions, obvious nonhuman annotator placeholders, and changed corpus bytes. These checks support but do not prove the human annotation process.

For a different batch, start from `data/gold-template/gold_set.json` or use `annotation-create`, then run:

```bash
evidence-rag validate-gold --gold-set path/to/gold_set.json
```

No human performance claim should be made from the synthetic smoke set.
