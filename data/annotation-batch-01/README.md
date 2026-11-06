# Annotation batch 01 — development

Status: **20/20 human decisions collected and relevance-scale review completed**. Six direct-answer labels were revised from 2 to 3 with owner approval; original annotator identifiers are preserved.

This package contains five new queries and four rank-blind candidates per query, for 20 required decisions. The corpus is the synthetic multilingual smoke snapshot; synthetic source text does not make the submitted human judgments synthetic. The later scale review is recorded in `adjudication-log.json`; this log does not reconstruct independent pre-adjudication annotations.

## Annotator instructions

1. Read `../../docs/annotation-guidelines.md` before starting.
2. For every candidate in `annotations.json`, set `relevance` to 0, 1, 2, or 3 and set `annotator` to the human adjudicator's stable identifier.
3. Judge only whether the chunk answers the query. Do not infer labels from candidate order, synthetic gold-set files, retrieval ranks, or generated answers.
4. Use `notes` for disagreement or adjudication context. Keep independent raw annotations in a separate log; this file holds final adjudicated decisions.
5. Leave no candidate partially filled. An answerable query is derived from at least one positive label; an unanswerable query has only zero labels.

Candidate order is ascending `chunk_id` and carries no relevance signal. The intended fifth query has no obvious answer in this snapshot, but annotators must decide independently from the text.

## Validate progress

From the repository root:

```bash
evidence-rag annotation-summary \
  --batch data/annotation-batch-01/annotations.json
```

The current summary shows 20 annotated pairs and 0 pending pairs. Export requires all 20 decisions to remain complete and the corpus SHA-256 to match:

```bash
evidence-rag annotation-export \
  --batch data/annotation-batch-01/annotations.json \
  --chunks artifacts/chunks.jsonl \
  --output artifacts/human-gold-batch-01.json

evidence-rag validate-gold --gold-set artifacts/human-gold-batch-01.json
```

The exporter validates completeness, provenance strings, answerability, and corpus identity. It cannot prove that a person performed the work; the annotation process and adjudication log provide that evidence.

## Recreate the pending package

```bash
evidence-rag annotation-create \
  --queries data/annotation-batch-01/queries.json \
  --chunks artifacts/chunks.jsonl \
  --output data/annotation-batch-01/annotations.json \
  --batch-id batch-01 \
  --name human-evidence-batch-01-development \
  --corpus-name synthetic-multilingual-smoke-corpus-v1 \
  --split development
```

Recreating overwrites decisions. Back up completed annotations and independent logs first.
