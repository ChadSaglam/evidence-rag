# Using Evidence RAG locally

Evidence RAG is a read-only evidence retrieval and evaluation project. It returns source-attributed chunks and retrieval metrics; it is not a chatbot, answer generator, or legal-advice system.

## Start the demo

```bash
cd evidence-rag
source .venv/bin/activate
pip install -e ".[demo]"
npm run dev -- --host 127.0.0.1 --port 7100
```

Open `http://127.0.0.1:7100`. Enter a query, choose the number of results, and select **Search**. You can also submit with Enter or choose an example.

Stop the server with `Ctrl+C` in the terminal running it.

## Read the result

The demo returns a `RetrievalResponse` JSON object:

- `rank`: result order, starting at 1.
- `score`: retriever-specific relevance score, not a probability or correctness estimate.
- `retriever`: scoring method; the demo uses `bm25`.
- `chunk.text`: retrieved source passage.
- `chunk.source`: source URI, title, language, and retrieval date.
- `start_char` and `end_char`: exact source span.

Inspect the text and source rather than relying on score alone. BM25 scores are not comparable with dense, hybrid, or cross-encoder scores. A requested result count can include zero-score chunks.

The bundled demo contains four synthetic chunks. Its current example query, `How long must accounting records be retained?`, retrieves `en-retention:000000` at rank 1; the passage states ten years.

## Use the CLI

BM25 requires no model download:

```bash
evidence-rag search \
  --chunks artifacts/chunks.jsonl \
  --method bm25 \
  --query "accounting records" \
  --top-k 2
```

Dense retrieval uses the pinned installation and configured multilingual E5 model:

```bash
evidence-rag search \
  --chunks artifacts/chunks.jsonl \
  --method dense \
  --model intfloat/multilingual-e5-small \
  --query "How long must accounting records be retained?" \
  --top-k 2
```

Use `--method hybrid` for reciprocal-rank fusion of BM25 and dense rankings. The first dense run can download the model; later runs use the local cache.

## Search another corpus

Validate and chunk UTF-8 JSONL documents:

```bash
evidence-rag validate --documents path/to/documents.jsonl
evidence-rag chunk \
  --documents path/to/documents.jsonl \
  --output artifacts/custom-chunks.jsonl
```

Use that file with `evidence-rag search --chunks`, or start the demo with:

```bash
EVIDENCE_RAG_CHUNKS=artifacts/custom-chunks.jsonl \
  npm run dev -- --host 127.0.0.1 --port 7100
```

Input documents must satisfy the source and offset contracts in `docs/data-schema.md`. Corpus conversion does not create human relevance labels.

## Reproduce the human test evaluation

```bash
evidence-rag annotation-summary \
  --batch data/annotation-batch-02/annotations.json

evidence-rag validate-gold \
  --gold-set artifacts/human-gold-batch-02.json

evidence-rag evaluate-selective \
  --policy data/selective-policies/dense-e5-small-batch-01.json \
  --gold-set artifacts/human-gold-batch-02.json \
  --chunks artifacts/chunks.jsonl \
  --output artifacts/selective-evaluation-batch-02-dense.json
```

Expected fixed-policy metrics: coverage `0.4`, selective accuracy `1.0` over two answered queries, wrong-answer risk `0.0`, and decision accuracy `0.6`. The threshold remains `0.822904370645214`.

## Verify the project

```bash
pytest
ruff check .
```

The completed milestone has 142 passing tests. Human annotation files, frozen policies, and exported reports use different roles: do not overwrite development batch 01 with test batch 02, and do not tune a threshold on the test batch.

## Current limits

The corpus is synthetic and tiny, batch 02 has one annotator, and its marginal invoice-background label means it has no all-zero unanswerable query. Real-corpus collection, additional human annotation, semantic answer verification, and publication to external accounts are separate follow-up projects, not features of the local demo.
