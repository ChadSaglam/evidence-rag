# Evidence RAG

[![Tests](https://github.com/ChadSaglam/evidence-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/ChadSaglam/evidence-rag/actions/workflows/tests.yml)

Evidence RAG is an evidence-first retrieval research repository. The initial baseline provides strict data contracts, deterministic chunking, BM25, multilingual dense retrieval with E5 prefixes, and reciprocal-rank fusion. It is designed for auditable experiments rather than answer generation without evidence.

The repository ships two corpora: a four-chunk synthetic fixture for offline tests, and a real multilingual corpus of 12 official GDPR articles in English, German and French (112 chunks, 30 queries, `data/real-gdpr/`). Nothing here is legal advice.

## Results on the real GDPR corpus

27 answerable queries in EN/DE/FR, k=10. Labels are deterministic article-level source alignment, not human judgments, so treat absolute numbers as indicative. Full tables, per-language nDCG, latency and pinned model revisions are in `results/gdpr-benchmark/report.md`.

| First stage | nDCG@10 | + cross-encoder rerank | Δ (95% bootstrap CI) |
|---|---|---|---|
| BM25 | 0.719 | 0.795 | +0.076 [+0.004, +0.152] |
| multilingual-e5-small | 0.704 | 0.761 | +0.058 [−0.011, +0.127] |
| hybrid RRF (BM25 + e5-small) | 0.748 | 0.790 | +0.042 [−0.022, +0.106] |
| multilingual-e5-base | **0.805** | 0.723 | **−0.081 [−0.146, −0.016]** |
| hybrid RRF (BM25 + e5-base) | 0.789 | 0.781 | −0.008 [−0.067, +0.047] |

The mMARCO cross-encoder (`mMiniLMv2-L12`) only helps weak first stages. On top of `multilingual-e5-base` it makes ranking significantly worse while adding about 1.7 s per query on CPU. A reranker is not a free upgrade, and the confidence intervals show that at 27 queries most deltas cannot be told apart from noise.

Citation verification on 38 constructed claims (`results/nli-benchmark/report.md`):

| Verifier | Binary accuracy | False-accept rate | False-reject rate |
|---|---|---|---|
| lexical overlap + numeric guard | 0.53 | **0.56** | 0.40 |
| multilingual NLI (`multilingual-MiniLMv2-L6-mnli-xnli`) | 0.82 | **0.06** | 0.30 |

Lexical overlap accepted every negated claim, because a negation reuses the evidence vocabulary. It also accepted numbers written as words ("six months"). NLI caught all of them. NLI still misses most free paraphrases, such as "three days" for "72 hours", so a paraphrased but correct answer can be wrongly rejected.

Building the real corpus surfaced two bugs that the four-chunk fixture had hidden. Both are fixed and covered by tests:

- BM25 returned chunks with a score of zero in chunk-ID order, and hybrid RRF then gave those chunks lexical credit.
- Hybrid search crashed on any corpus with more than 100 chunks.

## Current scope

- Runtime-validated Pydantic contracts and exported JSON Schemas
- UTF-8 JSONL corpus input with source attribution and exact character offsets
- Unicode-aware BM25 with deterministic tie-breaking
- Multilingual dense retrieval, defaulting to `intfloat/multilingual-e5-small`
- Hybrid retrieval using reciprocal-rank fusion
- Multilingual cross-encoder reranking with preserved first-stage scores
- Deterministic query decomposition with per-subquery fusion contributions
- Claim-level citation resolution, lexical support scoring, and numeric guards
- Optional multilingual NLI verification that labels claims as entailed, contradicted, or unsupported
- Two-annotator workflow with a browser labelling page, Cohen's κ, and adjudication
- Read-only MCP server validated by a raw JSON-RPC client that does not use the MCP SDK
- Offline deterministic tests using fixed embeddings and model scores

## Quickstart

For day-to-day local use, demo interpretation, CLI examples, custom corpora, and evaluation commands, see `docs/usage.md`.

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Validate and chunk the sample corpus:

```bash
evidence-rag validate --documents data/sample/documents.jsonl
evidence-rag chunk \
  --documents data/sample/documents.jsonl \
  --output artifacts/chunks.jsonl
```

Run lexical retrieval without a model download:

```bash
evidence-rag search \
  --chunks artifacts/chunks.jsonl \
  --method bm25 \
  --query "accounting records" \
  --top-k 2
```

The output is JSON containing the query, ranks, scores, chunk text, character offsets, and source attribution.

## Dense and hybrid retrieval

Install the optional embedding dependency:

```bash
pip install -e ".[dense]"
```

Then select `dense` or `hybrid`:

```bash
evidence-rag search \
  --chunks artifacts/chunks.jsonl \
  --method dense \
  --query "Wie lange müssen Geschäftsbücher aufbewahrt werden?" \
  --top-k 2
```

The first real dense run downloads the configured model from Hugging Face. The default model can be changed with `--model`; a device can be selected with `--device`. Tests never download a model.

## Query decomposition

The offline heuristic baseline splits explicit question boundaries, semicolons, newlines, and substantial clauses linked by English, German, French, or Turkish conjunctions. Short noun phrases remain intact. It is a deterministic baseline, not a semantic parser.

```bash
evidence-rag search \
  --chunks artifacts/chunks.jsonl \
  --method bm25 \
  --query "What is the retention period and what are the invoice requirements?" \
  --decompose \
  --max-subqueries 4 \
  --top-k 3
```

Each subquery is retrieved independently. Results are fused with reciprocal-rank fusion, and every fused hit records the contributing subquery IDs, ranks, scores, and retrievers. Decomposition can be combined with `--rerank`; the contribution trail is preserved inside each `base_hit`.

## Cross-encoder reranking

Install the optional reranking dependency:

```bash
pip install -e ".[rerank]"
```

Rerank a deeper BM25 candidate set with a multilingual cross-encoder:

```bash
evidence-rag search \
  --chunks artifacts/chunks.jsonl \
  --method bm25 \
  --query "How long must accounting records be retained?" \
  --candidate-k 4 \
  --top-k 2 \
  --rerank
```

The default reranker is `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`. Override it with `--rerank-model`. Reranking also works after `dense` or `hybrid` retrieval. Each output hit contains a `base_hit`, preserving the original rank, score, and retriever alongside the new cross-encoder score. The first real run downloads the configured model.

Reproduce the GDPR benchmark. Each first stage is cached, so the run can be split and resumed:

```bash
pip install -e ".[dense,rerank]"
python scripts/run_gdpr_benchmark.py
```

## Claim-level citation verification

Save any retrieval response and verify an answer against its displayed ranks:

```bash
evidence-rag search \
  --chunks artifacts/chunks.jsonl \
  --query "accounting records" \
  --top-k 2 > artifacts/retrieval.json

evidence-rag verify \
  --response artifacts/retrieval.json \
  --answer "Accounting records must be retained for ten years [1]."
```

The verifier extracts claims, resolves citations, computes lexical coverage over cited chunks, and rejects missing claim numerals. Reports distinguish `lexically_supported`, `insufficient_support`, `uncited`, and `invalid_citation`. Lexical support is not semantic entailment or real-world truth. See `docs/citation-verification.md` for the exact contract and limitations.

### Semantic (NLI) verification

```bash
pip install -e ".[nli]"
evidence-rag verify \
  --response artifacts/retrieval.json \
  --answer "Accounting records must be retained for ten years [1]." \
  --verifier nli
```

The NLI verifier keeps the same citation-resolution and numeric checks. It then scores sentence windows of each cited chunk with a multilingual NLI model and reports `semantically_supported`, `contradicted`, or `insufficient_support`, together with entailment and contradiction probabilities. Reproduce the comparison with `python scripts/run_nli_benchmark.py`.

## Gold sets and retrieval evaluation

Gold sets validate query IDs, graded judgments, duplicate labels, answerability, and annotation provenance. The runner computes Hit@k, Recall@k, Precision@k, reciprocal rank, and nDCG@k with per-query results and macro averages. Unanswerable queries are counted separately.

```bash
evidence-rag validate-gold --gold-set data/sample/gold_set.json
evidence-rag evaluate \
  --gold-set data/sample/gold_set.json \
  --chunks artifacts/chunks.jsonl \
  --method bm25 \
  --k 4
```

The bundled labels are synthetic smoke fixtures, not a human benchmark. See `docs/evaluation.md` for metric definitions and `docs/annotation-guidelines.md` for the human labeling workflow.

### Human annotation batches

`data/annotation-batch-01/` contains five development queries and 20 completed rank-blind query-chunk decisions. The submitted batch exports to `artifacts/human-gold-batch-01.json`; these development labels are not test-set results.

`data/annotation-batch-02/` contains five distinct test queries and 20 completed decisions over the same frozen corpus. The unchanged dense threshold achieved 0.4 coverage and zero empirical wrong-answer risk on this small test batch. See each batch README before editing decisions.

```bash
evidence-rag annotation-summary \
  --batch data/annotation-batch-01/annotations.json

evidence-rag annotation-summary \
  --batch data/annotation-batch-02/annotations.json

evidence-rag annotation-export \
  --batch data/annotation-batch-02/annotations.json \
  --chunks artifacts/chunks.jsonl \
  --output artifacts/human-gold-batch-02.json
```

Export requires every candidate to have a 0–3 relevance label and an explicit human annotator, and the corpus SHA-256 must match.

## Kaggle benchmark

Export a self-contained dataset with package source and SHA-256 file hashes, then generate deterministic JSON and Markdown evaluation reports:

```bash
evidence-rag kaggle-export \
  --documents data/sample/documents.jsonl \
  --chunks artifacts/chunks.jsonl \
  --gold-set data/sample/gold_set.json \
  --output-dir artifacts/kaggle-dataset \
  --dataset-id YOUR_KAGGLE_USERNAME/evidence-rag-benchmark \
  --k 4

evidence-rag benchmark-run \
  --manifest artifacts/kaggle-dataset/benchmark-manifest.json \
  --output-dir artifacts/kaggle-results
```

Export is local and does not publish data. The offline notebook is `kaggle/benchmark.ipynb`. Synthetic fixture metrics are not human benchmark results. See `docs/kaggle-benchmark.md`.

## Hugging Face dataset and demo

Export a dataset card with `documents`, `chunks`, and `gold_set` configs plus a SHA-256 manifest:

```bash
evidence-rag hf-export \
  --documents data/sample/documents.jsonl \
  --chunks artifacts/chunks.jsonl \
  --gold-set data/sample/gold_set.json \
  --output-dir artifacts/huggingface-dataset \
  --dataset-id YOUR_HF_USERNAME/evidence-rag

evidence-rag hf-verify \
  --manifest artifacts/huggingface-dataset/dataset-manifest.json
```

Install the optional demo and start the read-only BM25 interface:

```bash
pip install -e ".[demo]"
npm run dev -- --host 127.0.0.1 --port 7100
```

The demo returns attributed retrieval JSON, not generated answers. Export and demo setup do not publish datasets or Spaces. See `docs/huggingface.md`.

## Selective answering

`SelectiveAnswerer` returns `answer` or `abstain` after evidence-score and citation-verification gates. Thresholds must be calibrated separately for each retriever on a labeled development set.

```bash
evidence-rag calibrate-selective \
  --gold-set artifacts/human-gold-batch-01.json \
  --chunks artifacts/chunks.jsonl \
  --method bm25 \
  --max-risk 0 \
  --output artifacts/selective-calibration-bm25.json

evidence-rag evaluate-selective \
  --policy data/selective-policies/dense-e5-small-batch-01.json \
  --gold-set artifacts/human-gold-batch-02.json \
  --chunks artifacts/chunks.jsonl \
  --output artifacts/selective-evaluation-batch-02-dense.json

evidence-rag selective \
  --response artifacts/retrieval.json \
  --verification artifacts/verification.json \
  --min-top-score 0.5
```

Calibration maximizes coverage within an empirical wrong-answer risk limit. A correct answer decision means the top-1 chunk has a positive gold judgment; it does not prove generated-answer correctness. The report includes every candidate threshold and per-query evidence. The decision records its reason, score, threshold, chunk ID, and citation status. Neither output is a calibrated probability of correctness.

The selected dense threshold and model revision are frozen in `data/selective-policies/dense-e5-small-batch-01.json`. Batch 02 is reserved for unchanged-threshold evaluation after human annotation. See `docs/selective-answering.md`.

## RAG security tests

Treat retrieved text as untrusted input. The security commands provide deterministic heuristic injection screening and poisoned-document retrieval experiments without executing document-supplied instructions.

```bash
evidence-rag security-scan --chunks artifacts/chunks.jsonl
evidence-rag security-test --cases data/security/poisoned_cases.json
evidence-rag security-test --cases data/security/poisoned_cases.json --screen
```

Screening quarantines matching chunks without modifying the source corpus. Reports use text digests, rule IDs, and offsets instead of reproducing attack text. Heuristic matching is not a security guarantee and does not reliably detect misinformation or subtle poisoning. See `docs/security.md`.

## Read-only MCP tools

Serve deterministic evidence search, exact chunk lookup, and claim-level citation verification over stdio:

```bash
pip install -e ".[mcp]"
evidence-rag mcp-serve --chunks artifacts/chunks.jsonl
```

The allowlist contains only `search_evidence`, `get_evidence_chunk`, and `verify_citations`. All tools declare read-only, non-destructive, idempotent, closed-world annotations. No generation, corpus-mutation, shell, or network tool is registered. Validate the stdio server with a hand-written JSON-RPC client that shares no code with the MCP SDK. It checks unknown-tool and unknown-field rejection and a clean shutdown:

```bash
python scripts/validate_mcp_raw.py --chunks data/real-gdpr/chunks.jsonl
```

See `docs/mcp.md` for client configuration and protocol tests.

## Data contracts

Documents and chunks use versioned JSONL records. Every chunk retains its document ID, source, and half-open character range `[start_char, end_char)`. All contracts reject unknown fields.

```json
{
  "schema_version": "1.0",
  "document_id": "doc-001",
  "text": "Source text",
  "source": {
    "uri": "https://example.test/source",
    "title": "Source title",
    "language": "en",
    "retrieved_at": "2026-01-01"
  },
  "metadata": {}
}
```

Regenerate the public schemas after changing a contract:

```bash
evidence-rag export-schemas --output-dir schemas
```

See `docs/data-schema.md` for field rules and `docs/architecture.md` for module boundaries.

## Repository layout

```text
src/evidence_rag/
  retrieval/       BM25, dense, and hybrid baselines
  decomposition/   Query decomposition and subquery fusion
  reranking/       Cross-encoder adapter and two-stage pipeline
  citation/        Claim extraction, lexical and NLI citation verification
  evaluation/      Gold sets, retrieval metrics, and selective answering
  annotation/      Rank-blind batches and strict human-gold export
  benchmark/       Kaggle export and deterministic benchmark reports
  huggingface/     Dataset export and read-only demo service
  security/        Injection screening and poisoned retrieval tests
  mcp/             Read-only MCP service and stdio adapter
  chunking.py      Deterministic source-span chunking
  corpus.py        Strict JSONL input and output
  schemas.py       Runtime data contracts
data/real-gdpr/   Official GDPR articles (EN/DE/FR), queries, source-aligned labels
data/citation-nli/ Constructed claims for lexical vs NLI verification
data/annotation-batch-03/ Two-annotator GDPR package and browser labelling page
data/sample/      Synthetic multilingual fixtures and smoke labels
data/security/    Synthetic poisoned-document cases
data/gold-template/ Empty human annotation template
data/annotation-batch-01/ Completed human development annotations
data/annotation-batch-02/ Completed independent human test annotations
data/selective-policies/ Frozen threshold and model configurations
demo/             Gradio Space and local preview
kaggle/           Offline Kaggle benchmark notebook
schemas/          Exported JSON Schemas
tests/            Offline deterministic tests and fixtures
```

## Roadmap

1. Collect the two independent human annotations in `data/annotation-batch-03/` and report agreement against the source-aligned labels.
2. Grow the claim-verification set beyond 38 single-curator cases and calibrate NLI thresholds on a held-out split.
3. Benchmark stronger rerankers (for example `bge-reranker-v2-m3`) on GPU.

Each later stage must preserve deterministic fixtures, source attribution, and an offline test path.
