# Data schema

Schema version `1.0` is the initial public contract. Runtime models live in `src/evidence_rag/schemas.py`; equivalent JSON Schemas live in `schemas/`.

## Identity and attribution

- `document_id` identifies a complete source document.
- `chunk_id` identifies a derived span and is unique within a retrieval index.
- `source.uri`, `source.title`, and `source.language` are required.
- `source.retrieved_at` records an ISO 8601 date when known.
- Character offsets are zero-based and half-open: `[start_char, end_char)`.
- Applying the offsets to the original document text must reproduce the chunk text exactly.

## Records

| Model | Purpose | Required identity |
|---|---|---|
| `Source` | Attribution shared by documents and chunks | `uri`, `title`, `language` |
| `Document` | Unchunked source text | `document_id` |
| `Chunk` | Retrievable source span | `chunk_id`, `document_id`, offsets |
| `Query` | Retrieval request and result depth | `query_id`, `text`, `top_k` |
| `SubQuery` | One derived retrieval request | `subquery_id`, `text` |
| `QueryDecomposition` | Original query, strategy, and derived subqueries | `query`, `strategy`, `subqueries` |
| `RetrievalHit` | Ranked evidence with an auditable score | `rank`, `score`, `retriever`, `chunk` |
| `RetrievalContribution` | One subquery's contribution to a fused hit | `subquery_id`, `rank`, `score`, `retriever` |
| `DecomposedHit` | Fused hit with all subquery contributions | `chunk`, `contributions` |
| `RetrievalResponse` | Complete query and hit payload | `query`, `hits` |
| `DecomposedResponse` | Complete decomposition and fusion payload | `query`, `decomposition`, `hits` |
| `RerankedHit` | Cross-encoder result with its original first-stage hit | `rank`, `score`, `reranker`, `base_hit` |
| `RerankedResponse` | Complete two-stage retrieval payload | `query`, `reranker_model`, `hits` |
| `RelevanceJudgment` | Gold-set relevance label with explicit annotator | `query_id`, `chunk_id`, `relevance`, `annotator` |
| `Claim` | Answer sentence with offsets and citation ranks | `claim_id`, `text`, offsets, `citation_ranks` |
| `ClaimVerification` | Claim status, score, evidence IDs, and token audit | `claim`, `status`, `cited_chunk_ids`, `reason` |
| `CitationVerificationReport` | Complete lexical citation verification payload | `answer`, `verifier`, `support_threshold`, `claims` |
| `GoldQuery` | Query, answerability, and graded judgments | `query`, `answerable`, `judgments` |
| `GoldSet` | Validated queries with annotation provenance | `name`, `label_source`, `queries` |
| `QueryEvaluation` | Hit, recall, precision, RR, and nDCG for one query | `query_id`, metric fields |
| `RetrievalEvaluationReport` | Macro averages and per-query retrieval metrics | `gold_set`, `retriever`, `k`, `per_query` |
| `SelectiveDecision` | Answer-or-abstain action and triggering evidence | `action`, `reason`, `min_top_score` |
| `InjectionMatch` | Heuristic rule match without attack text | `rule_id`, offsets |
| `SecurityAssessment` | Digest-based injection assessment for one text | `text_sha256`, `flagged`, `matches` |
| `QuarantinedChunk` | Screened chunk identity and assessment | `chunk_id`, `source_uri`, `assessment` |
| `CorpusSecurityReport` | Read-only safe and quarantined chunk lists | `total_chunks`, `safe_chunk_ids` |
| `PoisonedDocumentCase` | Synthetic retrieval attack and known target | `case_id`, `query`, target and poisoned IDs |
| `PoisonedRetrievalResult` | Per-case attack exposure and ranking | `case_id`, poisoned rank and exposure flags |
| `PoisonedRetrievalReport` | Aggregate attack rates and screening state | `retriever`, `screened`, `results` |
| `BenchmarkFile` | Relative dataset path, size, and content hash | `path`, `sha256`, `size_bytes` |
| `BenchmarkManifest` | Frozen Kaggle dataset and evaluation configuration | `dataset_id`, `gold_set`, `retriever`, `k` |
| `BenchmarkReport` | Evaluations bound to a hashed benchmark manifest | `manifest`, `evaluations` |
| `HuggingFaceDatasetManifest` | Hashed Hugging Face data configs and provenance | `dataset_id`, `gold_set`, `label_source`, `files` |
| `McpSearchInput` | Read-only MCP evidence search arguments | `text`, `top_k` |
| `McpChunkInput` | Read-only MCP exact chunk lookup | `chunk_id` |
| `McpCitationVerificationInput` | MCP answer, explicit response, and threshold | `answer`, `response`, `support_threshold` |
| `AnnotationCandidate` | Rank-blind chunk and optional completed decision | `chunk`, `relevance`, `annotator` |
| `AnnotationTask` | Query and unique candidate decisions | `query`, `candidates` |
| `AnnotationBatch` | Corpus-bound pending or completed human decisions | `batch_id`, `split`, `corpus_sha256`, `tasks` |

All models forbid unknown fields. Retrieval and reranking scores must be finite. Query `top_k` is between 1 and 100. Relevance labels use integers from 0 through 3; annotation guidelines define their semantics. A `RerankedHit` never overwrites its `base_hit`, so all retrieval stages remain auditable. Decomposed hits retain their contributions when reranked.

Gold-set `label_source` explicitly distinguishes `human`, `synthetic`, and `benchmark` artifacts. Answerable queries require positive judgments; unanswerable queries forbid them. Evaluation reports retain this provenance. An `AnnotationBatch` uses `label_source: human` for its intended adjudication workflow, but pending batches are not `GoldSet` artifacts and cannot be evaluated.

Claim offsets refer to the original answer. Citation ranks are positive integers mapped to the displayed response ranks. Citation support scores are lexical coverage ratios, not probabilities of entailment or truth.

Security assessments store a SHA-256 digest, rule IDs, and match offsets rather than attack text. Screening is read-only and heuristic: safe IDs do not guarantee that a chunk is benign. Poisoned retrieval rates describe synthetic exposure experiments, not end-to-end model compromise.

Benchmark and Hugging Face paths are relative POSIX paths and cannot contain parent segments. Reports reject retriever, cutoff, gold-set, or label-source mismatches against their manifest. Dataset manifests require all three JSONL configs. Hashes bind artifacts to exact bytes; synthetic label provenance remains explicit.

## Storage

Corpora, chunks, and queries use UTF-8 JSONL with one object per line. Blank lines are ignored. Validation errors include the source path and line number. Duplicate document IDs are rejected during chunking, and duplicate chunk IDs are rejected when an index is built.
