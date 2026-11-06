# Architecture

The repository keeps ingestion, decomposition, retrieval, reranking, citation verification, and evaluation independent.

1. **Ingest:** validate `Document` records and derive `Chunk` records with stable IDs and exact source offsets.
2. **Decompose:** split a multi-intent query with a conservative deterministic heuristic.
3. **Retrieve:** run each subquery through BM25, multilingual dense cosine retrieval, or hybrid retrieval.
4. **Fuse:** combine subquery rankings with reciprocal-rank fusion and retain every contribution.
5. **Rerank:** retrieve a deeper candidate set and score original-query/passage pairs with a multilingual cross-encoder.
6. **Verify:** map answer claims to displayed evidence ranks and compute explicit lexical support checks.
7. **Inspect:** serialize complete response objects so every retrieval stage, score, and attribution remains visible.
8. **Evaluate:** add gold judgments and deterministic metrics without changing the corpus contract.

## Decomposition contract

`HeuristicQueryDecomposer` handles explicit question boundaries, semicolons, newlines, and substantial clauses linked by English, German, French, or Turkish conjunctions. It leaves short noun phrases intact and always returns at least one subquery. `DecomposedRetriever` can also accept another decomposer through a small protocol.

Each subquery receives a stable ID derived from the original query ID. `DecomposedHit` contains the fused RRF score and a list of `RetrievalContribution` records. This allows evaluation to distinguish evidence found by one intent from evidence shared across intents.

## Reranking contract

`RerankedRetriever` controls candidate depth and final output depth. `CrossEncoderReranker` accepts any model implementing the small `predict` interface. The SentenceTransformers adapter loads the real model only when instantiated. Every `RerankedHit` embeds its untouched `base_hit`, including decomposition contributions when present.

## Citation verification contract

`CitationClaimExtractor` converts answer sentences or lines into claims with source offsets and numeric citation ranks. `ClaimLevelCitationVerifier` accepts first-stage, decomposed, or reranked hits. For reranked hits, citations resolve to the outer displayed rank and evidence comes from the embedded base hit.

The verifier computes content-token coverage over the union of cited chunks and separately requires claim numerals to occur in that evidence. Its status is deliberately named `lexically_supported`; it does not claim semantic entailment. See `docs/citation-verification.md` for statuses and limitations.

## Evaluation and selective answering contracts

`GoldSet` validates stable query IDs, per-query judgment consistency, answerability, and explicit `human`, `synthetic`, or `benchmark` provenance. `evaluate_retriever` skips unanswerable queries for retrieval means and reports their count separately. Metric functions use fixed cutoffs and deterministic graded nDCG; the runner supports first-stage, decomposed, and reranked pipelines.

`SelectiveAnswerer` applies independent evidence-score and citation-verification gates. It returns an auditable decision rather than generated text or a correctness probability. Thresholds are configuration-specific and require development-set calibration.

## Annotation contract

`AnnotationBatch` binds rank-blind query-chunk candidates to an exact corpus SHA-256, split, and guidelines version. Candidates begin with no relevance or annotator and reject partially completed decisions. Candidate order is stable chunk ID order and carries no retrieval rank.

`export_human_gold_set` requires complete decisions, explicit non-placeholder annotator identifiers, and unchanged corpus bytes. It derives answerability from positive labels and emits `label_source: human`. Validation supports provenance but cannot prove human work; independent logs and adjudication remain part of the annotation process.

## Benchmark contract

`export_kaggle_dataset` copies canonical data inputs and package source into a self-contained dataset. `BenchmarkManifest` records relative paths, sizes, SHA-256 hashes, gold-set provenance, retriever, and cutoff without timestamps or machine-specific paths. `run_benchmark` verifies every file before invoking the existing evaluation runner. JSON and Markdown reports embed the manifest so metrics remain tied to exact input and source bytes.

The initial Kaggle runner is intentionally limited to offline BM25. Dense and reranker benchmarks must record model revisions and runtime environments separately.

## Hugging Face contract

`export_huggingface_dataset` validates source records, emits three JSONL dataset configs, renders a provenance-preserving dataset card, and hashes every exported file. `verify_huggingface_dataset` rejects missing or modified content. Export is local and does not call the Hub API.

`DemoSearch` exposes only deterministic BM25 retrieval as `RetrievalResponse` objects. The Gradio adapter is optional and contains no answer generation, external model calls, or corpus-mutation path. Dataset and Space publication are explicit user operations outside the offline test path.

## Security contract

`InjectionDetector` applies deterministic multilingual heuristics to untrusted chunk text. `screen_chunks` returns safe chunks and a quarantine report without modifying the corpus. Reports contain text digests, rule IDs, and offsets rather than attack text.

`evaluate_poisoned_retrieval` compares known target and poisoned chunk ranks, with optional screening before index construction. It measures retrieval exposure only. Pattern matching cannot guarantee detection of injections, misinformation, or subtle poisoning, and retrieval exposure does not prove answer-generation compromise.

## Determinism boundary

Chunking, decomposition, BM25, fusion, ranking, claim extraction, citation verification, gold-set validation, retrieval metrics, selective decisions, injection screening, poisoned retrieval evaluation, JSONL handling, and tests run offline. Dense retrieval accepts an encoder interface; tests inject fixed vectors. Cross-encoder tests inject documented scores from a multilingual JSON fixture. Citation fixtures cover valid, missing, and unknown citations, combined evidence, wrong numerals, and multilingual claims. Metric tests include manual nDCG calculations and unanswerable-query handling. Security fixtures cover multilingual injections, benign text, attack-text redaction, immutable screening, and subtle poisoning that evades heuristics. Benchmark tests cover byte-identical JSON and Markdown reports, content-hash tampering, path traversal, required inputs, and configuration mismatches. Hugging Face tests cover byte-identical exports, dataset cards, multilingual demo results, input validation, and hash tampering without Gradio or network access. MCP tests cover the exact allowlist, strict inputs, annotations, structured results, citation verification, unknown-tool rejection, and corpus immutability; a separate stdio smoke test covers initialize, tools/list, and tools/call. Annotation tests cover pending-state integrity, deterministic batch creation and export, partial decisions, duplicate candidates, corpus tampering, annotator placeholders, and derived answerability. Real model outputs can vary across model revisions and hardware, so model names and evaluation environments must be recorded by benchmark reports.

First-stage hits are ordered by descending score and then ascending `chunk_id`. Fused hits use the same rule. Reranked hits are ordered by descending cross-encoder score, then original rank, then `chunk_id`.

## MCP contract

`ReadOnlyEvidenceService` owns a frozen in-memory chunk sequence and dispatches only `search_evidence`, `get_evidence_chunk`, and `verify_citations`. Strict input models reject unknown fields. Citation verification requires an explicit response payload rather than hidden mutable session state.

The optional FastMCP adapter exposes the same allowlist over stdio with read-only, non-destructive, idempotent, closed-world annotations. It registers no generation, mutation, shell, credential, or network tool. Annotations are hints; the allowlist and absence of I/O operations enforce the read-only boundary.

## Future evaluation

Human relevance labels must remain separate from synthetic fixtures and lexical citation scores. Dense and reranker benchmarks require explicit model and environment provenance.
