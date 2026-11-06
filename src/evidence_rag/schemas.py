"""Versioned data contracts used by retrieval and future evaluation stages."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"


class ContractModel(BaseModel):
    """Reject unknown fields so schema drift is explicit."""

    model_config = ConfigDict(extra="forbid")


class Source(ContractModel):
    """Attribution for a document, retained on every derived chunk."""

    uri: str = Field(min_length=1)
    title: str = Field(min_length=1)
    language: str = Field(min_length=2, description="BCP 47 language tag, for example en or de-CH.")
    retrieved_at: str | None = Field(default=None, description="ISO 8601 retrieval date.")


class Document(ContractModel):
    """A source text before chunking."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: Source
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(ContractModel):
    """A text span with stable identity and exact source offsets."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=1)
    source: Source
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_offsets(self) -> "Chunk":
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        return self


class Query(ContractModel):
    """A retrieval request; language is metadata, not a routing restriction."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    query_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    language: str | None = Field(default=None, min_length=2)
    top_k: int = Field(default=10, ge=1, le=100)


class SubQuery(ContractModel):
    """One deterministic retrieval request derived from an original query."""

    subquery_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class QueryDecomposition(ContractModel):
    """Auditable decomposition output; one subquery means no split occurred."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    query: Query
    strategy: str = Field(min_length=1)
    subqueries: list[SubQuery] = Field(min_length=1)


class RetrievalHit(ContractModel):
    """A ranked chunk and the retriever-specific score that selected it."""

    rank: int = Field(ge=1)
    score: float = Field(allow_inf_nan=False)
    retriever: str = Field(min_length=1)
    chunk: Chunk


class RetrievalContribution(ContractModel):
    """The rank and score contributed by one subquery to a fused hit."""

    subquery_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    score: float = Field(allow_inf_nan=False)
    retriever: str = Field(min_length=1)


class DecomposedHit(RetrievalHit):
    """A fused hit that retains every contributing subquery ranking."""

    contributions: list[RetrievalContribution] = Field(min_length=1)


class RetrievalResponse(ContractModel):
    """Serializable search output, including enough context to audit results."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    query: Query
    hits: list[RetrievalHit]


class DecomposedResponse(ContractModel):
    """Serializable decomposition and reciprocal-rank fusion output."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    query: Query
    decomposition: QueryDecomposition
    hits: list[DecomposedHit]


class RerankedHit(ContractModel):
    """A cross-encoder score together with its untouched first-stage hit."""

    rank: int = Field(ge=1)
    score: float = Field(allow_inf_nan=False)
    reranker: str = Field(min_length=1)
    base_hit: RetrievalHit | DecomposedHit


class RerankedResponse(ContractModel):
    """Serializable two-stage retrieval output with model identity."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    query: Query
    decomposition: QueryDecomposition | None = None
    reranker_model: str = Field(min_length=1)
    hits: list[RerankedHit]


class RelevanceJudgment(ContractModel):
    """A human-, synthetic-, or benchmark-supplied relevance label."""

    query_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    relevance: int = Field(ge=0, le=3)
    annotator: str = Field(min_length=1)
    notes: str | None = None


class AdjudicationDecision(ContractModel):
    """An explicit final decision for one annotator disagreement."""

    query_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    relevance: int = Field(ge=0, le=3)
    adjudicator: str = Field(min_length=1)
    notes: str = Field(min_length=1)


class AnnotationDisagreement(ContractModel):
    """Two differing labels and their optional adjudication outcome."""

    query_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    primary_relevance: int = Field(ge=0, le=3)
    secondary_relevance: int = Field(ge=0, le=3)
    adjudicated_relevance: int | None = Field(default=None, ge=0, le=3)
    adjudicator: str | None = None
    notes: str | None = None


class AnnotationAgreementReport(ContractModel):
    """Pairwise ordinal-label agreement with explicit provenance and adjudication."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    provenance: Literal["human", "benchmark", "synthetic"]
    primary_annotator: str = Field(min_length=1)
    secondary_annotator: str = Field(min_length=1)
    pair_count: int = Field(ge=1)
    exact_agreement_rate: float = Field(ge=0, le=1)
    cohens_kappa: float | None = Field(default=None, ge=-1, le=1)
    quadratic_weighted_kappa: float | None = Field(default=None, ge=-1, le=1)
    disagreement_count: int = Field(ge=0)
    adjudicated_count: int = Field(ge=0)
    complete: bool
    disagreements: list[AnnotationDisagreement]


ClaimStatus = Literal[
    "lexically_supported",
    "semantically_supported",
    "contradicted",
    "insufficient_support",
    "uncited",
    "invalid_citation",
]


class Claim(ContractModel):
    """One answer sentence, its source offsets, and numeric citation ranks."""

    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=1)
    citation_ranks: list[int]

    @model_validator(mode="after")
    def validate_claim(self) -> "Claim":
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        if any(rank < 1 for rank in self.citation_ranks):
            raise ValueError("citation_ranks must contain positive integers")
        return self


class ClaimVerification(ContractModel):
    """Auditable lexical support evidence for one cited answer claim."""

    claim: Claim
    status: ClaimStatus
    support_score: float | None = Field(default=None, allow_inf_nan=False)
    entailment_score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    contradiction_score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    cited_chunk_ids: list[str]
    matched_tokens: list[str]
    missing_tokens: list[str]
    reason: str = Field(min_length=1)


class CitationVerificationReport(ContractModel):
    """Complete claim-level verification output with an explicit method boundary."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    answer: str
    verifier: str = Field(min_length=1)
    support_threshold: float = Field(gt=0, le=1)
    fully_supported: bool
    claims: list[ClaimVerification]


class GoldQuery(ContractModel):
    """A query and its graded relevance judgments."""

    query: Query
    answerable: bool = True
    judgments: list[RelevanceJudgment] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_judgments(self) -> "GoldQuery":
        chunk_ids: set[str] = set()
        for judgment in self.judgments:
            if judgment.query_id != self.query.query_id:
                raise ValueError("Judgment query_id must match the gold query_id")
            if judgment.chunk_id in chunk_ids:
                raise ValueError(f"Duplicate judgment for chunk_id: {judgment.chunk_id}")
            chunk_ids.add(judgment.chunk_id)
        has_relevant = any(item.relevance > 0 for item in self.judgments)
        if self.answerable and not has_relevant:
            raise ValueError("Answerable gold queries require a positive relevance judgment")
        if not self.answerable and has_relevant:
            raise ValueError("Unanswerable gold queries cannot have positive judgments")
        return self


LabelSource = Literal["human", "synthetic", "benchmark"]


class GoldSet(ContractModel):
    """A validated collection with an explicit annotation provenance label."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    label_source: LabelSource
    queries: list[GoldQuery]

    @model_validator(mode="after")
    def validate_unique_queries(self) -> "GoldSet":
        query_ids = [item.query.query_id for item in self.queries]
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("Gold query_id values must be unique")
        return self


class QueryEvaluation(ContractModel):
    """Retrieval metrics for one answerable gold query."""

    query_id: str = Field(min_length=1)
    hit_at_k: bool
    recall_at_k: float = Field(ge=0, le=1)
    precision_at_k: float = Field(ge=0, le=1)
    reciprocal_rank: float = Field(ge=0, le=1)
    ndcg_at_k: float = Field(ge=0, le=1)


class RetrievalEvaluationReport(ContractModel):
    """Macro-averaged retrieval metrics and auditable per-query values."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    gold_set: str = Field(min_length=1)
    gold_label_source: LabelSource
    retriever: str = Field(min_length=1)
    k: int = Field(ge=1, le=100)
    evaluated_queries: int = Field(ge=0)
    skipped_unanswerable_queries: int = Field(ge=0)
    mean_hit_at_k: float = Field(ge=0, le=1)
    mean_recall_at_k: float = Field(ge=0, le=1)
    mean_precision_at_k: float = Field(ge=0, le=1)
    mean_reciprocal_rank: float = Field(ge=0, le=1)
    mean_ndcg_at_k: float = Field(ge=0, le=1)
    per_query: list[QueryEvaluation]


class SelectiveDecision(ContractModel):
    """A deterministic answer-or-abstain decision and its triggering reason."""

    action: Literal["answer", "abstain"]
    reason: str = Field(min_length=1)
    top_score: float | None = Field(default=None, allow_inf_nan=False)
    min_top_score: float = Field(allow_inf_nan=False)
    top_chunk_id: str | None = None
    citation_fully_supported: bool | None = None


class SelectiveCalibrationQuery(ContractModel):
    """Top-1 evidence and its gold relevance for one calibration query."""

    query_id: str = Field(min_length=1)
    answerable: bool
    top_score: float | None = Field(default=None, allow_inf_nan=False)
    top_chunk_id: str | None = None
    top_chunk_relevance: int = Field(ge=0, le=3)
    top_chunk_relevant: bool


class SelectiveCalibrationPoint(ContractModel):
    """Outcomes for one deterministic top-score threshold."""

    threshold: float = Field(allow_inf_nan=False)
    answered_queries: int = Field(ge=0)
    correct_answers: int = Field(ge=0)
    wrong_answers: int = Field(ge=0)
    correct_abstentions: int = Field(ge=0)
    coverage: float = Field(ge=0, le=1)
    selective_accuracy: float = Field(ge=0, le=1)
    wrong_answer_rate: float = Field(ge=0, le=1)
    decision_accuracy: float = Field(ge=0, le=1)


class SelectiveCalibrationReport(ContractModel):
    """Risk-constrained threshold selection with auditable candidate outcomes."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    gold_set: str = Field(min_length=1)
    gold_label_source: LabelSource
    retriever: str = Field(min_length=1)
    objective: Literal["max_coverage_within_risk"] = "max_coverage_within_risk"
    max_risk: float = Field(ge=0, le=1)
    selected_threshold: float = Field(allow_inf_nan=False)
    query_count: int = Field(ge=1)
    answerable_queries: int = Field(ge=0)
    unanswerable_queries: int = Field(ge=0)
    selected: SelectiveCalibrationPoint
    candidates: list[SelectiveCalibrationPoint] = Field(min_length=1)
    per_query: list[SelectiveCalibrationQuery] = Field(min_length=1)


class FrozenSelectivePolicy(ContractModel):
    """Frozen retrieval and threshold configuration for an untouched test split."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    policy_id: str = Field(min_length=1)
    status: Literal["frozen_for_independent_evaluation"]
    frozen_at: str = Field(min_length=1)
    min_top_score: float = Field(allow_inf_nan=False)
    top_k: Literal[1]
    require_citations: bool
    decision_rule: str = Field(min_length=1)
    retriever: dict[str, Any]
    calibration: dict[str, Any]
    corpus: dict[str, Any]
    evaluation: dict[str, Any]

    @model_validator(mode="after")
    def validate_evaluation_lock(self) -> "FrozenSelectivePolicy":
        if self.evaluation.get("retune_allowed") is not False:
            raise ValueError("Frozen policy evaluation must forbid retuning")
        if not self.evaluation.get("batch_id"):
            raise ValueError("Frozen policy requires an evaluation batch_id")
        return self


class SelectivePolicyEvaluationReport(ContractModel):
    """Fixed-threshold evaluation without candidate selection or retuning."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    policy_id: str = Field(min_length=1)
    evaluation_split: str = Field(min_length=1)
    gold_set: str = Field(min_length=1)
    gold_label_source: LabelSource
    retriever: str = Field(min_length=1)
    min_top_score: float = Field(allow_inf_nan=False)
    require_citations: bool
    query_count: int = Field(ge=1)
    answerable_queries: int = Field(ge=0)
    unanswerable_queries: int = Field(ge=0)
    metrics: SelectiveCalibrationPoint
    per_query: list[SelectiveCalibrationQuery] = Field(min_length=1)


class InjectionMatch(ContractModel):
    """One heuristic injection-rule match without reproducing attack text."""

    rule_id: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_offsets(self) -> "InjectionMatch":
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        return self


class SecurityAssessment(ContractModel):
    """Deterministic prompt-injection assessment for one text."""

    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    flagged: bool
    matches: list[InjectionMatch]


class QuarantinedChunk(ContractModel):
    """A chunk withheld by corpus screening and its security assessment."""

    chunk_id: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    assessment: SecurityAssessment


class CorpusSecurityReport(ContractModel):
    """Read-only corpus screening output; safe IDs are not a safety guarantee."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    total_chunks: int = Field(ge=0)
    safe_chunk_ids: list[str]
    quarantined_chunks: list[QuarantinedChunk]


class PoisonedDocumentCase(ContractModel):
    """A retrieval attack fixture with known target and poisoned chunks."""

    case_id: str = Field(min_length=1)
    query: Query
    target_chunk_id: str = Field(min_length=1)
    poisoned_chunk_id: str = Field(min_length=1)
    chunks: list[Chunk] = Field(min_length=2, max_length=100)

    @model_validator(mode="after")
    def validate_case(self) -> "PoisonedDocumentCase":
        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Case chunk_id values must be unique")
        if self.target_chunk_id == self.poisoned_chunk_id:
            raise ValueError("Target and poisoned chunk IDs must differ")
        missing = {self.target_chunk_id, self.poisoned_chunk_id} - set(chunk_ids)
        if missing:
            raise ValueError(f"Target or poisoned chunks are missing: {sorted(missing)}")
        return self


class PoisonedRetrievalResult(ContractModel):
    """Per-case poisoned-document exposure and ranking outcome."""

    case_id: str = Field(min_length=1)
    target_rank: int | None = Field(default=None, ge=1)
    poisoned_rank: int | None = Field(default=None, ge=1)
    poisoned_top_1: bool
    poisoned_in_top_k: bool
    poisoned_outranks_target: bool
    quarantined_chunk_ids: list[str]


class PoisonedRetrievalReport(ContractModel):
    """Aggregate attack exposure rates with explicit screening configuration."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    retriever: str = Field(min_length=1)
    screened: bool
    top_1_attack_success_rate: float = Field(ge=0, le=1)
    top_k_exposure_rate: float = Field(ge=0, le=1)
    results: list[PoisonedRetrievalResult]


class BenchmarkFile(ContractModel):
    """A relative dataset file with its deterministic content identity."""

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_path(self) -> "BenchmarkFile":
        parts = self.path.split("/")
        if self.path.startswith(("/", "\\")) or "\\" in self.path:
            raise ValueError("Benchmark paths must be relative POSIX paths")
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError("Benchmark paths cannot contain empty or parent segments")
        return self


class BenchmarkManifest(ContractModel):
    """A self-contained, reproducible benchmark dataset configuration."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    dataset_id: str = Field(min_length=3)
    gold_set: str = Field(min_length=1)
    label_source: LabelSource
    retriever: Literal["bm25"] = "bm25"
    k: int = Field(ge=1, le=100)
    files: list[BenchmarkFile] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_files(self) -> "BenchmarkManifest":
        paths = [item.path for item in self.files]
        if len(set(paths)) != len(paths):
            raise ValueError("Benchmark file paths must be unique")
        required = {"documents.jsonl", "chunks.jsonl", "gold_set.json"}
        if not required.issubset(paths):
            raise ValueError("Benchmark manifest is missing required dataset files")
        return self


class BenchmarkReport(ContractModel):
    """Deterministic evaluation results bound to a hashed dataset manifest."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    manifest: BenchmarkManifest
    evaluations: list[RetrievalEvaluationReport] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_evaluations(self) -> "BenchmarkReport":
        retrievers = [item.retriever for item in self.evaluations]
        if len(set(retrievers)) != len(retrievers):
            raise ValueError("Benchmark evaluation retrievers must be unique")
        for evaluation in self.evaluations:
            if evaluation.retriever != self.manifest.retriever:
                raise ValueError("Evaluation retriever must match the benchmark manifest")
            if evaluation.k != self.manifest.k:
                raise ValueError("Evaluation cutoff must match the benchmark manifest")
            if evaluation.gold_set != self.manifest.gold_set:
                raise ValueError("Evaluation gold set must match the benchmark manifest")
            if evaluation.gold_label_source != self.manifest.label_source:
                raise ValueError("Evaluation label source must match the benchmark manifest")
        return self


class HuggingFaceDatasetManifest(ContractModel):
    """A deterministic Hugging Face dataset with content-addressed JSONL files."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    dataset_id: str = Field(min_length=3)
    gold_set: str = Field(min_length=1)
    label_source: LabelSource
    files: list[BenchmarkFile] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_files(self) -> "HuggingFaceDatasetManifest":
        paths = [item.path for item in self.files]
        if len(set(paths)) != len(paths):
            raise ValueError("Dataset file paths must be unique")
        required = {"data/documents.jsonl", "data/chunks.jsonl", "data/gold_set.json"}
        if not required.issubset(paths):
            raise ValueError("Dataset manifest is missing required data files")
        return self


class McpSearchInput(ContractModel):
    """Read-only MCP evidence search arguments."""

    text: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)

    @model_validator(mode="after")
    def validate_text(self) -> "McpSearchInput":
        if not self.text.strip():
            raise ValueError("Search text must not be blank")
        return self


class McpChunkInput(ContractModel):
    """Read-only MCP chunk lookup arguments."""

    chunk_id: str = Field(min_length=1)


class McpCitationVerificationInput(ContractModel):
    """Read-only MCP citation verification arguments."""

    answer: str = Field(min_length=1)
    response: dict[str, Any]
    support_threshold: float = Field(default=0.6, gt=0, le=1)


class AnnotationCandidate(ContractModel):
    """One query-chunk decision with optional completed human adjudication."""

    chunk: Chunk
    relevance: int | None = Field(default=None, ge=0, le=3)
    annotator: str | None = Field(default=None, min_length=1)
    notes: str | None = None

    @model_validator(mode="after")
    def validate_completion(self) -> "AnnotationCandidate":
        if self.relevance is None and self.annotator is not None:
            raise ValueError("An annotator requires a relevance decision")
        if self.relevance is not None and self.annotator is None:
            raise ValueError("A relevance decision requires an annotator")
        return self


class AnnotationTask(ContractModel):
    """One query and its rank-blind candidate decisions."""

    query: Query
    candidates: list[AnnotationCandidate] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_candidates(self) -> "AnnotationTask":
        chunk_ids = [item.chunk.chunk_id for item in self.candidates]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Annotation candidate chunk_id values must be unique")
        return self


class AnnotationBatch(ContractModel):
    """A corpus-bound batch of pending or completed human decisions."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    batch_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    label_source: Literal["human"] = "human"
    split: Literal["development", "test"]
    guidelines_version: str = Field(min_length=1)
    corpus_name: str = Field(min_length=1)
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tasks: list[AnnotationTask] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_tasks(self) -> "AnnotationBatch":
        query_ids = [item.query.query_id for item in self.tasks]
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("Annotation query_id values must be unique")
        return self
