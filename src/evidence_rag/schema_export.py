"""Generate public JSON Schema artifacts from the runtime contracts."""

import json
from pathlib import Path

from pydantic import BaseModel

from evidence_rag.schemas import (
    AnnotationBatch,
    AnnotationCandidate,
    AnnotationTask,
    BenchmarkFile,
    BenchmarkManifest,
    BenchmarkReport,
    Chunk,
    CitationVerificationReport,
    Claim,
    ClaimVerification,
    CorpusSecurityReport,
    DecomposedHit,
    DecomposedResponse,
    Document,
    FrozenSelectivePolicy,
    GoldQuery,
    GoldSet,
    HuggingFaceDatasetManifest,
    InjectionMatch,
    McpChunkInput,
    McpCitationVerificationInput,
    McpSearchInput,
    PoisonedDocumentCase,
    PoisonedRetrievalReport,
    PoisonedRetrievalResult,
    QuarantinedChunk,
    Query,
    QueryDecomposition,
    QueryEvaluation,
    RelevanceJudgment,
    RerankedHit,
    RerankedResponse,
    RetrievalContribution,
    RetrievalEvaluationReport,
    RetrievalHit,
    RetrievalResponse,
    SecurityAssessment,
    SelectiveCalibrationPoint,
    SelectiveCalibrationQuery,
    SelectiveCalibrationReport,
    SelectiveDecision,
    SelectivePolicyEvaluationReport,
    Source,
    SubQuery,
)

PUBLIC_MODELS: tuple[type[BaseModel], ...] = (
    Source,
    Document,
    Chunk,
    Query,
    SubQuery,
    QueryDecomposition,
    RetrievalHit,
    RetrievalContribution,
    DecomposedHit,
    RetrievalResponse,
    DecomposedResponse,
    RerankedHit,
    RerankedResponse,
    RelevanceJudgment,
    Claim,
    ClaimVerification,
    CitationVerificationReport,
    GoldQuery,
    GoldSet,
    QueryEvaluation,
    RetrievalEvaluationReport,
    SelectiveDecision,
    SelectiveCalibrationQuery,
    SelectiveCalibrationPoint,
    SelectiveCalibrationReport,
    FrozenSelectivePolicy,
    SelectivePolicyEvaluationReport,
    InjectionMatch,
    SecurityAssessment,
    QuarantinedChunk,
    CorpusSecurityReport,
    PoisonedDocumentCase,
    PoisonedRetrievalResult,
    PoisonedRetrievalReport,
    AnnotationCandidate,
    AnnotationTask,
    AnnotationBatch,
    BenchmarkFile,
    BenchmarkManifest,
    BenchmarkReport,
    HuggingFaceDatasetManifest,
    McpSearchInput,
    McpChunkInput,
    McpCitationVerificationInput,
)


def export_schemas(output_dir: str | Path) -> list[Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for model in PUBLIC_MODELS:
        path = destination / f"{model.__name__.lower()}.schema.json"
        schema = model.model_json_schema()
        schema["$id"] = f"https://example.test/evidence-rag/schemas/{path.name}"
        path.write_text(
            json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        paths.append(path)
    return paths
