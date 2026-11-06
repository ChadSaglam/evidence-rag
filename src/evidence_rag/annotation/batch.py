"""Create rank-blind annotation batches and export only complete human gold sets."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from evidence_rag.benchmark.kaggle import sha256_file, write_json
from evidence_rag.schemas import (
    AnnotationBatch,
    AnnotationCandidate,
    AnnotationTask,
    Chunk,
    GoldQuery,
    GoldSet,
    Query,
    RelevanceJudgment,
)

PLACEHOLDER_ANNOTATORS = {"fixture", "template", "todo", "unknown"}


def load_annotation_queries(path: str | Path) -> list[Query]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return TypeAdapter(list[Query]).validate_python(payload)


def load_annotation_batch(path: str | Path) -> AnnotationBatch:
    return AnnotationBatch.model_validate_json(Path(path).read_text(encoding="utf-8"))


def create_annotation_batch(
    *,
    queries: Sequence[Query],
    chunks: Sequence[Chunk],
    chunks_path: str | Path,
    batch_id: str,
    name: str,
    description: str,
    split: str,
    corpus_name: str,
    guidelines_version: str = "1.0",
) -> AnnotationBatch:
    if not queries:
        raise ValueError("At least one annotation query is required")
    if not chunks:
        raise ValueError("At least one annotation chunk is required")
    ordered_chunks = sorted(chunks, key=lambda chunk: chunk.chunk_id)
    tasks = [
        AnnotationTask(
            query=query,
            candidates=[AnnotationCandidate(chunk=chunk) for chunk in ordered_chunks],
        )
        for query in queries
    ]
    return AnnotationBatch(
        batch_id=batch_id,
        name=name,
        description=description,
        split=split,
        guidelines_version=guidelines_version,
        corpus_name=corpus_name,
        corpus_sha256=sha256_file(chunks_path),
        tasks=tasks,
    )


def summarize_annotation_batch(batch: AnnotationBatch) -> dict[str, Any]:
    candidates = [candidate for task in batch.tasks for candidate in task.candidates]
    annotated = [item for item in candidates if item.relevance is not None]
    return {
        "batch_id": batch.batch_id,
        "split": batch.split,
        "label_source": batch.label_source,
        "queries": len(batch.tasks),
        "pairs": len(candidates),
        "annotated_pairs": len(annotated),
        "pending_pairs": len(candidates) - len(annotated),
        "annotators": sorted({item.annotator for item in annotated if item.annotator}),
        "complete": len(annotated) == len(candidates),
    }


def export_human_gold_set(batch: AnnotationBatch, chunks_path: str | Path) -> GoldSet:
    if sha256_file(chunks_path) != batch.corpus_sha256:
        raise ValueError("Annotation corpus hash does not match the batch")
    summary = summarize_annotation_batch(batch)
    if not summary["complete"]:
        raise ValueError(f"Annotation batch has {summary['pending_pairs']} pending decisions")

    queries = []
    for task in batch.tasks:
        judgments = []
        for candidate in task.candidates:
            assert candidate.relevance is not None and candidate.annotator is not None
            annotator = candidate.annotator.strip()
            normalized = annotator.casefold()
            if normalized in PLACEHOLDER_ANNOTATORS or "synthetic" in normalized:
                raise ValueError(f"Annotator is not an explicit human identifier: {annotator}")
            judgments.append(
                RelevanceJudgment(
                    query_id=task.query.query_id,
                    chunk_id=candidate.chunk.chunk_id,
                    relevance=candidate.relevance,
                    annotator=annotator,
                    notes=candidate.notes,
                )
            )
        answerable = any(item.relevance > 0 for item in judgments)
        queries.append(
            GoldQuery(query=task.query, answerable=answerable, judgments=judgments)
        )
    return GoldSet(
        name=batch.name,
        description=(
            f"Human-adjudicated {batch.split} annotations for batch {batch.batch_id} "
            f"over {batch.corpus_name}."
        ),
        label_source="human",
        queries=queries,
    )


def write_annotation_batch(batch: AnnotationBatch, path: str | Path) -> None:
    write_json(Path(path), batch.model_dump(mode="json"))
