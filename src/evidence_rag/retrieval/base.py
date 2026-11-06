"""Shared validation and deterministic ranking helpers."""

from collections.abc import Sequence

import numpy as np

from evidence_rag.schemas import Chunk, Query, RetrievalHit


def validate_chunks(chunks: Sequence[Chunk]) -> tuple[Chunk, ...]:
    if not chunks:
        raise ValueError("At least one chunk is required")
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("chunk_id values must be unique")
    return tuple(chunks)


def resolve_query(query: Query | str, top_k: int | None = None) -> Query:
    if isinstance(query, str):
        return Query(query_id="adhoc", text=query, top_k=10 if top_k is None else top_k)
    if top_k is not None:
        return Query.model_validate({**query.model_dump(), "top_k": top_k})
    return query


def rank_hits(
    query: Query,
    chunks: Sequence[Chunk],
    scores: Sequence[float] | np.ndarray,
    retriever: str,
) -> list[RetrievalHit]:
    """Sort by descending score, then chunk ID to make ties reproducible."""

    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(chunks),):
        raise ValueError("Expected one score per chunk")
    if not np.all(np.isfinite(values)):
        raise ValueError("Scores must be finite")

    order = sorted(range(len(chunks)), key=lambda index: (-values[index], chunks[index].chunk_id))
    return [
        RetrievalHit(
            rank=rank,
            score=float(values[index]),
            retriever=retriever,
            chunk=chunks[index],
        )
        for rank, index in enumerate(order[: query.top_k], start=1)
    ]
