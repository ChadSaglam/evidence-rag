"""Multilingual cross-encoder reranking over first-stage retrieval hits."""

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from evidence_rag.retrieval.base import resolve_query
from evidence_rag.schemas import Query, RerankedHit, RetrievalHit

DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class CrossEncoderModel(Protocol):
    """Minimal model interface that also supports deterministic test doubles."""

    def predict(self, pairs: Sequence[tuple[str, str]]) -> Any: ...


class SentenceTransformerCrossEncoder:
    """Lazy SentenceTransformers adapter with progress output disabled."""

    def __init__(
        self, model_name: str = DEFAULT_RERANKER_MODEL, device: str | None = None
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise ImportError(
                "Reranking requires the optional dependency: pip install -e '.[rerank]'"
            ) from error
        self.model_name = model_name
        self._model = CrossEncoder(model_name, device=device)

    def predict(self, pairs: Sequence[tuple[str, str]]) -> Any:
        return self._model.predict(list(pairs), show_progress_bar=False)


class CrossEncoderReranker:
    """Rerank candidate hits without discarding first-stage evidence."""

    def __init__(
        self,
        model: CrossEncoderModel | None = None,
        *,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str | None = None,
    ) -> None:
        self.model = model or SentenceTransformerCrossEncoder(model_name, device=device)
        self.model_name = getattr(model, "model_name", model_name)

    def rerank(
        self,
        query: Query | str,
        hits: Sequence[RetrievalHit],
        top_k: int | None = None,
    ) -> list[RerankedHit]:
        resolved = resolve_query(query, top_k)
        if not hits:
            return []
        chunk_ids = [hit.chunk.chunk_id for hit in hits]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Candidate chunk_id values must be unique")

        pairs = [(resolved.text, hit.chunk.text) for hit in hits]
        scores = np.asarray(self.model.predict(pairs), dtype=np.float64)
        if scores.shape != (len(hits),):
            raise ValueError("Cross-encoder must return one score per candidate")
        if not np.all(np.isfinite(scores)):
            raise ValueError("Cross-encoder scores must be finite")

        order = sorted(
            range(len(hits)),
            key=lambda index: (-scores[index], hits[index].rank, hits[index].chunk.chunk_id),
        )
        return [
            RerankedHit(
                rank=rank,
                score=float(scores[index]),
                reranker=self.model_name,
                base_hit=hits[index],
            )
            for rank, index in enumerate(order[: resolved.top_k], start=1)
        ]
