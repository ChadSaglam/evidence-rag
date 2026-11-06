"""Multilingual dense retrieval with an injectable encoder for offline tests."""

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from evidence_rag.retrieval.base import rank_hits, resolve_query, validate_chunks
from evidence_rag.schemas import Chunk, Query, RetrievalHit

DEFAULT_DENSE_MODEL = "intfloat/multilingual-e5-small"


class Encoder(Protocol):
    """Minimal encoder interface implemented by E5Encoder and test doubles."""

    def encode(self, texts: Sequence[str], *, normalize_embeddings: bool = True) -> Any: ...


class E5Encoder:
    """Lazy SentenceTransformers adapter; importing this module never downloads a model."""

    def __init__(
        self,
        model_name: str = DEFAULT_DENSE_MODEL,
        device: str | None = None,
        revision: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError(
                "Dense retrieval requires the optional dependency: pip install -e '.[dense]'"
            ) from error
        self.model_name = model_name
        self.model_revision = revision
        self._model = SentenceTransformer(model_name, device=device, revision=revision)

    def encode(self, texts: Sequence[str], *, normalize_embeddings: bool = True) -> Any:
        return self._model.encode(
            list(texts),
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False,
        )


class DenseRetriever:
    """Cosine retrieval over normalized passage embeddings."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        encoder: Encoder | None = None,
        *,
        model_name: str = DEFAULT_DENSE_MODEL,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
    ) -> None:
        self.chunks = validate_chunks(chunks)
        self.encoder = encoder or E5Encoder(model_name)
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        passages = [f"{passage_prefix}{chunk.text}" for chunk in self.chunks]
        self._vectors = self._encode(passages, expected_rows=len(chunks))

    def _encode(self, texts: Sequence[str], *, expected_rows: int) -> np.ndarray:
        vectors = np.asarray(
            self.encoder.encode(texts, normalize_embeddings=True), dtype=np.float64
        )
        if vectors.ndim != 2 or vectors.shape[0] != expected_rows:
            raise ValueError("Encoder must return a two-dimensional vector per input text")
        if not np.all(np.isfinite(vectors)):
            raise ValueError("Encoder returned non-finite values")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Encoder returned a zero vector")
        return vectors / norms

    def get_scores(self, query: Query | str) -> np.ndarray:
        resolved = resolve_query(query)
        query_vector = self._encode([f"{self.query_prefix}{resolved.text}"], expected_rows=1)[0]
        return self._vectors @ query_vector

    def search(self, query: Query | str, top_k: int | None = None) -> list[RetrievalHit]:
        resolved = resolve_query(query, top_k)
        return rank_hits(resolved, self.chunks, self.get_scores(resolved), "dense")
