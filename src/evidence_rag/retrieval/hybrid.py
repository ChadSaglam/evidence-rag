"""Reciprocal-rank fusion for the lexical and dense baselines."""

import numpy as np

from evidence_rag.retrieval.base import rank_hits, resolve_query
from evidence_rag.retrieval.bm25 import BM25Retriever
from evidence_rag.retrieval.dense import DenseRetriever
from evidence_rag.schemas import Query, RetrievalHit


class HybridRetriever:
    """Combine complete BM25 and dense rankings without score calibration."""

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        *,
        rrf_k: int = 60,
        lexical_weight: float = 1.0,
        dense_weight: float = 1.0,
    ) -> None:
        if bm25.chunks != dense.chunks:
            raise ValueError("BM25 and dense retrievers must contain identical chunks")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")
        if lexical_weight < 0 or dense_weight < 0 or lexical_weight + dense_weight == 0:
            raise ValueError("Weights must be non-negative and at least one must be positive")
        self.chunks = bm25.chunks
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k
        self.lexical_weight = float(lexical_weight)
        self.dense_weight = float(dense_weight)

    def search(self, query: Query | str, top_k: int | None = None) -> list[RetrievalHit]:
        resolved = resolve_query(query, top_k)
        scores = np.zeros(len(self.chunks), dtype=np.float64)
        lexical = self.bm25.get_scores(resolved)
        dense = self.dense.get_scores(resolved)
        # Rank complete score vectors directly: Query.top_k is capped at 100, but fusion
        # needs every chunk's rank. Chunks with zero BM25 score receive no lexical credit.
        for weight, values, positive_only in (
            (self.lexical_weight, lexical, True),
            (self.dense_weight, dense, False),
        ):
            order = sorted(range(len(values)), key=lambda i: (-values[i], self.chunks[i].chunk_id))
            for rank, index in enumerate(order, start=1):
                if positive_only and values[index] <= 0.0:
                    continue
                scores[index] += weight / (self.rrf_k + rank)
        return rank_hits(resolved, self.chunks, scores, "hybrid-rrf")
