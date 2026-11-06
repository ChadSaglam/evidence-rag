"""Composable first-stage retrieval and cross-encoder reranking."""

from typing import Protocol

from evidence_rag.reranking.cross_encoder import CrossEncoderReranker
from evidence_rag.retrieval.base import resolve_query
from evidence_rag.schemas import Query, RerankedHit, RetrievalHit


class FirstStageRetriever(Protocol):
    def search(self, query: Query | str, top_k: int | None = None) -> list[RetrievalHit]: ...


class RerankedRetriever:
    """Retrieve a deeper candidate set, then return a smaller reranked result."""

    def __init__(self, retriever: FirstStageRetriever, reranker: CrossEncoderReranker) -> None:
        self.retriever = retriever
        self.reranker = reranker

    def search(
        self,
        query: Query | str,
        *,
        candidate_k: int | None = None,
        top_k: int | None = None,
    ) -> list[RerankedHit]:
        resolved = resolve_query(query, top_k)
        depth = max(20, resolved.top_k) if candidate_k is None else candidate_k
        if depth < resolved.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        candidate_query = resolve_query(resolved, depth)
        candidates = self.retriever.search(candidate_query)
        return self.reranker.rerank(resolved, candidates)
