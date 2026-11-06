"""Retrieve each subquery and fuse evidence with reciprocal-rank fusion."""

from typing import Protocol

from evidence_rag.decomposition.heuristic import HeuristicQueryDecomposer
from evidence_rag.retrieval.base import resolve_query
from evidence_rag.schemas import (
    Chunk,
    DecomposedHit,
    Query,
    QueryDecomposition,
    RetrievalContribution,
    RetrievalHit,
)


class SubqueryRetriever(Protocol):
    def search(self, query: Query | str, top_k: int | None = None) -> list[RetrievalHit]: ...


class QueryDecomposer(Protocol):
    def decompose(self, query: Query | str) -> QueryDecomposition: ...


class DecomposedRetriever:
    """Fuse subquery rankings while retaining their individual contributions."""

    def __init__(
        self,
        retriever: SubqueryRetriever,
        decomposer: QueryDecomposer | None = None,
        *,
        rrf_k: int = 60,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")
        self.retriever = retriever
        self.decomposer = decomposer or HeuristicQueryDecomposer()
        self.rrf_k = rrf_k

    def decompose(self, query: Query | str) -> QueryDecomposition:
        return self.decomposer.decompose(query)

    def search(self, query: Query | str, top_k: int | None = None) -> list[DecomposedHit]:
        resolved = resolve_query(query, top_k)
        return self.search_decomposition(self.decompose(resolved))

    def search_decomposition(
        self, decomposition: QueryDecomposition, *, candidate_k: int | None = None
    ) -> list[DecomposedHit]:
        query = decomposition.query
        depth = max(20, query.top_k) if candidate_k is None else candidate_k
        if depth < query.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")

        chunks: dict[str, Chunk] = {}
        scores: dict[str, float] = {}
        contributions: dict[str, list[RetrievalContribution]] = {}
        for subquery in decomposition.subqueries:
            subquery_request = Query(
                query_id=subquery.subquery_id,
                text=subquery.text,
                language=query.language,
                top_k=depth,
            )
            hits = self.retriever.search(subquery_request)
            for hit in hits:
                chunk_id = hit.chunk.chunk_id
                if chunk_id in chunks and chunks[chunk_id] != hit.chunk:
                    raise ValueError(f"Conflicting chunk content for chunk_id: {chunk_id}")
                chunks[chunk_id] = hit.chunk
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (self.rrf_k + hit.rank)
                contributions.setdefault(chunk_id, []).append(
                    RetrievalContribution(
                        subquery_id=subquery.subquery_id,
                        rank=hit.rank,
                        score=hit.score,
                        retriever=hit.retriever,
                    )
                )

        order = sorted(chunks, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
        return [
            DecomposedHit(
                rank=rank,
                score=scores[chunk_id],
                retriever="decomposed-rrf",
                chunk=chunks[chunk_id],
                contributions=sorted(
                    contributions[chunk_id], key=lambda item: item.subquery_id
                ),
            )
            for rank, chunk_id in enumerate(order[: query.top_k], start=1)
        ]
