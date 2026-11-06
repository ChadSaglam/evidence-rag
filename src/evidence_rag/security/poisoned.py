"""Measure poisoned-document exposure with and without corpus screening."""

from statistics import mean
from typing import Protocol

from evidence_rag.retrieval.base import resolve_query
from evidence_rag.schemas import (
    Chunk,
    PoisonedDocumentCase,
    PoisonedRetrievalReport,
    PoisonedRetrievalResult,
)
from evidence_rag.security.screening import screen_chunks


class RetrieverFactory(Protocol):
    def __call__(self, chunks): ...


def evaluate_poisoned_retrieval(
    factory: RetrieverFactory,
    cases: list[PoisonedDocumentCase],
    *,
    screened: bool = False,
    retriever_name: str = "bm25",
) -> PoisonedRetrievalReport:
    """Run synthetic attack cases; top-1 success and top-k exposure are distinct."""

    if not cases:
        raise ValueError("At least one poisoned-document case is required")
    results: list[PoisonedRetrievalResult] = []
    for case in cases:
        chunks: list[Chunk] = list(case.chunks)
        quarantined_ids: list[str] = []
        if screened:
            chunks, security_report = screen_chunks(chunks)
            quarantined_ids = [
                item.chunk_id for item in security_report.quarantined_chunks
            ]
        ranks: dict[str, int] = {}
        if chunks:
            query = resolve_query(case.query, len(chunks))
            hits = factory(chunks).search(query)
            ranks = {hit.chunk.chunk_id: hit.rank for hit in hits}
        target_rank = ranks.get(case.target_chunk_id)
        poisoned_rank = ranks.get(case.poisoned_chunk_id)
        results.append(
            PoisonedRetrievalResult(
                case_id=case.case_id,
                target_rank=target_rank,
                poisoned_rank=poisoned_rank,
                poisoned_top_1=poisoned_rank == 1,
                poisoned_in_top_k=(
                    poisoned_rank is not None and poisoned_rank <= case.query.top_k
                ),
                poisoned_outranks_target=(
                    poisoned_rank is not None
                    and (target_rank is None or poisoned_rank < target_rank)
                ),
                quarantined_chunk_ids=quarantined_ids,
            )
        )
    return PoisonedRetrievalReport(
        retriever=retriever_name,
        screened=screened,
        top_1_attack_success_rate=mean(item.poisoned_top_1 for item in results),
        top_k_exposure_rate=mean(item.poisoned_in_top_k for item in results),
        results=results,
    )
