"""Deterministic selective answering based on evidence and citation checks."""

from collections.abc import Sequence

from evidence_rag.schemas import (
    CitationVerificationReport,
    RerankedHit,
    RetrievalHit,
    SelectiveDecision,
)


class SelectiveAnswerer:
    """Abstain unless retrieval and optional citation gates both pass."""

    def __init__(self, *, min_top_score: float, require_citations: bool = True) -> None:
        self.min_top_score = float(min_top_score)
        self.require_citations = require_citations

    def decide(
        self,
        hits: Sequence[RetrievalHit | RerankedHit],
        citation_report: CitationVerificationReport | None = None,
    ) -> SelectiveDecision:
        if not hits:
            return self._decision(
                "abstain", "no_evidence", citation_report=citation_report
            )
        top = min(hits, key=lambda hit: (hit.rank, self._chunk_id(hit)))
        base = self._decision(
            "answer",
            "evidence_and_citations_passed",
            top_score=top.score,
            top_chunk_id=self._chunk_id(top),
            citation_report=citation_report,
        )
        if top.score < self.min_top_score:
            base.action = "abstain"
            base.reason = "low_retrieval_score"
            return base
        if self.require_citations and citation_report is None:
            base.action = "abstain"
            base.reason = "citation_verification_required"
            return base
        if self.require_citations and not citation_report.fully_supported:
            base.action = "abstain"
            base.reason = "citation_verification_failed"
            return base
        if not self.require_citations:
            base.reason = "retrieval_score_passed"
        return base

    def _decision(
        self,
        action: str,
        reason: str,
        *,
        top_score: float | None = None,
        top_chunk_id: str | None = None,
        citation_report: CitationVerificationReport | None,
    ) -> SelectiveDecision:
        return SelectiveDecision(
            action=action,
            reason=reason,
            top_score=top_score,
            min_top_score=self.min_top_score,
            top_chunk_id=top_chunk_id,
            citation_fully_supported=(
                citation_report.fully_supported if citation_report is not None else None
            ),
        )

    @staticmethod
    def _chunk_id(hit: RetrievalHit | RerankedHit) -> str:
        return hit.base_hit.chunk.chunk_id if isinstance(hit, RerankedHit) else hit.chunk.chunk_id
