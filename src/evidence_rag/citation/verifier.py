"""Lexical claim-level citation verification with explicit numeric checks."""

import re
from collections.abc import Sequence

from evidence_rag.citation.extractor import CitationClaimExtractor
from evidence_rag.schemas import (
    Chunk,
    CitationVerificationReport,
    ClaimVerification,
    RerankedHit,
    RetrievalHit,
)
from evidence_rag.tokenization import tokenize

NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "how", "in", "into", "is", "it", "must", "of", "on", "or", "that", "the",
    "to", "was", "were", "what", "when", "which", "with",
    "aber", "auf", "aus", "das", "dem", "den", "der", "des", "die", "ein",
    "eine", "für", "im", "ist", "mit", "oder", "sein", "sind", "und",
    "von", "werden", "wie", "zu",
    "à", "au", "aux", "ce", "ces", "dans", "de", "du", "elle", "en",
    "est", "et", "être", "la", "le", "les", "ou", "par", "pour", "que", "qui",
    "sur", "un", "une",
    "bir", "bu", "da", "için", "ile", "ise", "mi", "mı", "mu", "mü",
    "ne", "o", "şu", "ve", "veya", "ya",
}


def content_tokens(text: str) -> set[str]:
    return {
        token
        for token in tokenize(text)
        if token not in STOPWORDS and (len(token) > 1 or token.isdigit())
    }


class ClaimLevelCitationVerifier:
    """Verify citation resolution and lexical coverage, not semantic entailment."""

    verifier_name = "lexical-overlap-v1"

    def __init__(
        self,
        *,
        support_threshold: float = 0.6,
        extractor: CitationClaimExtractor | None = None,
    ) -> None:
        if not 0 < support_threshold <= 1:
            raise ValueError("support_threshold must be greater than zero and at most one")
        self.support_threshold = float(support_threshold)
        self.extractor = extractor or CitationClaimExtractor()

    def verify(
        self, answer: str, hits: Sequence[RetrievalHit | RerankedHit]
    ) -> CitationVerificationReport:
        evidence = self._evidence_by_rank(hits)
        claims = self.extractor.extract(answer)
        verifications = [self._verify_claim(claim, evidence) for claim in claims]
        return CitationVerificationReport(
            answer=answer,
            verifier=self.verifier_name,
            support_threshold=self.support_threshold,
            fully_supported=bool(verifications)
            and all(item.status == "lexically_supported" for item in verifications),
            claims=verifications,
        )

    @staticmethod
    def _evidence_by_rank(hits: Sequence[RetrievalHit | RerankedHit]) -> dict[int, Chunk]:
        evidence: dict[int, Chunk] = {}
        for hit in hits:
            chunk = hit.base_hit.chunk if isinstance(hit, RerankedHit) else hit.chunk
            if hit.rank in evidence:
                raise ValueError(f"Duplicate evidence rank: {hit.rank}")
            evidence[hit.rank] = chunk
        return evidence

    def _verify_claim(self, claim, evidence: dict[int, Chunk]) -> ClaimVerification:
        if not claim.citation_ranks:
            return ClaimVerification(
                claim=claim,
                status="uncited",
                cited_chunk_ids=[],
                matched_tokens=[],
                missing_tokens=[],
                reason="The claim has no numeric citation.",
            )
        missing_ranks = [rank for rank in claim.citation_ranks if rank not in evidence]
        if missing_ranks:
            return ClaimVerification(
                claim=claim,
                status="invalid_citation",
                cited_chunk_ids=[],
                matched_tokens=[],
                missing_tokens=[],
                reason=f"Unknown citation ranks: {missing_ranks}.",
            )

        chunks = [evidence[rank] for rank in claim.citation_ranks]
        claim_tokens = content_tokens(claim.text)
        evidence_tokens = set().union(*(content_tokens(chunk.text) for chunk in chunks))
        matched = sorted(claim_tokens & evidence_tokens)
        missing = sorted(claim_tokens - evidence_tokens)
        score = len(matched) / len(claim_tokens) if claim_tokens else 0.0
        claim_numbers = set(NUMBER_PATTERN.findall(claim.text))
        evidence_numbers = set().union(
            *(set(NUMBER_PATTERN.findall(chunk.text)) for chunk in chunks)
        )
        missing_numbers = sorted(claim_numbers - evidence_numbers)
        supported = bool(claim_tokens) and score >= self.support_threshold and not missing_numbers
        if supported:
            status = "lexically_supported"
            reason = "Citation ranks resolve and lexical coverage meets the threshold."
        else:
            status = "insufficient_support"
            if missing_numbers:
                reason = f"Claim numbers absent from cited evidence: {missing_numbers}."
            else:
                reason = (
                    f"Lexical coverage {score:.2f} is below threshold "
                    f"{self.support_threshold:.2f}."
                )
        return ClaimVerification(
            claim=claim,
            status=status,
            support_score=score,
            cited_chunk_ids=[chunk.chunk_id for chunk in chunks],
            matched_tokens=matched,
            missing_tokens=missing,
            reason=reason,
        )
