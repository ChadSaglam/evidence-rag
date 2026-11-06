"""Deterministic answer-sentence and numeric citation extraction."""

import re

from evidence_rag.schemas import Claim

CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
SENTENCE_PATTERN = re.compile(r".+?(?:[.!?]+(?=\s|$)|\n+|$)")


class CitationClaimExtractor:
    """Treat each answer sentence or non-empty line as one claim."""

    def extract(self, answer: str) -> list[Claim]:
        claims: list[Claim] = []
        for match in SENTENCE_PATTERN.finditer(answer):
            raw = match.group(0)
            segment = raw.strip()
            if not segment:
                continue
            start = match.start() + len(raw) - len(raw.lstrip())
            end = match.start() + len(raw.rstrip())
            ranks = sorted(
                {
                    int(rank)
                    for group in CITATION_PATTERN.findall(segment)
                    for rank in group.split(",")
                }
            )
            text = CITATION_PATTERN.sub("", segment)
            text = re.sub(r"\s+([.!?])", r"\1", text)
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            claims.append(
                Claim(
                    claim_id=f"claim-{len(claims) + 1:03d}",
                    text=text,
                    start_char=start,
                    end_char=end,
                    citation_ranks=ranks,
                )
            )
        return claims
