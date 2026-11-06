"""Read-only corpus screening with explicit quarantine records."""

from collections.abc import Sequence

from evidence_rag.schemas import Chunk, CorpusSecurityReport, QuarantinedChunk
from evidence_rag.security.injection import InjectionDetector


def screen_chunks(
    chunks: Sequence[Chunk], detector: InjectionDetector | None = None
) -> tuple[list[Chunk], CorpusSecurityReport]:
    """Return original safe chunks and a report; never mutate the input corpus."""

    detector = detector or InjectionDetector()
    safe: list[Chunk] = []
    quarantined: list[QuarantinedChunk] = []
    for chunk in chunks:
        assessment = detector.assess(chunk.text)
        if assessment.flagged:
            quarantined.append(
                QuarantinedChunk(
                    chunk_id=chunk.chunk_id,
                    source_uri=chunk.source.uri,
                    assessment=assessment,
                )
            )
        else:
            safe.append(chunk)
    report = CorpusSecurityReport(
        total_chunks=len(chunks),
        safe_chunk_ids=[chunk.chunk_id for chunk in safe],
        quarantined_chunks=quarantined,
    )
    return safe, report
