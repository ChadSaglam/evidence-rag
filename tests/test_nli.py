import numpy as np
import pytest

from evidence_rag.citation.nli import NLICitationVerifier, evidence_windows
from evidence_rag.schemas import Chunk, RetrievalHit, Source


class KeywordNLI:
    """Deterministic stand-in: contradiction on 'never', entailment on 'ten'."""

    model_name = "fake-nli"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def predict(self, pairs):
        self.calls.extend(pairs)
        rows = []
        for _, hypothesis in pairs:
            if "never" in hypothesis:
                rows.append([0.05, 0.05, 0.90])
            elif "ten" in hypothesis or "zehn" in hypothesis:
                rows.append([0.90, 0.05, 0.05])
            else:
                rows.append([0.10, 0.80, 0.10])
        return np.asarray(rows)


def make_hit(text: str, rank: int = 1) -> RetrievalHit:
    chunk = Chunk(
        chunk_id=f"doc:{rank}",
        document_id="doc",
        text=text,
        start_char=0,
        end_char=len(text),
        source=Source(
            uri="https://example.test", title="t", language="en", retrieved_at="2026-01-01"
        ),
    )
    return RetrievalHit(rank=rank, score=1.0, retriever="fixture", chunk=chunk)


EVIDENCE = make_hit("Accounting records must be retained for ten years.")


def test_entailed_paraphrase_is_semantically_supported() -> None:
    report = NLICitationVerifier(KeywordNLI()).verify("Books are kept ten years [1].", [EVIDENCE])
    claim = report.claims[0]
    assert claim.status == "semantically_supported"
    assert claim.entailment_score == pytest.approx(0.9)
    assert report.fully_supported is True


def test_negation_with_full_lexical_overlap_is_contradicted() -> None:
    answer = "Accounting records must never be retained for ten years [1]."
    claim = NLICitationVerifier(KeywordNLI()).verify(answer, [EVIDENCE]).claims[0]
    assert claim.status == "contradicted"
    assert claim.contradiction_score == pytest.approx(0.9)


def test_neutral_claim_is_insufficient() -> None:
    report = NLICitationVerifier(KeywordNLI()).verify("Invoices need a date [1].", [EVIDENCE])
    assert report.claims[0].status == "insufficient_support"
    assert report.fully_supported is False


def test_numeric_guard_still_applies() -> None:
    claim = NLICitationVerifier(KeywordNLI()).verify(
        "Records are kept ten years and 20 days [1].", [EVIDENCE]
    ).claims[0]
    assert claim.status == "insufficient_support"
    assert "20" in claim.reason


def test_citation_errors_bypass_the_model() -> None:
    model = KeywordNLI()
    verifier = NLICitationVerifier(model)
    assert verifier.verify("Kept ten years.", [EVIDENCE]).claims[0].status == "uncited"
    assert verifier.verify("Kept ten years [4].", [EVIDENCE]).claims[0].status == "invalid_citation"
    assert model.calls == []


def test_long_evidence_is_windowed() -> None:
    text = "One. Two. Three. Four. Five."
    assert evidence_windows(text, 3) == [
        "One. Two. Three.",
        "Two. Three. Four.",
        "Three. Four. Five.",
    ]
    assert evidence_windows("Short.", 3) == ["Short."]


def test_rejects_malformed_model_output() -> None:
    class Broken:
        def predict(self, pairs):
            return np.zeros((len(pairs), 2))

    with pytest.raises(ValueError):
        NLICitationVerifier(Broken()).verify("Kept ten years [1].", [EVIDENCE])
