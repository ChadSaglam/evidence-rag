from conftest import make_chunk

from evidence_rag.citation import ClaimLevelCitationVerifier
from evidence_rag.evaluation import SelectiveAnswerer
from evidence_rag.schemas import RetrievalHit


def make_hit(text="alpha", score=2.0):
    return RetrievalHit(
        rank=1,
        score=score,
        retriever="fixture",
        chunk=make_chunk("doc:0", text),
    )


def test_selective_answerer_abstains_without_evidence_or_sufficient_score() -> None:
    answerer = SelectiveAnswerer(min_top_score=1.0)
    assert answerer.decide([]).reason == "no_evidence"
    low = answerer.decide([make_hit(score=0.5)])
    assert low.action == "abstain"
    assert low.reason == "low_retrieval_score"


def test_selective_answerer_requires_and_checks_citations() -> None:
    hits = [make_hit()]
    answerer = SelectiveAnswerer(min_top_score=1.0)
    assert answerer.decide(hits).reason == "citation_verification_required"
    failed_report = ClaimLevelCitationVerifier().verify("beta [1].", hits)
    assert answerer.decide(hits, failed_report).reason == "citation_verification_failed"
    passed_report = ClaimLevelCitationVerifier().verify("alpha [1].", hits)
    decision = answerer.decide(hits, passed_report)
    assert decision.action == "answer"
    assert decision.reason == "evidence_and_citations_passed"
    assert decision.top_chunk_id == "doc:0"
    assert decision.citation_fully_supported is True


def test_selective_answerer_allows_explicit_pre_generation_mode() -> None:
    decision = SelectiveAnswerer(
        min_top_score=1.0, require_citations=False
    ).decide([make_hit()])
    assert decision.action == "answer"
    assert decision.reason == "retrieval_score_passed"
