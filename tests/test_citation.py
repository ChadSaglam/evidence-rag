import json
from pathlib import Path

import pytest
from conftest import make_chunk

from evidence_rag.citation import CitationClaimExtractor, ClaimLevelCitationVerifier
from evidence_rag.cli import main
from evidence_rag.schemas import Query, RerankedHit, RetrievalHit, RetrievalResponse


def load_fixture():
    path = Path(__file__).parent / "fixtures" / "citation_cases.json"
    return json.loads(path.read_text(encoding="utf-8"))


def make_hit(chunk_id: str, text: str, rank: int) -> RetrievalHit:
    return RetrievalHit(
        rank=rank,
        score=1.0,
        retriever="fixture-retriever",
        chunk=make_chunk(chunk_id, text),
    )


def fixture_hits():
    return [
        make_hit(item["chunk_id"], item["text"], item["rank"])
        for item in load_fixture()["evidence"]
    ]


@pytest.mark.parametrize(
    "case", load_fixture()["cases"], ids=lambda case: case["case_id"]
)
def test_citation_verification_fixtures(case) -> None:
    hits = fixture_hits()
    verifier = ClaimLevelCitationVerifier()
    report = verifier.verify(case["answer"], hits)
    assert report == verifier.verify(case["answer"], hits)
    assert len(report.claims) == 1
    result = report.claims[0]
    assert result.status == case["expected_status"]
    assert result.support_score == case["expected_score"]
    assert result.cited_chunk_ids == case["expected_chunk_ids"]
    assert report.fully_supported == (case["expected_status"] == "lexically_supported")


def test_claim_extraction_preserves_offsets_and_citation_lists() -> None:
    answer = "First claim [1, 2].\nSecond line without citation"
    claims = CitationClaimExtractor().extract(answer)
    assert len(claims) == 2
    assert claims[0].text == "First claim."
    assert claims[0].citation_ranks == [1, 2]
    assert claims[1].citation_ranks == []
    for claim in claims:
        assert answer[claim.start_char : claim.end_char].strip()


def test_decimal_numbers_do_not_split_claims() -> None:
    hits = [make_hit("rate:0", "The rate is 10.5 percent.", 1)]
    report = ClaimLevelCitationVerifier().verify("The rate is 10.5 percent [1].", hits)
    assert len(report.claims) == 1
    assert report.claims[0].status == "lexically_supported"


def test_reranked_citations_use_outer_rank_and_base_chunk() -> None:
    base = make_hit("doc:0", "Accounting records are retained for 10 years.", rank=7)
    reranked = RerankedHit(rank=1, score=0.9, reranker="fixture", base_hit=base)
    report = ClaimLevelCitationVerifier().verify(
        "Accounting records are retained for 10 years [1].", [reranked]
    )
    assert report.claims[0].status == "lexically_supported"
    assert report.claims[0].cited_chunk_ids == ["doc:0"]


def test_duplicate_evidence_ranks_are_rejected() -> None:
    hits = [make_hit("a:0", "alpha", 1), make_hit("b:0", "beta", 1)]
    with pytest.raises(ValueError, match="Duplicate evidence rank"):
        ClaimLevelCitationVerifier().verify("Alpha [1].", hits)


def test_empty_answer_is_not_fully_supported() -> None:
    report = ClaimLevelCitationVerifier().verify("", fixture_hits())
    assert report.claims == []
    assert report.fully_supported is False


@pytest.mark.parametrize("threshold", [0, 1.1])
def test_invalid_support_threshold_is_rejected(threshold) -> None:
    with pytest.raises(ValueError, match="support_threshold"):
        ClaimLevelCitationVerifier(support_threshold=threshold)


def test_verification_cli_reads_retrieval_response(tmp_path, capsys) -> None:
    response = RetrievalResponse(
        query=Query(query_id="q", text="retention"),
        hits=fixture_hits(),
    )
    path = tmp_path / "response.json"
    path.write_text(response.model_dump_json(), encoding="utf-8")
    main(
        [
            "verify",
            "--response",
            str(path),
            "--answer",
            "Accounting records must be retained for 10 years [1].",
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert output["verifier"] == "lexical-overlap-v1"
    assert output["fully_supported"] is True
    assert output["claims"][0]["status"] == "lexically_supported"
