import json
from pathlib import Path

import pytest
from conftest import make_chunk

from evidence_rag.cli import main
from evidence_rag.corpus import write_jsonl
from evidence_rag.decomposition import DecomposedRetriever, HeuristicQueryDecomposer
from evidence_rag.reranking import CrossEncoderReranker, RerankedRetriever
from evidence_rag.schemas import Query, RerankedResponse, RetrievalHit


def load_cases():
    path = Path(__file__).parent / "fixtures" / "decomposition_cases.json"
    return json.loads(path.read_text(encoding="utf-8"))


def make_hit(chunk_id: str, text: str, rank: int, score: float = 1.0) -> RetrievalHit:
    return RetrievalHit(
        rank=rank,
        score=score,
        retriever="fixture-retriever",
        chunk=make_chunk(chunk_id, text),
    )


@pytest.mark.parametrize("case", load_cases(), ids=lambda case: case["case_id"])
def test_multilingual_decomposition_fixtures(case) -> None:
    query = Query(
        query_id=case["query_id"],
        text=case["text"],
        language=case["language"],
    )
    decomposition = HeuristicQueryDecomposer().decompose(query)
    assert [subquery.text for subquery in decomposition.subqueries] == case["expected"]
    assert decomposition.strategy == "heuristic-v1"
    assert decomposition.query == query


def test_decomposition_is_repeatable_and_does_not_mutate_query() -> None:
    query = Query(query_id="q", text="What is VAT? When is it due?", top_k=7)
    decomposer = HeuristicQueryDecomposer()
    first = decomposer.decompose(query)
    second = decomposer.decompose(query)
    assert first == second
    assert [item.subquery_id for item in first.subqueries] == ["q:sub-001", "q:sub-002"]
    assert query.text == "What is VAT? When is it due?"
    assert query.top_k == 7


def test_decomposition_respects_max_subqueries() -> None:
    decomposer = HeuristicQueryDecomposer(max_subqueries=2)
    result = decomposer.decompose("First clause here; second clause here; third clause here")
    assert len(result.subqueries) == 2
    assert result.subqueries[1].text == "second clause here third clause here"


@pytest.mark.parametrize("kwargs", [{"max_subqueries": 0}, {"min_clause_words": 0}])
def test_decomposer_rejects_invalid_configuration(kwargs) -> None:
    with pytest.raises(ValueError):
        HeuristicQueryDecomposer(**kwargs)


class RankingRetriever:
    def __init__(self, rankings: dict[str, list[RetrievalHit]]) -> None:
        self.rankings = rankings
        self.calls: list[Query] = []

    def search(self, query, top_k=None):
        self.calls.append(query)
        return self.rankings[query.text]


def make_fused_retriever():
    first_text = "What is the retention period?"
    second_text = "what are invoice requirements?"
    rankings = {
        first_text: [make_hit("a:0", "retention", 1), make_hit("b:0", "shared", 2)],
        second_text: [make_hit("b:0", "shared", 1), make_hit("c:0", "invoice", 2)],
    }
    retriever = RankingRetriever(rankings)
    return DecomposedRetriever(retriever), retriever


def test_subquery_rrf_fuses_scores_and_retains_contributions() -> None:
    retriever, base = make_fused_retriever()
    query = Query(
        query_id="q",
        text="What is the retention period and what are invoice requirements?",
        top_k=3,
    )
    hits = retriever.search(query)
    assert [hit.chunk.chunk_id for hit in hits] == ["b:0", "a:0", "c:0"]
    assert hits[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert hits[1].score == pytest.approx(1 / 61)
    assert [item.subquery_id for item in hits[0].contributions] == ["q:sub-001", "q:sub-002"]
    assert [call.top_k for call in base.calls] == [20, 20]
    assert retriever.search(query) == hits


def test_subquery_fusion_rejects_shallow_candidate_depth() -> None:
    retriever, _ = make_fused_retriever()
    query = Query(query_id="q", text="What is VAT? When is it due?", top_k=2)
    decomposition = retriever.decompose(query)
    with pytest.raises(ValueError, match="candidate_k"):
        retriever.search_decomposition(decomposition, candidate_k=1)


def test_subquery_fusion_rejects_conflicting_chunk_content() -> None:
    rankings = {
        "What is VAT?": [make_hit("same:0", "first", 1)],
        "When is it due?": [make_hit("same:0", "second", 1)],
    }
    retriever = DecomposedRetriever(RankingRetriever(rankings))
    with pytest.raises(ValueError, match="Conflicting chunk content"):
        retriever.search("What is VAT? When is it due?")


def test_reranking_preserves_decomposition_contributions() -> None:
    retriever, _ = make_fused_retriever()

    class StaticCrossEncoder:
        def predict(self, pairs):
            return [0.1, 0.9, 0.8]

    query = Query(
        query_id="q",
        text="What is the retention period and what are invoice requirements?",
        top_k=2,
    )
    pipeline = RerankedRetriever(retriever, CrossEncoderReranker(StaticCrossEncoder()))
    hits = pipeline.search(query, candidate_k=3)
    response = RerankedResponse(
        query=query,
        decomposition=retriever.decompose(query),
        reranker_model="fixture",
        hits=hits,
    )
    payload = response.model_dump(mode="json")
    assert payload["decomposition"]["subqueries"][0]["subquery_id"] == "q:sub-001"
    assert payload["hits"][0]["base_hit"]["contributions"]


def test_decomposed_cli_emits_subqueries_and_contributions(tmp_path, capsys) -> None:
    path = tmp_path / "chunks.jsonl"
    write_jsonl(path, [make_chunk("a:0", "alpha"), make_chunk("b:0", "beta")])
    main(
        [
            "search",
            "--chunks",
            str(path),
            "--query",
            "What is alpha and what is beta?",
            "--decompose",
            "--top-k",
            "2",
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert len(output["decomposition"]["subqueries"]) == 2
    assert output["hits"][0]["retriever"] == "decomposed-rrf"
    assert output["hits"][0]["contributions"]
