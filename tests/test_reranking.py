import json
from pathlib import Path

import numpy as np
import pytest
from conftest import make_chunk

from evidence_rag.cli import build_parser
from evidence_rag.reranking import CrossEncoderReranker, RerankedRetriever
from evidence_rag.retrieval.bm25 import BM25Retriever
from evidence_rag.schemas import Query, RerankedResponse, RetrievalHit


class MappingCrossEncoder:
    model_name = "fixture-cross-encoder"

    def __init__(self, scores: dict[tuple[str, str], float]) -> None:
        self.scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs):
        self.calls.append(list(pairs))
        return [self.scores[pair] for pair in pairs]


class StaticCrossEncoder:
    def __init__(self, scores) -> None:
        self.scores = scores
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        return self.scores


def make_hit(chunk_id: str, text: str, rank: int = 1, score: float = 1.0) -> RetrievalHit:
    return RetrievalHit(
        rank=rank,
        score=score,
        retriever="fixture-retriever",
        chunk=make_chunk(chunk_id, text),
    )


def load_fixture():
    path = Path(__file__).parent / "fixtures" / "reranking_case.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_multilingual_fixture_reranks_deterministically() -> None:
    fixture = load_fixture()
    query = Query.model_validate(fixture["query"])
    hits = [
        make_hit(item["chunk_id"], item["text"], item["base_rank"], item["base_score"])
        for item in fixture["candidates"]
    ]
    scores = {
        (query.text, item["text"]): item["cross_encoder_score"]
        for item in fixture["candidates"]
    }
    model = MappingCrossEncoder(scores)
    reranker = CrossEncoderReranker(model)

    first = reranker.rerank(query, hits)
    second = reranker.rerank(query, hits)

    assert first == second
    assert [hit.base_hit.chunk.chunk_id for hit in first] == fixture["expected_order"]
    assert first[0].base_hit == hits[1]
    assert first[0].base_hit.score == 1.8
    assert first[0].reranker == "fixture-cross-encoder"
    response = RerankedResponse(
        query=query, reranker_model=reranker.model_name, hits=first
    )
    assert response.hits[0].base_hit.chunk.source.uri == "https://example.test/doc-de:0"


def test_reranking_ties_use_base_rank_before_chunk_id() -> None:
    query = Query(query_id="q", text="query")
    hits = [make_hit("z:0", "same", rank=2), make_hit("a:0", "same", rank=1)]
    model = MappingCrossEncoder({("query", "same"): 0.5})
    ranked = CrossEncoderReranker(model).rerank(query, hits)
    assert [hit.base_hit.chunk.chunk_id for hit in ranked] == ["a:0", "z:0"]


def test_empty_candidates_do_not_call_the_model() -> None:
    model = StaticCrossEncoder([])
    assert CrossEncoderReranker(model).rerank("query", []) == []
    assert model.calls == 0


def test_duplicate_candidates_are_rejected() -> None:
    hit = make_hit("doc:0", "text")
    model = StaticCrossEncoder([0.1, 0.2])
    with pytest.raises(ValueError, match="unique"):
        CrossEncoderReranker(model).rerank("query", [hit, hit])


@pytest.mark.parametrize(
    ("scores", "message"),
    [([0.1], "one score"), ([np.nan, 0.2], "finite")],
)
def test_invalid_cross_encoder_scores_are_rejected(scores, message) -> None:
    hits = [make_hit("doc:0", "alpha"), make_hit("doc:1", "beta")]
    with pytest.raises(ValueError, match=message):
        CrossEncoderReranker(StaticCrossEncoder(scores)).rerank("query", hits)


def test_pipeline_retrieves_deep_candidates_then_reranks() -> None:
    chunks = [make_chunk("doc:0", "alpha alpha"), make_chunk("doc:1", "alpha beta")]
    base = BM25Retriever(chunks)
    model = MappingCrossEncoder(
        {
            ("alpha", "alpha alpha"): 0.1,
            ("alpha", "alpha beta"): 0.9,
        }
    )
    pipeline = RerankedRetriever(base, CrossEncoderReranker(model))
    query = Query(query_id="q", text="alpha", top_k=1)
    hits = pipeline.search(query, candidate_k=2)
    assert len(model.calls[0]) == 2
    assert hits[0].base_hit.chunk.chunk_id == "doc:1"
    assert hits[0].base_hit.rank == 2
    assert hits[0].base_hit.retriever == "bm25"


def test_pipeline_rejects_candidate_depth_below_top_k() -> None:
    hit = make_hit("doc:0", "alpha")

    class OneHitRetriever:
        def search(self, query, top_k=None):
            return [hit]

    pipeline = RerankedRetriever(OneHitRetriever(), CrossEncoderReranker(StaticCrossEncoder([])))
    with pytest.raises(ValueError, match="candidate_k"):
        pipeline.search(Query(query_id="q", text="alpha", top_k=2), candidate_k=1)


def test_cli_exposes_reranking_options() -> None:
    args = build_parser().parse_args(
        [
            "search",
            "--chunks",
            "chunks.jsonl",
            "--query",
            "query",
            "--rerank",
            "--rerank-model",
            "fixture-model",
            "--candidate-k",
            "7",
        ]
    )
    assert args.rerank is True
    assert args.rerank_model == "fixture-model"
    assert args.candidate_k == 7
