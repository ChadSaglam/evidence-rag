import math

import numpy as np
import pytest
from conftest import make_chunk

from evidence_rag.retrieval.bm25 import BM25Retriever
from evidence_rag.schemas import Query
from evidence_rag.tokenization import tokenize


def test_bm25_ranks_matching_chunk_first(lexical_chunks) -> None:
    retriever = BM25Retriever(lexical_chunks)
    first = retriever.search("gamma", top_k=1)[0]
    assert first.chunk.chunk_id == "doc-a:0"
    assert first.score > 0
    assert first.retriever == "bm25"


def test_bm25_matches_documented_okapi_formula(lexical_chunks) -> None:
    retriever = BM25Retriever(lexical_chunks, k1=1.5, b=0.75)
    scores = retriever.get_scores("alpha")
    # N=3, df=2, doc 0: tf=2, dl=3, avgdl=3.
    idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    expected = idf * 2 * 2.5 / (2 + 1.5 * (0.25 + 0.75 * 3 / 3))
    assert scores[0] == pytest.approx(expected)
    assert scores[2] == 0


def test_bm25_omits_zero_score_chunks_by_default(lexical_chunks) -> None:
    assert BM25Retriever(lexical_chunks).search("missing") == []


def test_bm25_uses_chunk_id_for_zero_score_ties(lexical_chunks) -> None:
    hits = BM25Retriever(lexical_chunks).search("missing", include_zero=True)
    assert [hit.chunk.chunk_id for hit in hits] == ["doc-a:0", "doc-b:0", "doc-c:0"]
    assert [hit.rank for hit in hits] == [1, 2, 3]


def test_bm25_is_repeatable_and_does_not_mutate_query(lexical_chunks) -> None:
    retriever = BM25Retriever(lexical_chunks)
    query = Query(query_id="q", text="alpha beta", top_k=2)
    first = retriever.search(query)
    second = retriever.search(query)
    assert first == second
    assert query.top_k == 2


def test_unicode_tokenization_case_folds_without_stop_words() -> None:
    assert tokenize("İKİ Geschäftsbücher and l'année") == [
        "i̇ki̇",
        "geschäftsbücher",
        "and",
        "l'année",
    ]


@pytest.mark.parametrize("query", ["alpha", Query(query_id="q", text="alpha")])
def test_search_validates_top_k_override(lexical_chunks, query) -> None:
    with pytest.raises(ValueError, match="top_k"):
        BM25Retriever(lexical_chunks).search(query, top_k=0)


def test_empty_or_duplicate_chunks_are_rejected() -> None:
    with pytest.raises(ValueError, match="At least one"):
        BM25Retriever([])
    chunk = make_chunk("doc:0", "text")
    with pytest.raises(ValueError, match="unique"):
        BM25Retriever([chunk, chunk])


def test_all_zero_length_tokenized_chunks_return_zero_scores() -> None:
    chunks = [make_chunk("doc:0", "!!!"), make_chunk("doc:1", "???")]
    scores = BM25Retriever(chunks).get_scores("anything")
    assert np.array_equal(scores, np.zeros(2))
