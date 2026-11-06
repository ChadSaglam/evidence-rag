import pytest
from conftest import MappingEncoder, make_chunk

from evidence_rag.retrieval.bm25 import BM25Retriever
from evidence_rag.retrieval.dense import DenseRetriever
from evidence_rag.retrieval.hybrid import HybridRetriever


def test_hybrid_uses_reciprocal_rank_fusion() -> None:
    chunks = [make_chunk("doc:0", "alpha"), make_chunk("doc:1", "beta")]
    encoder = MappingEncoder(
        {
            "passage: alpha": [0.0, 1.0],
            "passage: beta": [1.0, 0.0],
            "query: alpha": [1.0, 0.0],
        }
    )
    hybrid = HybridRetriever(BM25Retriever(chunks), DenseRetriever(chunks, encoder), rrf_k=60)
    hits = hybrid.search("alpha")
    by_id = {hit.chunk.chunk_id: hit for hit in hits}
    assert by_id["doc:0"].score == pytest.approx(1 / 61 + 1 / 62)
    # doc:1 shares no term with the query, so BM25 contributes nothing to it.
    assert by_id["doc:1"].score == pytest.approx(1 / 61)
    assert hits[0].retriever == "hybrid-rrf"


def test_hybrid_requires_identical_chunks() -> None:
    first = make_chunk("doc:0", "alpha")
    second = make_chunk("doc:1", "beta")
    encoder = MappingEncoder({"passage: beta": [1.0, 0.0]})
    with pytest.raises(ValueError, match="identical"):
        HybridRetriever(BM25Retriever([first]), DenseRetriever([second], encoder))


def test_hybrid_handles_corpora_larger_than_query_top_k_cap() -> None:
    chunks = [make_chunk(f"doc:{i:03d}", "alpha" if i == 7 else f"filler{i}") for i in range(150)]

    class ConstantEncoder:
        def encode(self, texts, *, normalize_embeddings=True):
            return [[1.0, 0.0] for _ in texts]

    hybrid = HybridRetriever(BM25Retriever(chunks), DenseRetriever(chunks, ConstantEncoder()))
    hits = hybrid.search("alpha", top_k=5)
    assert len(hits) == 5
    assert hits[0].chunk.chunk_id == "doc:007"
