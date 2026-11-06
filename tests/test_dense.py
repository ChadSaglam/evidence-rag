import sys
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import MappingEncoder, make_chunk

from evidence_rag.retrieval.dense import DenseRetriever, E5Encoder
from evidence_rag.schemas import Query


def make_dense_retriever(chunks, query_vector=(2.0, 0.0)):
    vectors = {
        "passage: alpha": [2.0, 0.0],
        "passage: beta": [0.0, 3.0],
        "passage: gamma": [1.0, 1.0],
        "query: find alpha": list(query_vector),
    }
    encoder = MappingEncoder(vectors)
    return DenseRetriever(chunks, encoder), encoder


def test_dense_uses_e5_prefixes_and_normalized_cosine() -> None:
    chunks = [make_chunk("doc:0", "alpha"), make_chunk("doc:1", "beta")]
    retriever, encoder = make_dense_retriever(chunks)
    hits = retriever.search("find alpha", top_k=2)
    assert encoder.calls == [
        ["passage: alpha", "passage: beta"],
        ["query: find alpha"],
    ]
    assert [hit.chunk.chunk_id for hit in hits] == ["doc:0", "doc:1"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(0.0)
    assert np.allclose(np.linalg.norm(retriever._vectors, axis=1), 1.0)


def test_dense_ties_are_sorted_by_chunk_id() -> None:
    chunks = [make_chunk("z:0", "alpha"), make_chunk("a:0", "alpha")]
    retriever, _ = make_dense_retriever(chunks)
    hits = retriever.search("find alpha")
    assert [hit.chunk.chunk_id for hit in hits] == ["a:0", "z:0"]


def test_dense_results_are_repeatable() -> None:
    chunks = [make_chunk("doc:0", "alpha"), make_chunk("doc:1", "beta")]
    retriever, _ = make_dense_retriever(chunks)
    query = Query(query_id="q", text="find alpha", top_k=2)
    assert retriever.search(query) == retriever.search(query)


def test_dense_rejects_zero_and_non_finite_vectors() -> None:
    chunk = make_chunk("doc:0", "alpha")
    zero = MappingEncoder({"passage: alpha": [0.0, 0.0]})
    with pytest.raises(ValueError, match="zero vector"):
        DenseRetriever([chunk], zero)
    infinite = MappingEncoder({"passage: alpha": [np.inf, 1.0]})
    with pytest.raises(ValueError, match="non-finite"):
        DenseRetriever([chunk], infinite)


def test_dense_rejects_wrong_vector_count() -> None:
    class ShortEncoder:
        def encode(self, texts, *, normalize_embeddings=True):
            return np.asarray([[1.0, 0.0]])

    chunks = [make_chunk("doc:0", "alpha"), make_chunk("doc:1", "beta")]
    with pytest.raises(ValueError, match="two-dimensional"):
        DenseRetriever(chunks, ShortEncoder())


def test_e5_encoder_pins_model_revision(monkeypatch) -> None:
    captured = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name, device=None, revision=None):
            captured.update(model_name=model_name, device=device, revision=revision)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    encoder = E5Encoder("fixture-model", device="cpu", revision="revision-123")
    assert captured == {
        "model_name": "fixture-model",
        "device": "cpu",
        "revision": "revision-123",
    }
    assert encoder.model_revision == "revision-123"
