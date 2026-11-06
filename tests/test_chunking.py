import pytest

from evidence_rag.chunking import chunk_document, chunk_documents
from evidence_rag.schemas import Document, Source


def make_document(document_id: str = "doc", text: str = "abcdefghij") -> Document:
    return Document(
        document_id=document_id,
        text=text,
        source=Source(uri="https://example.test/doc", title="Document", language="en"),
    )


def test_chunk_offsets_reconstruct_original_text() -> None:
    document = make_document(text="abcdefghij")
    chunks = chunk_document(document, size=4, overlap=1)
    assert [chunk.text for chunk in chunks] == ["abcd", "defg", "ghij"]
    assert [chunk.chunk_id for chunk in chunks] == [
        "doc:000000",
        "doc:000001",
        "doc:000002",
    ]
    for chunk in chunks:
        assert document.text[chunk.start_char : chunk.end_char] == chunk.text


def test_chunking_rejects_invalid_overlap_and_duplicate_documents() -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_document(make_document(), size=10, overlap=10)
    with pytest.raises(ValueError, match="Duplicate document_id"):
        chunk_documents([make_document(), make_document()])
