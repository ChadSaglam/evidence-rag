"""Deterministic character chunking with lossless source offsets."""

from collections.abc import Iterable

from evidence_rag.schemas import Chunk, Document


def chunk_document(document: Document, size: int = 900, overlap: int = 150) -> list[Chunk]:
    """Split one document without altering the text covered by its offsets."""

    if size <= 0:
        raise ValueError("size must be greater than zero")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than size")

    chunks: list[Chunk] = []
    step = size - overlap
    for index, start in enumerate(range(0, len(document.text), step)):
        end = min(start + size, len(document.text))
        text = document.text[start:end]
        if text.strip():
            chunks.append(
                Chunk(
                    chunk_id=f"{document.document_id}:{index:06d}",
                    document_id=document.document_id,
                    text=text,
                    start_char=start,
                    end_char=end,
                    source=document.source,
                    metadata=document.metadata,
                )
            )
        if end == len(document.text):
            break
    return chunks


def chunk_documents(
    documents: Iterable[Document], size: int = 900, overlap: int = 150
) -> list[Chunk]:
    """Chunk documents in input order and reject duplicate document IDs."""

    chunks: list[Chunk] = []
    seen: set[str] = set()
    for document in documents:
        if document.document_id in seen:
            raise ValueError(f"Duplicate document_id: {document.document_id}")
        seen.add(document.document_id)
        chunks.extend(chunk_document(document, size=size, overlap=overlap))
    return chunks
