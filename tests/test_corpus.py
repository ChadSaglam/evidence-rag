import pytest

from evidence_rag.corpus import load_documents, load_jsonl, write_jsonl
from evidence_rag.schemas import Document, Query


def test_jsonl_round_trip_preserves_unicode(tmp_path) -> None:
    path = tmp_path / "queries.jsonl"
    queries = [Query(query_id="q-tr", text="Kayıtlar kaç yıl saklanır?", language="tr")]
    write_jsonl(path, queries)
    assert load_jsonl(path, Query) == queries
    assert "Kayıtlar" in path.read_text(encoding="utf-8")


def test_invalid_jsonl_reports_line_number(tmp_path) -> None:
    path = tmp_path / "documents.jsonl"
    path.write_text("\n{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"documents\.jsonl:2"):
        load_documents(path)


def test_empty_jsonl_is_an_empty_corpus(tmp_path) -> None:
    path = tmp_path / "documents.jsonl"
    path.write_text("\n", encoding="utf-8")
    assert load_jsonl(path, Document) == []
