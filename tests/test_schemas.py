import json

import pytest
from pydantic import ValidationError

from evidence_rag.schema_export import PUBLIC_MODELS, export_schemas
from evidence_rag.schemas import Chunk, Query, Source


def test_chunk_rejects_invalid_offsets() -> None:
    source = Source(uri="https://example.test/doc", title="Document", language="en")
    with pytest.raises(ValidationError, match="end_char"):
        Chunk(
            chunk_id="doc:0",
            document_id="doc",
            text="text",
            start_char=4,
            end_char=4,
            source=source,
        )


def test_contract_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="unexpected"):
        Query(query_id="q", text="query", unexpected=True)


def test_query_bounds_top_k() -> None:
    with pytest.raises(ValidationError):
        Query(query_id="q", text="query", top_k=0)
    with pytest.raises(ValidationError):
        Query(query_id="q", text="query", top_k=101)


def test_exported_schemas_match_runtime_models(tmp_path) -> None:
    paths = export_schemas(tmp_path)
    assert len(paths) == len(PUBLIC_MODELS)
    for model, path in zip(PUBLIC_MODELS, paths, strict=True):
        exported = json.loads(path.read_text(encoding="utf-8"))
        assert exported["properties"] == model.model_json_schema()["properties"]
        assert exported["additionalProperties"] is False
