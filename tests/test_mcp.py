import asyncio

import pytest
from pydantic import ValidationError

from evidence_rag.chunking import chunk_documents
from evidence_rag.corpus import load_documents
from evidence_rag.mcp import READ_ONLY_ANNOTATIONS, ReadOnlyEvidenceService
from evidence_rag.mcp.server import create_mcp_server


@pytest.fixture
def service():
    chunks = chunk_documents(load_documents("data/sample/documents.jsonl"))
    return ReadOnlyEvidenceService(chunks)


def test_tool_contracts_are_allowlisted_and_read_only(service) -> None:
    contracts = service.tool_contracts()
    assert [item["name"] for item in contracts] == [
        "search_evidence",
        "get_evidence_chunk",
        "verify_citations",
    ]
    for contract in contracts:
        assert contract["annotations"] == READ_ONLY_ANNOTATIONS
        assert contract["annotations"]["readOnlyHint"] is True
        assert contract["annotations"]["destructiveHint"] is False
        assert contract["annotations"]["openWorldHint"] is False
        assert contract["inputSchema"]["additionalProperties"] is False
    assert contracts[0]["inputSchema"]["required"] == ["text"]


def test_search_tool_is_deterministic_and_attributed(service) -> None:
    arguments = {"text": "How long must accounting records be retained?", "top_k": 3}
    first = service.call_tool("search_evidence", arguments)
    second = service.call_tool("search_evidence", arguments)
    assert first == second
    assert first["query"]["query_id"] == "mcp"
    assert first["hits"][0]["chunk"]["chunk_id"] == "en-retention:000000"
    assert first["hits"][0]["chunk"]["source"]["uri"] == "https://example.test/accounting/en"


def test_chunk_tool_returns_exact_chunk_and_rejects_unknown_id(service) -> None:
    chunk = service.call_tool("get_evidence_chunk", {"chunk_id": "en-retention:000000"})
    document = load_documents("data/sample/documents.jsonl")[0]
    assert document.text[chunk["start_char"] : chunk["end_char"]] == chunk["text"]
    with pytest.raises(ValueError, match="Unknown chunk_id"):
        service.call_tool("get_evidence_chunk", {"chunk_id": "missing"})


def test_citation_tool_uses_explicit_response(service) -> None:
    response = service.call_tool("search_evidence", {"text": "accounting records", "top_k": 1})
    report = service.call_tool(
        "verify_citations",
        {
            "answer": "Accounting records must be retained for ten years [1].",
            "response": response,
        },
    )
    assert report["fully_supported"] is True
    assert report["claims"][0]["status"] == "lexically_supported"
    assert report["claims"][0]["cited_chunk_ids"] == ["en-retention:000000"]


def test_citation_tool_rejects_unknown_displayed_rank(service) -> None:
    response = service.call_tool("search_evidence", {"text": "accounting records", "top_k": 1})
    report = service.call_tool(
        "verify_citations",
        {"answer": "Accounting records are covered [9].", "response": response},
    )
    assert report["fully_supported"] is False
    assert report["claims"][0]["status"] == "invalid_citation"


def test_unknown_and_mutating_tools_are_not_dispatchable(service) -> None:
    for name in ("write_document", "delete_chunk", "execute_command", "unknown"):
        with pytest.raises(ValueError, match="Unknown read-only tool"):
            service.call_tool(name, {})


def test_tool_inputs_reject_blank_extra_and_invalid_values(service) -> None:
    with pytest.raises(ValidationError):
        service.call_tool("search_evidence", {"text": "  "})
    with pytest.raises(ValidationError):
        service.call_tool("search_evidence", {"text": "records", "top_k": 0})
    with pytest.raises(ValidationError, match="Extra inputs"):
        service.call_tool("get_evidence_chunk", {"chunk_id": "x", "write": True})


def test_tool_calls_do_not_mutate_corpus(service) -> None:
    before = [chunk.model_dump(mode="json") for chunk in service.chunks]
    response = service.call_tool("search_evidence", {"text": "accounting records", "top_k": 2})
    service.call_tool("get_evidence_chunk", {"chunk_id": "en-retention:000000"})
    service.call_tool(
        "verify_citations",
        {"answer": "Accounting records must be retained for ten years [1].", "response": response},
    )
    assert [chunk.model_dump(mode="json") for chunk in service.chunks] == before


def test_fastmcp_protocol_exposes_annotations_and_structured_results(service) -> None:
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session

    async def exercise() -> None:
        server = create_mcp_server(service)
        async with create_connected_server_and_client_session(server) as session:
            await session.initialize()
            listed = await session.list_tools()
            assert {tool.name for tool in listed.tools} == {
                "search_evidence",
                "get_evidence_chunk",
                "verify_citations",
            }
            for tool in listed.tools:
                annotations = tool.annotations.model_dump(by_alias=True)
                assert annotations["readOnlyHint"] is True
                assert annotations["destructiveHint"] is False
                assert annotations["openWorldHint"] is False
            result = await session.call_tool(
                "search_evidence", {"text": "accounting records", "top_k": 1}
            )
            assert result.isError is False
            assert result.structuredContent["hits"][0]["chunk"]["chunk_id"] == (
                "en-retention:000000"
            )

    asyncio.run(exercise())


def test_protocol_rejects_unknown_argument_fields(service) -> None:
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session

    async def exercise() -> None:
        server = create_mcp_server(service)
        async with create_connected_server_and_client_session(server) as session:
            await session.initialize()
            listed = await session.list_tools()
            for tool in listed.tools:
                assert tool.inputSchema.get("additionalProperties") is False
            result = await session.call_tool(
                "search_evidence", {"text": "accounting records", "unexpected": True}
            )
            assert result.isError is True
            unknown = await session.call_tool("execute_command", {"command": "id"})
            assert unknown.isError is True

    asyncio.run(exercise())
