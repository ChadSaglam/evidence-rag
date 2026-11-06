# Read-only MCP tools

The MCP server exposes an allowlist of three deterministic operations over a frozen chunk JSONL artifact. It uses stdio, makes no network requests, and registers no answer-generation, file-write, corpus-mutation, shell, or credential tools.

Retrieved document text is untrusted data. The server returns it as structured evidence and never executes document-supplied instructions.

## Install and run

```bash
pip install -e ".[mcp]"
evidence-rag mcp-serve --chunks artifacts/chunks.jsonl
```

The process speaks MCP over stdio. It does not print retrieval output to stdout outside the protocol and does not start an HTTP listener.

Example client configuration:

```json
{
  "mcpServers": {
    "evidence-rag": {
      "command": "/path/to/evidence-rag/.venv/bin/evidence-rag",
      "args": [
        "mcp-serve",
        "--chunks",
        "/path/to/evidence-rag/artifacts/chunks.jsonl"
      ]
    }
  }
}
```

Replace both placeholder paths with absolute local paths.

## Tool contracts

| Tool | Input | Structured result |
|---|---|---|
| `search_evidence` | `text`, optional `top_k` | `RetrievalResponse` with ranked, attributed chunks |
| `get_evidence_chunk` | `chunk_id` | Exact `Chunk` with source and character offsets |
| `verify_citations` | `answer`, explicit `response`, optional `support_threshold` | `CitationVerificationReport` |

Every tool sets `readOnlyHint: true`, `destructiveHint: false`, `idempotentHint: true`, and `openWorldHint: false`. Input schemas reject unknown fields. Calls dispatch only through the allowlist; names such as `write_document` or `execute_command` are rejected.

Annotations are client-facing hints, not a security boundary. Read-only behavior comes from the implementation: the service holds an in-memory corpus and exposes no mutation or general I/O operation.

`verify_citations` accepts first-stage, decomposed, or reranked response payloads supplied explicitly by the client. It performs lexical support and numeric checks, not semantic entailment or real-world truth verification.

## Deterministic verification

```bash
pytest tests/test_mcp.py
python scripts/smoke_mcp.py
python scripts/validate_mcp_raw.py --chunks data/real-gdpr/chunks.jsonl
```

`validate_mcp_raw.py` speaks newline-delimited JSON-RPC 2.0 directly and does not import the MCP SDK, so it can catch problems the SDK-based smoke test would hide. It found one: the earlier FastMCP adapter generated its own argument models from Python signatures and silently dropped unknown fields before the strict Pydantic contracts ran. The adapter now uses the low-level MCP server. The advertised `inputSchema` is exactly the exported contract, with `additionalProperties: false`, and the SDK validates every call against it.

To try the server in the official MCP Inspector (TypeScript client):

```bash
npx @modelcontextprotocol/inspector --cli .venv/bin/evidence-rag mcp-serve --chunks data/real-gdpr/chunks.jsonl --method tools/list
```

Unit and in-memory protocol tests cover exact tool names, annotations, strict inputs, deterministic ranking, source offsets, citation results, unknown-tool rejection, and corpus immutability. The stdio smoke test performs initialize, `tools/list`, and `tools/call`, then closes the transport and child server.
