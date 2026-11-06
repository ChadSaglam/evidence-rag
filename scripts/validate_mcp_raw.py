"""Protocol-level MCP check with a hand-written JSON-RPC client (no MCP SDK).

scripts/smoke_mcp.py uses the same Python SDK as the server, so SDK bugs would cancel
out. This client speaks newline-delimited JSON-RPC 2.0 over stdio directly, following
the MCP specification, and fails on any protocol deviation.

Usage:
    python scripts/validate_mcp_raw.py --chunks data/real-gdpr/chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROTOCOL_VERSION = "2025-06-18"
EXPECTED_TOOLS = {"search_evidence", "get_evidence_chunk", "verify_citations"}


class RawClient:
    def __init__(self, command: list[str]) -> None:
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        self.next_id = 1

    def send(self, message: dict) -> None:
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def request(self, method: str, params: dict | None = None) -> dict:
        request_id = self.next_id
        self.next_id += 1
        self.send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"Server closed stdout; stderr: {self.process.stderr.read()}")
            message = json.loads(line)
            assert message.get("jsonrpc") == "2.0", f"Not JSON-RPC 2.0: {message}"
            if message.get("id") == request_id:
                return message

    def close(self) -> int:
        self.process.stdin.close()
        try:
            return self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            raise


def check(condition: bool, label: str) -> None:
    print(("PASS " if condition else "FAIL ") + label)
    if not condition:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=Path("data/real-gdpr/chunks.jsonl"))
    parser.add_argument("--server", default=shutil.which("evidence-rag") or "evidence-rag")
    args = parser.parse_args()

    client = RawClient([args.server, "mcp-serve", "--chunks", str(args.chunks)])
    try:
        init = client.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "raw-jsonrpc-validator", "version": "1.0"},
            },
        )
        result = init.get("result", {})
        check("tools" in result.get("capabilities", {}), "initialize advertises tools capability")
        check(bool(result.get("protocolVersion")), "initialize returns a protocol version")
        client.send({"jsonrpc": "2.0", "method": "notifications/initialized"})

        tools = client.request("tools/list")["result"]["tools"]
        names = {tool["name"] for tool in tools}
        check(names == EXPECTED_TOOLS, f"tools/list exposes exactly {sorted(EXPECTED_TOOLS)}")
        for tool in tools:
            hints = tool.get("annotations", {})
            check(
                hints.get("readOnlyHint") is True and hints.get("destructiveHint") is False,
                f"{tool['name']} is annotated read-only and non-destructive",
            )
            has_schema = tool["inputSchema"].get("type") == "object"
            check(has_schema, f"{tool['name']} has an input schema")

        search = client.request(
            "tools/call",
            {
                "name": "search_evidence",
                "arguments": {
                    "text": "How quickly must a controller notify a personal data breach?",
                    "top_k": 3,
                },
            },
        )["result"]
        hits = search["structuredContent"]["hits"]
        check(not search.get("isError") and len(hits) == 3, "search_evidence returns 3 hits")
        top = hits[0]["chunk"]
        check("article-33" in top["chunk_id"], f"top hit is Article 33 ({top['chunk_id']})")
        check(
            top["source"]["uri"].startswith("https://"), "hit carries source attribution"
        )

        chunk = client.request(
            "tools/call",
            {"name": "get_evidence_chunk", "arguments": {"chunk_id": top["chunk_id"]}},
        )["result"]["structuredContent"]
        check(chunk["text"] == top["text"], "get_evidence_chunk returns the identical chunk")

        verification = client.request(
            "tools/call",
            {
                "name": "verify_citations",
                "arguments": {
                    "answer": "The controller must notify the supervisory authority "
                    "not later than 72 hours after having become aware of it [1].",
                    "response": search["structuredContent"],
                },
            },
        )["result"]["structuredContent"]
        check(verification["claims"][0]["status"] != "invalid_citation", "citation [1] resolves")

        unknown = client.request(
            "tools/call", {"name": "execute_command", "arguments": {"command": "id"}}
        )
        rejected = "error" in unknown or unknown.get("result", {}).get("isError") is True
        check(rejected, "unknown tool execute_command is rejected")

        bad_args = client.request(
            "tools/call",
            {"name": "search_evidence", "arguments": {"text": "x", "unexpected": True}},
        )
        rejected = "error" in bad_args or bad_args.get("result", {}).get("isError") is True
        check(rejected, "unknown argument fields are rejected")
    finally:
        code = client.close()
    check(code == 0, "server exits cleanly when stdin closes")
    print("Raw JSON-RPC MCP validation passed.")


if __name__ == "__main__":
    sys.exit(main())
