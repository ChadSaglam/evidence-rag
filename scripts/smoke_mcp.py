"""Exercise the real stdio MCP server and close its client transport."""

import asyncio
import shutil
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


async def main() -> None:
    parameters = StdioServerParameters(
        command=shutil.which("evidence-rag") or str(ROOT / ".venv" / "bin" / "evidence-rag"),
        args=["mcp-serve", "--chunks", str(ROOT / "artifacts" / "chunks.jsonl")],
    )
    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            listed = await session.list_tools()
            names = {tool.name for tool in listed.tools}
            assert names == {"search_evidence", "get_evidence_chunk", "verify_citations"}
            for tool in listed.tools:
                annotations = tool.annotations.model_dump(by_alias=True)
                assert annotations["readOnlyHint"] is True
                assert annotations["destructiveHint"] is False

            search = await session.call_tool(
                "search_evidence", {"text": "accounting records", "top_k": 1}
            )
            response = search.structuredContent
            assert response["hits"][0]["chunk"]["chunk_id"] == "en-retention:000000"
            verification = await session.call_tool(
                "verify_citations",
                {
                    "answer": "Accounting records must be retained for ten years [1].",
                    "response": response,
                },
            )
            assert verification.structuredContent["fully_supported"] is True
            print("Stdio MCP initialize, tools/list, and tools/call passed; transport closed")


if __name__ == "__main__":
    asyncio.run(main())
