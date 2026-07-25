"""Tests for the MCP server (grafito.mcp).

Skipped unless the optional ``mcp`` SDK is installed (``grafito[mcp]``). The
round-trip test spawns the real server as a subprocess and drives it through an
MCP client — the same path a Claude Desktop / Claude Code client takes.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")
pytest.importorskip("yaml")

from grafito.mcp.cli import build_tools  # noqa: E402
from grafito.mcp.server import _to_mcp_tools  # noqa: E402

BUNDLE = str(Path("examples") / "okf" / "okf_knowledge_base")


def test_to_mcp_tools_translates_the_canonical_schema():
    """function-wrapper peeled, parameters -> inputSchema."""
    schemas = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Find things.",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            },
        }
    ]
    (tool,) = _to_mcp_tools(schemas)
    assert tool.name == "search"
    assert tool.description == "Find things."
    assert tool.inputSchema == {"type": "object", "properties": {"q": {"type": "string"}}}


def test_to_mcp_tools_defaults_missing_parameters():
    (tool,) = _to_mcp_tools([{"type": "function", "function": {"name": "ping"}}])
    assert tool.inputSchema == {"type": "object", "properties": {}}


def _tool_names(tools) -> list[str]:
    return [schema["function"]["name"] for schema in tools.schemas]


def test_build_tools_is_read_only_by_default():
    tools = build_tools(BUNDLE)
    try:
        assert "remember" not in _tool_names(tools)
    finally:
        tools.run(lambda t: t.kb.db.close())
        tools.close()


def test_build_tools_exposes_remember_when_writes_enabled():
    tools = build_tools(BUNDLE, enable_writes=True)
    try:
        assert "remember" in _tool_names(tools)
    finally:
        tools.run(lambda t: t.kb.db.close())
        tools.close()


async def _roundtrip() -> tuple[str, list[str], str, str]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "grafito.mcp", "--bundle", BUNDLE],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            listed = await session.list_tools()
            opened = await session.call_tool(
                "open", {"concept_id": "decisions/0001-use-sqlite"}
            )
            bad = await session.call_tool("open", {"concept_id": "nope/nope"})
            return (
                init.serverInfo.name,
                [t.name for t in listed.tools],
                opened.content[0].text,
                bad.content[0].text,
            )


def test_server_round_trip_over_stdio():
    """Spawn the real server and drive it end to end through an MCP client."""
    name, names, opened, bad = asyncio.run(_roundtrip())
    assert name == "grafito"
    assert "search" in names and "open" in names
    assert "remember" not in names  # read-only by default
    assert "use-sqlite" in opened  # grounded content came back
    assert "error" in bad  # a bad id is the tool's payload, not a crash
