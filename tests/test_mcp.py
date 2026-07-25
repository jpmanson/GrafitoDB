"""Tests for the MCP server (grafito.mcp).

Skipped unless the optional ``mcp`` SDK is installed (``grafito[mcp]``). The
round-trip test spawns the real server as a subprocess and drives it through an
MCP client — the same path a Claude Desktop / Claude Code client takes.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")
pytest.importorskip("yaml")

from grafito.mcp.cli import build_tools, resolve_embedder  # noqa: E402
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
        names = _tool_names(tools)
        assert "context" in names  # the star tool is always present
        assert "remember" not in names
    finally:
        tools.close()


def test_build_tools_exposes_remember_when_writes_enabled():
    tools = build_tools(BUNDLE, enable_writes=True)
    try:
        assert "remember" in _tool_names(tools)
    finally:
        tools.close()


def test_resolve_embedder_none_is_none():
    assert resolve_embedder(None) is None


def test_resolve_embedder_unknown_name_lists_available():
    with pytest.raises(ValueError, match="Unknown embedding function"):
        resolve_embedder("not-a-real-embedder")


def test_resolve_embedder_builds_a_registered_function():
    """String -> instance via the registry, config passed as kwargs."""
    from grafito.embedding_functions import (
        EmbeddingFunction,
        register_embedding_function_class,
    )

    class _Fake(EmbeddingFunction):
        def __init__(self, dim: int = 4) -> None:
            self.dim = dim

        def __call__(self, input):
            return [[0.0] * self.dim for _ in input]

        @staticmethod
        def name() -> str:
            return "_fake_mcp_test"

        def default_space(self) -> str:
            return "cosine"

        def supported_spaces(self):
            return ["cosine"]

        @staticmethod
        def build_from_config(config):
            return _Fake(**config)

        def get_config(self):
            return {"dim": self.dim}

        @staticmethod
        def validate_config(config):
            pass

    register_embedding_function_class(_Fake)
    embedder = resolve_embedder("_fake_mcp_test", {"dim": 8})
    assert isinstance(embedder, _Fake) and embedder.dim == 8


def test_build_tools_wires_the_embedder_into_retrieval():
    """A passed embedder reaches search — hybrid mode uses the vector index."""
    sys.path.insert(0, str(Path("examples") / "okf"))
    from okf_knowledge_base import HashingEmbeddingFunction  # dep-free demo embedder

    tools = build_tools(BUNDLE, embed=HashingEmbeddingFunction())
    try:
        payload = json.loads(tools.call("context", {"query": "why sqlite"}))
        assert payload["concepts"] and payload["text"]
        hits = json.loads(tools.call("search", {"query": "why sqlite", "k": 3}))
        # A vector score rides along only when an embedder is present.
        assert hits and all("score" in h for h in hits)
    finally:
        tools.close()


async def _roundtrip() -> tuple[str, list[str], str, str, str]:
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
            grounded = await session.call_tool(
                "context", {"query": "why did we pick sqlite?"}
            )
            opened = await session.call_tool(
                "open", {"concept_id": "decisions/0001-use-sqlite"}
            )
            bad = await session.call_tool("open", {"concept_id": "nope/nope"})
            return (
                init.serverInfo.name,
                [t.name for t in listed.tools],
                grounded.content[0].text,
                opened.content[0].text,
                bad.content[0].text,
            )


def test_server_round_trip_over_stdio():
    """Spawn the real server and drive it end to end through an MCP client."""
    name, names, grounded, opened, bad = asyncio.run(_roundtrip())
    assert name == "grafito"
    assert "context" in names and "search" in names and "open" in names
    assert "remember" not in names  # read-only by default
    # The star tool returns packed grounded text plus its citations.
    payload = json.loads(grounded)
    assert payload["text"] and "citations" in payload
    assert any("sqlite" in cid.lower() for cid in payload["concepts"])
    assert "use-sqlite" in opened  # grounded content came back
    assert "error" in bad  # a bad id is the tool's payload, not a crash
