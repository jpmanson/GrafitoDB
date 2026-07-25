"""Tests for the MCP server (grafito.mcp).

Skipped unless the optional ``mcp`` SDK is installed (``grafito[mcp]``). The
round-trip test spawns the real server as a subprocess and drives it through an
MCP client — the same path a Claude Desktop / Claude Code client takes.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")
pytest.importorskip("yaml")

from grafito import GrafitoDatabase  # noqa: E402
from grafito.mcp.cli import (  # noqa: E402
    _PersistAfterWrite,
    build_graph_tools,
    build_tools,
    main,
    resolve_embedder,
)
from grafito.mcp.server import _to_mcp_tools  # noqa: E402
from grafito.okf import OKFBundle  # noqa: E402

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


class _RecordingToolSet:
    """A ToolSet whose call returns whatever it is told, recording invocations."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[str] = []
        self.schemas = [
            {"type": "function", "function": {"name": "remember", "description": "w"}}
        ]

    def call(self, name: str, args: dict) -> str:
        self.calls.append(name)
        return self.reply


def test_persist_after_write_saves_only_on_successful_write():
    saves = []

    ok = _PersistAfterWrite(_RecordingToolSet('{"saved": "x"}'), lambda: saves.append(1))
    ok.call("remember", {})
    assert saves == [1]  # a successful write persisted

    failed = _PersistAfterWrite(_RecordingToolSet('{"error": "nope"}'), lambda: saves.append(2))
    failed.call("remember", {})
    assert saves == [1]  # a failed write did not


def test_persist_after_write_ignores_reads():
    saves = []
    reader = _RecordingToolSet('{"id": "c"}')
    reader.schemas = [{"type": "function", "function": {"name": "open"}}]
    _PersistAfterWrite(reader, lambda: saves.append(1)).call("open", {})
    assert saves == []  # only write tools trigger a save


def test_writes_persist_back_to_the_bundle_across_reload(tmp_path):
    """A remembered note survives the process: written to markdown, reloadable."""
    dest = tmp_path / "kb"
    shutil.copytree(BUNDLE, dest)

    tools = build_tools(str(dest), enable_writes=True)
    try:
        tools.call(
            "remember",
            {"concept_id": "notes/persisted", "title": "Persisted note", "body": "survives"},
        )
    finally:
        tools.close()

    assert (dest / "notes" / "persisted.md").exists()
    reloaded = OKFBundle.load(str(dest), import_log=True)
    try:
        note = reloaded.concept("notes/persisted")
        assert note is not None and note.title == "Persisted note"
        assert any("persisted" in entry["text"] for entry in reloaded.log())
    finally:
        reloaded.db.close()


def test_build_graph_tools_exposes_only_graph_tiers(tmp_path):
    """--db mode: graph tools over a raw database, no OKF tools."""
    db_path = str(tmp_path / "graph.db")
    db = GrafitoDatabase(db_path)
    a = db.create_node(["Person"], {"name": "Alice"}).id
    b = db.create_node(["Company"], {"name": "Acme"}).id
    db.create_relationship(a, b, "WORKS_AT", {})
    db.close()

    tools = build_graph_tools(db_path)
    try:
        names = _tool_names(tools)
        assert set(names) == {
            "graph_schema", "text_search", "vector_search", "graph_neighbors", "graph_query"
        }
        # No OKF tools leaked in.
        assert not {"context", "browse", "search", "remember"} & set(names)
        schema = json.loads(tools.call("graph_schema", {}))
        assert schema["node_count"] == 2
    finally:
        tools.close()


def test_db_mode_rejects_bundle_only_flags():
    with pytest.raises(SystemExit):
        main(["--db", "x.db", "--enable-writes"])


def test_requires_a_source():
    with pytest.raises(SystemExit):
        main(["--name", "x"])


def test_bundle_and_db_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        main(["--bundle", "a", "--db", "b"])


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
    assert "graph_query" not in names  # graph tier off by default
    # The star tool returns packed grounded text plus its citations.
    payload = json.loads(grounded)
    assert payload["text"] and "citations" in payload
    assert any("sqlite" in cid.lower() for cid in payload["concepts"])
    assert "use-sqlite" in opened  # grounded content came back
    assert "error" in bad  # a bad id is the tool's payload, not a crash


def test_build_tools_adds_graph_tier_when_enabled():
    tools = build_tools(BUNDLE, enable_graph=True)
    try:
        names = _tool_names(tools)
        # escalón 2-3 tools compose alongside escalón 1, no name collision.
        for expected in ("context", "graph_schema", "graph_neighbors", "graph_query"):
            assert expected in names
    finally:
        tools.close()


def test_graph_tier_is_off_by_default():
    tools = build_tools(BUNDLE)
    try:
        assert "graph_query" not in _tool_names(tools)
    finally:
        tools.close()
