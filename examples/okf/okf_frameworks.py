"""Plugging an OKF bundle into third-party agent frameworks.

``run_agent`` is Grafito's own tool-calling loop, but it is not the point of
:class:`~grafito.okf.agent.BundleTools` — the toolset is the point. It exposes
exactly the pair every framework asks for:

* ``tools.schemas`` — OpenAI-style function schemas (JSON Schema parameters)
* ``tools.call(name, args) -> str`` — dispatch, returning JSON

so adapting it to PydanticAI, CrewAI, LangChain/LangGraph or MCP is a schema
translation plus a closure, not a rewrite. This script prints the translated
schemas for each target (no framework needs to be installed to run it) and
carries the real adapter code alongside each one.

    python examples/okf/okf_frameworks.py

**The one rule**: always route through ``tools.call()``. Reimplementing tools
against ``kb.search()`` / ``kb[concept_id]`` directly is the tempting shortcut
and it silently drops the ``where``/``tag`` access boundary — the filter lives
in ``BundleTools``, not in the bundle.

**Threading**: an ``OKFBundle`` belongs to the thread that opened it, and agent
frameworks like to run tools elsewhere. Each adapter below says how it handles
that; :class:`ThreadConfinedTools` is the general answer. This is the one part
that is not a pure schema translation, so read that section before writing an
adapter of your own.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from grafito.okf import BundleTools, OKFBundle, ThreadConfinedTools, ToolSet

sys.path.insert(0, str(Path(__file__).parent))
from okf_knowledge_base import HashingEmbeddingFunction  # noqa: E402  (demo embedder)

BUNDLE = Path(__file__).parent / "okf_knowledge_base"


# --- Threading: the constraint every adapter has to respect -------------------
#
# An OKFBundle belongs to the thread that opened it. SQLite connections are
# created with check_same_thread=True, so touching the bundle from another
# thread raises:
#
#     sqlite3.ProgrammingError: SQLite objects created in a thread can only be
#     used in that same thread.
#
# That collides head-on with agent frameworks, which routinely run tools off the
# main thread — CrewAI does, and PydanticAI does for *sync* tools (its async
# ones stay on the event loop's thread, which is why `pydantic_ai_tools` below
# defines them with `async def`). CrewAI has no async tool path, so it needs
# `ThreadConfinedTools`: one dedicated thread owns the bundle and every call is
# handed to it through a queue.
#
# `ThreadConfinedTools` (from grafito.okf) is a `ToolSet` like any other —
# `schemas` + `call` — so it drops into `run_agent(extra_tools=[...])` or any
# adapter below unchanged. Its `run()` is how you reach the bundle for anything
# that is not a tool call, `kb.save()` included.


# --- MCP ----------------------------------------------------------------------
#
# The closest fit: MCP's tool descriptor is Grafito's with the "function"
# wrapper peeled off and "parameters" renamed. A server is then ~20 lines:
#
#     from mcp.server import Server
#     import mcp.types as types
#
#     server, tools = Server("grafito-okf"), BundleTools(kb)
#
#     @server.list_tools()
#     async def list_tools() -> list[types.Tool]:
#         return [types.Tool(**descriptor) for descriptor in mcp_tools(tools)]
#
#     @server.call_tool()
#     async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
#         return [types.TextContent(type="text", text=tools.call(name, arguments))]
#
# Called directly, *not* via asyncio.to_thread: the bundle was opened on the
# thread running the event loop, and to_thread would move the call off it and
# trip check_same_thread. It does block the loop for the duration of the query —
# acceptable for a stdio server with one client, and `ThreadConfinedTools`
# is the escape hatch if that stops being true.


def mcp_tools(tools: ToolSet) -> list[dict]:
    """``BundleTools.schemas`` -> MCP tool descriptors."""
    return [
        {
            "name": schema["function"]["name"],
            "description": schema["function"]["description"],
            "inputSchema": schema["function"].get(
                "parameters", {"type": "object", "properties": {}}
            ),
        }
        for schema in tools.schemas
    ]


# --- PydanticAI ---------------------------------------------------------------
#
# `Tool.from_schema` takes a plain callable plus the JSON Schema, so no pydantic
# model has to be generated. Build the toolset with `raise_errors=True`: PydanticAI
# turns exceptions into retries, and a `{"error": ...}` string would read as
# success and defeat that.
#
#     from pydantic_ai import Agent, Tool
#
#     tools = BundleTools(kb, raise_errors=True)
#     agent = Agent("anthropic:claude-opus-4-8", tools=pydantic_ai_tools(tools))
#     result = agent.run_sync("why did we pick SQLite?")


def pydantic_ai_tools(tools: ToolSet) -> list:
    """``BundleTools`` -> a list of PydanticAI ``Tool``s (import is lazy)."""
    from pydantic_ai import Tool

    def make(name: str):
        # `async def` on purpose, and it matters: PydanticAI runs *sync* tools in
        # a worker thread, and an OKFBundle belongs to the thread that opened it
        # (SQLite raises "objects created in a thread can only be used in that
        # same thread"). An async tool runs on the event loop's own thread, so
        # the bundle stays where it was created. Do not "fix" this with
        # asyncio.to_thread — that reintroduces the very crash it looks like it
        # would prevent. See the threading note in the module docstring.
        async def run_tool(**kwargs) -> str:
            return tools.call(name, kwargs)

        return run_tool

    return [
        Tool.from_schema(
            make(schema["function"]["name"]),
            name=schema["function"]["name"],
            description=schema["function"]["description"],
            json_schema=schema["function"].get(
                "parameters", {"type": "object", "properties": {}}
            ),
        )
        for schema in tools.schemas
    ]


# --- CrewAI -------------------------------------------------------------------
#
# CrewAI wants `BaseTool` subclasses with a pydantic `args_schema`, which
# `pydantic.create_model` can synthesize from the JSON Schema for the shallow
# argument shapes these tools use.
#
# CrewAI runs tools off the main thread and has no async tool path, so pass it a
# `ThreadConfinedTools` rather than a bare `BundleTools` — with the latter every
# call fails on check_same_thread and the agent gives up (verified against a
# live model, which duly reported "thread-safety issue with the underlying
# SQLite connection" instead of an answer):
#
#     tools = ThreadConfinedTools(
#         lambda: BundleTools(OKFBundle.load(path), raise_errors=True)
#     )
#     agent = Agent(role="KB analyst", goal="...", tools=crewai_tools(tools))


_JSON_TO_PY = {"string": str, "integer": int, "number": float, "boolean": bool, "array": list}


def crewai_tools(tools: ToolSet) -> list:
    """``BundleTools`` -> a list of CrewAI ``BaseTool`` instances (lazy import)."""
    from crewai.tools import BaseTool
    from pydantic import create_model

    def build(schema: dict):
        function = schema["function"]
        name = function["name"]
        parameters = function.get("parameters", {})
        required = set(parameters.get("required", []))
        fields = {
            argument: (
                _JSON_TO_PY.get(spec.get("type"), str)
                if argument in required
                else _JSON_TO_PY.get(spec.get("type"), str) | None,
                ... if argument in required else spec.get("default"),
            )
            for argument, spec in parameters.get("properties", {}).items()
        }
        # Built with type() rather than a `class` statement on purpose: a class
        # body cannot close over the loop variables here (class scopes are
        # skipped in lexical lookup once the same name is assigned inside).
        return type(
            f"{name.title()}Tool",
            (BaseTool,),
            {
                # pydantic's metaclass reads __module__ off the namespace, which
                # a `class` statement supplies and type() does not.
                "__module__": __name__,
                "__annotations__": {"name": str, "description": str, "args_schema": type},
                "name": name,
                "description": function["description"],
                "args_schema": create_model(f"{name.title()}Args", **fields),
                "_run": lambda self, _name=name, **kwargs: tools.call(_name, kwargs),
            },
        )()

    return [build(schema) for schema in tools.schemas]


# --- LangChain / LangGraph ----------------------------------------------------
#
# LangChain's `bind_tools` accepts OpenAI-format schemas unchanged, so there is
# nothing to translate — only dispatch to wire:
#
#     model = ChatAnthropic(model="claude-opus-4-8").bind_tools(tools.schemas)
#     ...
#     for call in response.tool_calls:
#         ToolMessage(content=tools.call(call["name"], call["args"]),
#                     tool_call_id=call["id"])


def main() -> None:
    kb = OKFBundle.load(str(BUNDLE), embed=HashingEmbeddingFunction())

    # Two axes of scoping, independent of each other: `where`/`tag` limit what
    # the tools can reach, `include`/`exclude` limit which tools exist at all.
    # Here: retrieval only, over public concepts, with errors raised.
    tools = BundleTools(kb, exclude=["remember"], raise_errors=True)

    print(f"all tools:      {[s['function']['name'] for s in BundleTools.ALL_SCHEMAS]}")
    print(f"this toolset:   {[s['function']['name'] for s in tools.schemas]}\n")

    print("As MCP tool descriptors:")
    for descriptor in mcp_tools(tools):
        print(f"  {descriptor['name']}: {json.dumps(descriptor['inputSchema'])[:88]}...")

    # `raise_errors=True` is what PydanticAI/CrewAI need to drive their retries;
    # a disabled tool is refused as firmly as a nonexistent one.
    print("\nWith raise_errors=True, failures propagate instead of being wrapped:")
    for name, args in (("open", {"concept_id": "nope/missing"}), ("remember", {})):
        try:
            tools.call(name, args)
        except Exception as exc:
            print(f"  {name}(...) -> {type(exc).__name__}: {exc}")

    # The same call on the default toolset comes back as JSON the model reads.
    print("\nWith the default raise_errors=False, the loop keeps going:")
    print(f"  {BundleTools(kb).call('open', {'concept_id': 'nope/missing'})}")

    # The bundle is single-threaded; frameworks that call tools from another
    # thread need it confined to one. Note the bundle is opened *inside* the
    # factory, i.e. on the thread that will own it.
    print("\nSame toolset, reachable from any thread (CrewAI needs this):")
    confined = ThreadConfinedTools(
        lambda: BundleTools(
            OKFBundle.load(str(BUNDLE), embed=HashingEmbeddingFunction()),
            exclude=["remember"],
        )
    )
    try:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=4) as pool:
            hits = list(
                pool.map(
                    lambda q: json.loads(confined.call("search", {"query": q, "k": 1})),
                    ["slow query", "sqlite", "cypher"],
                )
            )
        for query, hit in zip(["slow query", "sqlite", "cypher"], hits):
            print(f"  {query!r} -> {hit[0]['id'] if hit else '(none)'}")
        print("  (a bare BundleTools raises sqlite3.ProgrammingError here)")
    finally:
        confined.close()

    print(
        "\nAdapter code for PydanticAI, CrewAI, LangChain and MCP is in this "
        "file's comments — each one is a schema translation plus a closure over "
        "tools.call(). Always route through tools.call(): rebuilding the tools "
        "on kb.search()/kb[id] directly drops the where=/tag= access boundary."
    )


if __name__ == "__main__":
    main()
