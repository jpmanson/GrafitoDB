"""A Model Context Protocol server over a :class:`~grafito.okf.ToolRegistry`.

This is the generic half of the MCP integration: it hangs on a
:class:`ToolRegistry` — a bag of :class:`ToolSet`\\ s — and knows nothing about
OKF, bundles, or graphs. What the tools *are* is decided by whoever builds the
registry (see :mod:`grafito.mcp.cli` for the OKF-bundle wiring). That is the
seam that lets the same server front an OKF bundle today and a plain graph
later without change: add a ``ToolSet``, not a new server.

The registry speaks the canonical OpenAI-style tool schema; this module
translates it to MCP ``Tool`` descriptors at the edge (``_to_mcp_tools``), the
same one-way presentation concern ``pydantic_ai_tools`` / ``mcp_tools`` handle
for their own consumers. The ``mcp`` dependency is imported lazily, so importing
this module without ``grafito[mcp]`` installed is fine — only :func:`serve_mcp`
needs it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from grafito.okf import ToolRegistry


def _to_mcp_tools(schemas: list[dict]) -> list[Any]:
    """Canonical OpenAI-style tool schemas -> MCP ``Tool`` descriptors.

    Peels the ``function`` wrapper and renames ``parameters`` to ``inputSchema``
    — the same translation as ``mcp_tools`` in the frameworks example, kept here
    so the server has no dependency on that example file.
    """
    from mcp.types import Tool

    return [
        Tool(
            name=schema["function"]["name"],
            description=schema["function"].get("description", ""),
            inputSchema=schema["function"].get(
                "parameters", {"type": "object", "properties": {}}
            ),
        )
        for schema in schemas
    ]


async def _run(registry: "ToolRegistry", name: str) -> None:
    """Build the MCP server around ``registry`` and serve it over stdio."""
    import anyio
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent

    server: "Server" = Server(name)

    @server.list_tools()
    async def _list_tools() -> list[Any]:
        return _to_mcp_tools(registry.schemas)

    @server.call_tool()
    async def _call_tool(tool: str, arguments: dict | None) -> list[Any]:
        # registry.call is blocking (it may queue onto a ThreadConfinedTools
        # owner thread); run it off the event loop so the server stays
        # responsive. Routing to any thread is safe by construction — that is
        # exactly what ThreadConfinedTools guarantees.
        result = await anyio.to_thread.run_sync(
            lambda: registry.call(tool, arguments or {})
        )
        return [TextContent(type="text", text=result)]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


def serve_mcp(registry: "ToolRegistry", *, name: str = "grafito") -> None:
    """Serve ``registry``'s tools over MCP on stdio until the client disconnects.

    Blocks. ``registry`` is any :class:`~grafito.okf.ToolRegistry`; the server
    exposes every tool it dispatches and routes each ``tools/call`` back through
    it, so tool errors surface as the tool's own ``{"error": ...}`` payload
    (text content) rather than as protocol errors.

    Requires ``grafito[mcp]``. Raises a clear error if the ``mcp`` SDK is
    missing, matching the lazy-dependency pattern of the LLM clients.
    """
    try:
        import anyio  # noqa: F401  (checked here so the message is about MCP)
        import mcp  # noqa: F401
    except ImportError as exc:
        raise ValueError(
            "The MCP SDK is not installed. Install with `pip install grafito[mcp]`."
        ) from exc

    import anyio

    anyio.run(_run, registry, name)
