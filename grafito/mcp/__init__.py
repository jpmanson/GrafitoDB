"""Model Context Protocol server for Grafito.

Exposes a :class:`~grafito.okf.ToolRegistry` to any MCP client over stdio, so a
bundle (or, later, a graph) is usable from Claude Desktop / Claude Code without
writing integration code. :func:`serve_mcp` is the generic server; the
``grafito-mcp`` CLI (:mod:`grafito.mcp.cli`) wires an OKF bundle into it.

Requires ``grafito[mcp]``. Importing this module is cheap and dependency-free;
the ``mcp`` SDK is only imported when :func:`serve_mcp` actually runs.
"""

from __future__ import annotations

from .server import serve_mcp

__all__ = ["serve_mcp"]
