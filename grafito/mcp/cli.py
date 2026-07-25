"""``grafito-mcp`` — expose an OKF bundle to any MCP client over stdio.

This is the OKF-specific half: it turns command-line configuration into a
:class:`~grafito.okf.ToolRegistry` and hands it to the generic
:func:`grafito.mcp.server.serve_mcp`. The registry is the only thing the server
sees, so widening the surface later (a graph tool tier, a ``context`` tool) is a
matter of adding ``ToolSet``\\ s here, not touching the server.

Configuration mirrors the proposal's Option A (the OKF/agent-memory tier):

    grafito-mcp --bundle ./okf_bundle            # read-only exploration
    grafito-mcp --bundle ./okf_bundle --enable-writes   # also lets the client remember()

Installed via ``grafito[mcp]``; run without a build step with
``uvx --from 'grafitodb[mcp]' grafito-mcp --bundle ./okf_bundle``.
"""

from __future__ import annotations

import argparse

from ..okf import BundleTools, OKFBundle, ThreadConfinedTools, ToolRegistry
from .server import serve_mcp


def build_tools(
    bundle_path: str,
    *,
    enable_writes: bool = False,
    tag: str | None = None,
) -> ThreadConfinedTools:
    """A thread-confined :class:`BundleTools` over the bundle at ``bundle_path``.

    The bundle is opened **inside** the confinement thread (via the factory, not
    a closure over an already-open bundle): an :class:`OKFBundle` is bound to the
    thread that opened it, and the MCP server runs tool calls off the event loop,
    so a bare :class:`BundleTools` would raise ``sqlite3.ProgrammingError``.

    Writes are gated: ``remember`` is excluded unless ``enable_writes`` — the
    server starts read-only. ``autolog`` follows the same flag, since it only
    matters once writes are possible.
    """
    exclude = None if enable_writes else ["remember"]

    def factory() -> BundleTools:
        kb = OKFBundle.load(bundle_path, autolog=enable_writes)
        return BundleTools(kb, tag=tag, exclude=exclude)

    return ThreadConfinedTools(factory, name="grafito-mcp-bundle")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="grafito-mcp",
        description="Serve an OKF bundle to MCP clients over stdio.",
    )
    parser.add_argument("--bundle", required=True, help="Path to the OKF bundle directory.")
    parser.add_argument("--name", default="grafito", help="Server name reported to the client.")
    parser.add_argument("--tag", default=None, help="Scope every tool to concepts with this tag.")
    parser.add_argument(
        "--enable-writes",
        action="store_true",
        help="Expose the write tool (remember). Off by default — the server is read-only.",
    )
    args = parser.parse_args(argv)

    tools = build_tools(args.bundle, enable_writes=args.enable_writes, tag=args.tag)
    try:
        # tools is a ToolSet; the server always sees a ToolRegistry, so nesting
        # more tiers later is just a longer list here.
        serve_mcp(ToolRegistry([tools]), name=args.name)
    finally:
        # The bundle lives on the confinement thread — close it there, then stop
        # the thread. Both are no-ops-safe if the client never connected.
        tools.run(lambda toolset: toolset.kb.db.close())
        tools.close()


if __name__ == "__main__":
    main()
