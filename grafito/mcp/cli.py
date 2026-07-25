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

from ..okf import (
    BundleTools,
    ContextTools,
    OKFBundle,
    ThreadConfinedTools,
    ToolRegistry,
)
from .server import serve_mcp


def build_tools(
    bundle_path: str,
    *,
    enable_writes: bool = False,
    tag: str | None = None,
    budget_tokens: int = 2000,
) -> ThreadConfinedTools:
    """A thread-confined toolset over the bundle at ``bundle_path``.

    Composes the two escalón-1 toolsets — :class:`ContextTools` (the one-shot
    ``context`` star tool) and :class:`BundleTools` (browse/search/open/...) —
    into one :class:`ToolRegistry`. Since a registry is itself a ``ToolSet``,
    :class:`ThreadConfinedTools` wraps the pair as a unit: one bundle, one owning
    thread, both tiers on it.

    The bundle is opened **inside** the confinement thread (via the factory, not
    a closure over an already-open bundle): an :class:`OKFBundle` is bound to the
    thread that opened it, and the MCP server runs tool calls off the event loop,
    so a bare toolset would raise ``sqlite3.ProgrammingError``.

    Writes are gated: ``remember`` is excluded unless ``enable_writes`` — the
    server starts read-only. ``autolog`` follows the same flag, since it only
    matters once writes are possible. ``budget_tokens`` caps ``context`` output;
    it is a deployment decision, not a per-call one, so it is fixed here.
    """
    exclude = None if enable_writes else ["remember"]

    def factory() -> ToolRegistry:
        kb = OKFBundle.load(bundle_path, autolog=enable_writes)
        return ToolRegistry(
            [
                ContextTools(kb, tag=tag, budget_tokens=budget_tokens),
                BundleTools(kb, tag=tag, exclude=exclude),
            ]
        )

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
        "--budget-tokens",
        type=int,
        default=2000,
        help="Token budget for the context tool's grounded output (default 2000).",
    )
    parser.add_argument(
        "--enable-writes",
        action="store_true",
        help="Expose the write tool (remember). Off by default — the server is read-only.",
    )
    args = parser.parse_args(argv)

    tools = build_tools(
        args.bundle,
        enable_writes=args.enable_writes,
        tag=args.tag,
        budget_tokens=args.budget_tokens,
    )
    try:
        # tools is a ToolSet; the server always sees a ToolRegistry, so nesting
        # more tiers later is just a longer list here.
        serve_mcp(ToolRegistry([tools]), name=args.name)
    finally:
        # Stop the confinement thread. The in-memory bundle is discarded with the
        # process moments later; there is no save-back in this MVP, so an
        # explicit db close would buy nothing.
        tools.close()


if __name__ == "__main__":
    main()
