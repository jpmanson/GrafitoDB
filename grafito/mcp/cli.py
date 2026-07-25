"""``grafito-mcp`` — expose an OKF bundle to any MCP client over stdio.

This is the OKF-specific half: it turns command-line configuration into a
:class:`~grafito.ToolRegistry` and hands it to the generic
:func:`grafito.mcp.server.serve_mcp`. The registry is the only thing the server
sees, so widening the surface later (a graph tool tier, a ``context`` tool) is a
matter of adding ``ToolSet``\\ s here, not touching the server.

Configuration mirrors the proposal's Option A (the OKF/agent-memory tier):

    grafito-mcp --bundle ./okf_bundle            # read-only, text-mode retrieval
    grafito-mcp --bundle ./okf_bundle --embed sentence_transformer   # semantic search
    grafito-mcp --bundle ./okf_bundle --enable-writes   # remember(), saved to the bundle

With ``--enable-writes`` a remembered note is persisted back to the bundle
directory (markdown + changelog), so it survives the process and shows up in
``git diff`` — the bundle is the memory.

Installed via ``grafito[mcp]``; run without a build step with
``uvx --from 'grafitodb[mcp]' grafito-mcp --bundle ./okf_bundle``.
"""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING, Callable

from ..okf import BundleTools, ContextTools, OKFBundle, ThreadConfinedTools
from ..tools import ToolRegistry
from .server import serve_mcp

if TYPE_CHECKING:
    from ..embedding_functions import EmbeddingFunction
    from ..tools import ToolSet

# BundleTools' only write tool; a call to it that succeeds is what we persist on.
_WRITE_TOOLS = frozenset({"remember"})


class _PersistAfterWrite:
    """Wrap a :class:`ToolSet` so a successful write is flushed to disk.

    ``remember`` mutates the in-memory bundle; without this the note is lost when
    the process exits. This persists after each successful write rather than only
    at shutdown, so a crash cannot swallow what the client was told was saved. It
    is a :class:`ToolSet` itself (delegates ``schemas``/``call``), so it drops
    into the registry in place of the toolset it wraps.

    ``persist`` runs on the same (owning) thread as the call — an
    :class:`OKFBundle` may only be touched there — because the whole toolset
    lives inside the :class:`ThreadConfinedTools` factory.
    """

    def __init__(self, inner: "ToolSet", persist: Callable[[], object]) -> None:
        self._inner = inner
        self._persist = persist
        self.schemas = inner.schemas

    def call(self, name: str, args: dict) -> str:
        result = self._inner.call(name, args)
        if name in _WRITE_TOOLS and not _is_error(result):
            self._persist()
        return result


def _is_error(result: str) -> bool:
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        return False
    return isinstance(data, dict) and "error" in data


def resolve_embedder(
    name: str | None, config: dict | None = None
) -> "EmbeddingFunction | None":
    """Build an embedding function from a registry ``name`` and optional config.

    ``None`` (the default) means no embedder — ``search``/``context`` then run in
    text mode, still useful but keyword-only. A name (e.g. ``sentence_transformer``,
    ``openai``, ``ollama``) is looked up in Grafito's embedding-function registry
    and instantiated with ``config`` as keyword arguments, so the class defaults
    apply — ``resolve_embedder("sentence_transformer")`` is the default MiniLM
    model with no config. The embedder's own optional dependency (e.g.
    ``sentence_transformers``) must be installed; a missing one raises a clear
    error from the class itself.
    """
    if name is None:
        return None
    from ..embedding_functions import get_embedding_function_class

    return get_embedding_function_class(name)(**(config or {}))


def build_tools(
    bundle_path: str,
    *,
    enable_writes: bool = False,
    enable_graph: bool = False,
    tag: str | None = None,
    budget_tokens: int = 2000,
    embed: "EmbeddingFunction | None" = None,
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
    server starts read-only. When writes are on, the bundle is opened with
    ``autolog`` and each successful write is **persisted back to the bundle
    directory** (markdown + changelog), so a remembered note survives the
    process — see :class:`_PersistAfterWrite`. ``budget_tokens`` caps ``context``
    output; it is a deployment decision, not a per-call one, so it is fixed here.
    ``embed`` supplies the embedding function so ``search``/``context`` retrieve
    by meaning; without it retrieval degrades to text mode.

    ``enable_graph`` adds the escalón 2-3 tiers — :class:`~grafito.GraphTools`
    (structured read-only: schema/neighbours/text-search) and
    :class:`~grafito.CypherTools` (a read-only Cypher escape hatch) — over the
    bundle's underlying graph. These hang on ``kb.db``, not on the bundle, so the
    same tools serve a non-OKF graph unchanged; here they simply widen what a
    client can do with the same registry. All read-only.
    """
    exclude = None if enable_writes else ["remember"]

    def factory() -> ToolRegistry:
        kb = OKFBundle.load(bundle_path, embed=embed, autolog=enable_writes)
        bundle_tools: "ToolSet" = BundleTools(kb, tag=tag, exclude=exclude)
        if enable_writes:
            # save() with no path mirrors the graph back to the load directory.
            bundle_tools = _PersistAfterWrite(bundle_tools, kb.save)
        toolsets: list["ToolSet"] = [
            ContextTools(kb, tag=tag, budget_tokens=budget_tokens),
            bundle_tools,
        ]
        if enable_graph:
            from ..tools import CypherTools, GraphTools

            toolsets += [GraphTools(kb.db), CypherTools(kb.db)]
        return ToolRegistry(toolsets)

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
        "--embed",
        default=None,
        metavar="NAME",
        help="Embedding function for semantic search (e.g. sentence_transformer, "
        "openai, ollama). Omit for text-mode retrieval. Needs the embedder's own "
        "optional dependency installed.",
    )
    parser.add_argument(
        "--embed-config",
        default=None,
        metavar="JSON",
        help='Config for --embed as a JSON object, e.g. \'{"model_name": "BAAI/bge-small-en"}\'.',
    )
    parser.add_argument(
        "--enable-graph",
        action="store_true",
        help="Also expose read-only graph tools (schema, neighbours, text search, and "
        "a read-only Cypher escape hatch) over the underlying graph.",
    )
    parser.add_argument(
        "--enable-writes",
        action="store_true",
        help="Expose the write tool (remember) and persist writes back to the bundle "
        "directory. Off by default — the server is read-only.",
    )
    args = parser.parse_args(argv)

    embed_config = json.loads(args.embed_config) if args.embed_config else None
    embedder = resolve_embedder(args.embed, embed_config)

    tools = build_tools(
        args.bundle,
        enable_writes=args.enable_writes,
        enable_graph=args.enable_graph,
        tag=args.tag,
        budget_tokens=args.budget_tokens,
        embed=embedder,
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
