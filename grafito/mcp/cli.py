"""``grafito-mcp`` — expose an OKF bundle to any MCP client over stdio.

This is the OKF-specific half: it turns command-line configuration into a
:class:`~grafito.ToolRegistry` and hands it to the generic
:func:`grafito.mcp.server.serve_mcp`. The registry is the only thing the server
sees, so widening the surface later (a graph tool tier, a ``context`` tool) is a
matter of adding ``ToolSet``\\ s here, not touching the server.

Configuration mirrors the proposal's Option A (the OKF/agent-memory tier):

    grafito-mcp --bundle ./okf_bundle            # read-only, text-mode retrieval
    grafito-mcp --bundle ./okf_bundle --embed sentence_transformer   # semantic search
    grafito-mcp --bundle ./okf_bundle --rerank lexical   # rerank context grounding
    grafito-mcp --bundle ./okf_bundle --enable-writes   # remember(), saved to the bundle

With ``--enable-writes`` a remembered note is persisted back to the bundle
directory (markdown + changelog), so it survives the process and shows up in
``git diff`` — the bundle is the memory.

Or point it at a plain graph instead of a bundle — no OKF involved, just the
read-only graph tiers (schema/neighbours/text-search + a read-only Cypher
escape hatch)::

    grafito-mcp --db ./graph.db

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
    from ..okf import Reranker
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


def resolve_reranker(name: str | None, config: dict | None = None) -> "Reranker | None":
    """Build a reranker from a name and optional config, or ``None``.

    Reranking is a server-side decision (whether to pay for a second scoring
    pass), not a per-call one, so it is fixed here and improves the ``context``
    tool's grounding. ``lexical`` is dependency-free; the others
    (``cross_encoder``, ``cohere``, ``voyage``, ``jina``) need their own optional
    dependency or API credentials. Unknown names raise ``ValueError``.
    """
    if name is None:
        return None
    from ..okf import (
        CohereReranker,
        CrossEncoderReranker,
        JinaReranker,
        LexicalReranker,
        VoyageReranker,
    )

    rerankers = {
        "lexical": LexicalReranker,
        "cross_encoder": CrossEncoderReranker,
        "cohere": CohereReranker,
        "voyage": VoyageReranker,
        "jina": JinaReranker,
    }
    if name not in rerankers:
        raise ValueError(f"Unknown reranker '{name}'. Available: {sorted(rerankers)}")
    return rerankers[name](**(config or {}))


def build_tools(
    bundle_path: str,
    *,
    enable_writes: bool = False,
    enable_graph: bool = False,
    tag: str | None = None,
    budget_tokens: int = 2000,
    embed: "EmbeddingFunction | None" = None,
    rerank: "Reranker | None" = None,
    max_rows: int = 100,
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
    by meaning; without it retrieval degrades to text mode. ``rerank`` adds a
    reranker to the ``context`` tool's retrieval, a server-side quality knob.

    ``enable_graph`` adds the escalón 2-3 tiers — :class:`~grafito.GraphTools`
    (structured read-only: schema/neighbours/text-search) and
    :class:`~grafito.CypherTools` (a read-only Cypher escape hatch) — over the
    bundle's underlying graph. These hang on ``kb.db``, not on the bundle, so the
    same tools serve a non-OKF graph unchanged; here they simply widen what a
    client can do with the same registry. All read-only. ``max_rows`` caps
    ``graph_query`` output when the graph tier is on.
    """
    exclude = None if enable_writes else ["remember"]

    def factory() -> ToolRegistry:
        kb = OKFBundle.load(bundle_path, embed=embed, autolog=enable_writes)
        bundle_tools: "ToolSet" = BundleTools(kb, tag=tag, exclude=exclude)
        if enable_writes:
            # save() with no path mirrors the graph back to the load directory.
            bundle_tools = _PersistAfterWrite(bundle_tools, kb.save)
        toolsets: list["ToolSet"] = [
            ContextTools(kb, tag=tag, budget_tokens=budget_tokens, rerank=rerank),
            bundle_tools,
        ]
        if enable_graph:
            from ..tools import CypherTools, GraphTools

            toolsets += [GraphTools(kb.db), CypherTools(kb.db, max_rows=max_rows)]
        return ToolRegistry(toolsets)

    return ThreadConfinedTools(factory, name="grafito-mcp-bundle")


def build_graph_tools(db_path: str, *, max_rows: int = 100) -> ThreadConfinedTools:
    """A thread-confined graph toolset over the Grafito database at ``db_path``.

    The db-mode counterpart to :func:`build_tools`: no OKF, no bundle — just the
    escalón 2-3 graph tiers (:class:`~grafito.GraphTools` +
    :class:`~grafito.CypherTools`) over a plain :class:`~grafito.GrafitoDatabase`.
    This is what the OKF-independence of those toolsets buys: the same server
    fronts an arbitrary graph by loading these instead of the bundle tiers, with
    no OKF anywhere in the process.

    The database is opened **inside** the confinement thread, for the same
    thread-affinity reason as the bundle path — the MCP server runs tool calls
    off the event loop. All read-only: nothing here can mutate the graph.
    """

    def factory() -> ToolRegistry:
        from .. import CypherTools, GrafitoDatabase, GraphTools

        db = GrafitoDatabase(db_path)
        return ToolRegistry([GraphTools(db), CypherTools(db, max_rows=max_rows)])

    return ThreadConfinedTools(factory, name="grafito-mcp-graph")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="grafito-mcp",
        description="Serve an OKF bundle or a Grafito graph to MCP clients over stdio.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", help="Path to an OKF bundle directory.")
    source.add_argument(
        "--db",
        help="Path to a Grafito .db graph file. Serves read-only graph tools only "
        "(no OKF); the --bundle-only flags below do not apply.",
    )
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
        "--rerank",
        default=None,
        metavar="NAME",
        help="Reranker for the context tool's grounding (lexical, cross_encoder, "
        "cohere, voyage, jina). 'lexical' is dependency-free; others need their own "
        "dependency or API credentials.",
    )
    parser.add_argument(
        "--rerank-config",
        default=None,
        metavar="JSON",
        help='Config for --rerank as a JSON object, e.g. \'{"model": "rerank-english-v3.0"}\'.',
    )
    parser.add_argument(
        "--enable-graph",
        action="store_true",
        help="Also expose read-only graph tools (schema, neighbours, text search, and "
        "a read-only Cypher escape hatch) over the underlying graph.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=100,
        help="Cap on rows returned by the read-only Cypher tool (default 100). Applies "
        "with --db or --enable-graph.",
    )
    parser.add_argument(
        "--enable-writes",
        action="store_true",
        help="Expose the write tool (remember) and persist writes back to the bundle "
        "directory. Off by default — the server is read-only.",
    )
    args = parser.parse_args(argv)

    if args.db is not None:
        # The bundle-only flags have no meaning over a raw graph; reject them
        # rather than silently ignore, so the client is not misled.
        misused = [
            flag
            for flag, used in (
                ("--tag", args.tag is not None),
                ("--budget-tokens", args.budget_tokens != 2000),
                ("--embed", args.embed is not None),
                ("--embed-config", args.embed_config is not None),
                ("--rerank", args.rerank is not None),
                ("--rerank-config", args.rerank_config is not None),
                ("--enable-graph", args.enable_graph),
                ("--enable-writes", args.enable_writes),
            )
            if used
        ]
        if misused:
            parser.error(f"{', '.join(misused)} apply to --bundle, not --db")
        tools = build_graph_tools(args.db, max_rows=args.max_rows)
    else:
        embed_config = json.loads(args.embed_config) if args.embed_config else None
        rerank_config = json.loads(args.rerank_config) if args.rerank_config else None
        tools = build_tools(
            args.bundle,
            enable_writes=args.enable_writes,
            enable_graph=args.enable_graph,
            tag=args.tag,
            budget_tokens=args.budget_tokens,
            embed=resolve_embedder(args.embed, embed_config),
            rerank=resolve_reranker(args.rerank, rerank_config),
            max_rows=args.max_rows,
        )

    try:
        # tools is a ToolSet; the server always sees a ToolRegistry, so nesting
        # more tiers later is just a longer list here.
        serve_mcp(ToolRegistry([tools]), name=args.name)
    finally:
        # Stop the confinement thread. A file db is read-only here and the
        # process exits immediately after, so closing the connection explicitly
        # would buy nothing; an in-memory bundle is discarded with the process.
        tools.close()


if __name__ == "__main__":
    main()
