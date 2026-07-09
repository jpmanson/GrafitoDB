"""High-level OKF façade over GrafitoDB.

:class:`OKFBundle` groups OKF operations (load, save, concept lookup, layer
navigation, search) behind an OKF-flavored API, while exposing the underlying
:class:`~grafito.database.GrafitoDatabase` via ``bundle.db`` for full graph
power. It *delegates* to the existing ``import_okf_bundle`` / ``export_okf_bundle``
implementations — it does not duplicate them.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

from ..importers.okf import (
    DEFAULT_EMBED_FIELDS,
    REFERENCE_LABEL,
    concept_document,
)
from ..models import Node
from .concept import Concept, ContextPack, Hit, Proposal

if TYPE_CHECKING:
    from ..database import GrafitoDatabase
    from .rerank import Reranker

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_RRF_K = 60  # reciprocal-rank-fusion constant for hybrid search

# Sentinel distinguishing "not passed" from an explicit None in update_concept.
_UNSET: Any = object()

# Node properties that are grafito/OKF bookkeeping, not editable frontmatter.
_PROTECTED_PROPS = frozenset(
    {
        "concept_id",
        "stub",
        "okf_auto",
        "directory",
        "log",
        "okf_hash",
        "pending_review",
        "pending_similar",
    }
)

# SQL predicate selecting real concept rows (the SQL twin of `_is_concept`).
# `concept_id` is served by the expression index created at import time.
_CONCEPT_SQL = (
    "json_extract(n.properties, '$.concept_id') IS NOT NULL"
    " AND json_extract(n.properties, '$.stub') IS NULL"
    " AND json_extract(n.properties, '$.okf_auto') IS NULL"
    " AND json_extract(n.properties, '$.directory') IS NULL"
    " AND json_extract(n.properties, '$.log') IS NULL"
    " AND json_extract(n.properties, '$.pending_review') IS NULL"
)

# Default cosine-similarity threshold above which `propose()` stages a
# concept for review instead of writing it straight away (semantic mode only).
_DEFAULT_REVIEW_THRESHOLD = 0.85

# Word-token extractor for turning free text into a safe FTS5 MATCH query
# (bareword tokens only — no quotes/colons/hyphens for the query parser to trip on).
_FTS_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# First (alphabetical) label of a node — matches Node.labels ordering.
_FIRST_LABEL_SQL = (
    "(SELECT MIN(l.name) FROM node_labels nl JOIN labels l ON l.id = nl.label_id"
    " WHERE nl.node_id = n.id)"
)


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class OKFBundle:
    """An OKF knowledge bundle backed by a GrafitoDB graph."""

    def __init__(
        self,
        db: "GrafitoDatabase",
        *,
        uri_prefix: str = "okf:",
        embed_index: str = "okf",
        embed_fields: tuple[str, ...] = DEFAULT_EMBED_FIELDS,
        source_path: str | None = None,
        okf_version: str | None = None,
        summary: dict | None = None,
        autolog: bool = False,
    ) -> None:
        self._db = db
        self._uri_prefix = uri_prefix
        self._embed_index = embed_index
        self._embed_fields = embed_fields
        self._source_path = source_path
        self._okf_version = okf_version
        self._summary = summary or {}
        self.autolog = autolog

    # --- lifecycle ---------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        db: "GrafitoDatabase | None" = None,
        embed: Any = None,
        configure_fts: bool = True,
        uri_prefix: str = "okf:",
        embed_index: str = "okf",
        embed_fields: tuple[str, ...] = DEFAULT_EMBED_FIELDS,
        directory_nodes: bool = False,
        import_log: bool = False,
        autolog: bool = False,
        **import_kw: Any,
    ) -> "OKFBundle":
        """Import an OKF bundle and return a façade over it.

        Creates an in-memory database when ``db`` is None. Pass
        ``directory_nodes=True`` / ``import_log=True`` to also materialize the
        directory tree (``CONTAINS``) and ``log.md`` history in the graph.
        ``autolog=True`` makes ``add_concept``/``update_concept``/
        ``remove_concept`` append changelog entries automatically (see
        :meth:`log_entry`); combine with ``import_log=True`` so new entries
        extend the existing history instead of replacing it on ``save()``.
        """
        from ..database import GrafitoDatabase

        if db is None:
            db = GrafitoDatabase(":memory:")
        summary = db.import_okf_bundle(
            str(path),
            embed=embed,
            configure_fts=configure_fts,
            uri_prefix=uri_prefix,
            embed_index=embed_index,
            embed_fields=embed_fields,
            directory_nodes=directory_nodes,
            import_log=import_log,
            **import_kw,
        )
        return cls(
            db,
            uri_prefix=uri_prefix,
            embed_index=embed_index,
            embed_fields=embed_fields,
            source_path=str(path),
            okf_version=cls._read_okf_version(path, uri_prefix),
            summary=summary,
            autolog=autolog,
        )

    @classmethod
    def open(
        cls,
        db: "GrafitoDatabase",
        *,
        embed: Any = None,
        uri_prefix: str = "okf:",
        embed_index: str = "okf",
        embed_fields: tuple[str, ...] = DEFAULT_EMBED_FIELDS,
        source_path: str | Path | None = None,
        autolog: bool = False,
    ) -> "OKFBundle":
        """Wrap an already-imported database without re-importing the bundle.

        The durable-reuse path: ``load(path, db=GrafitoDatabase("kb.db"),
        embed=..., embed_options={"store_embeddings": True})`` once, then in
        later sessions ``open(GrafitoDatabase("kb.db"))`` — concepts, FTS, and
        the persisted vector index are all served from the database file, with
        no markdown parsing and no re-embedding.

        ``embed``: pass the bundle's embedding function when it is a custom one
        the registry cannot rebuild by name; built-in functions (e.g.
        SentenceTransformer) are rehydrated automatically from the index
        metadata. ``source_path`` (optional) sets the default ``save()`` target
        and lets ``okf_version`` be read from the bundle's root ``index.md``.
        """
        if embed is not None:
            db.register_embedding_function(embed.name(), embed)
        # Idempotent; present already for bundles imported by recent versions.
        db.create_node_index(None, "concept_id")
        bundle = cls(
            db,
            uri_prefix=uri_prefix,
            embed_index=embed_index,
            embed_fields=embed_fields,
            source_path=str(source_path) if source_path is not None else None,
            okf_version=(
                cls._read_okf_version(source_path, uri_prefix)
                if source_path is not None
                else None
            ),
            autolog=autolog,
        )
        bundle._summary = {"nodes": len(bundle)}
        return bundle

    def save(
        self,
        path: str | Path | None = None,
        *,
        write_index: bool = True,
        write_viz: bool = False,
        write_log: bool = True,
        prune: bool = True,
    ) -> dict:
        """Export the graph back to an OKF bundle (defaults to the load path).

        By default the written bundle *mirrors* the graph: concept ``.md`` files
        that no longer correspond to a concept (e.g. after ``remove_concept``)
        are deleted, so removals round-trip. Only non-reserved markdown files
        are ever pruned — ``log.md``, images, ``viz.html`` etc. are untouched.
        Pass ``prune=False`` to only add/overwrite files.

        ``write_log`` regenerates per-scope ``log.md`` from the graph's
        ``LogEntry`` nodes (imported history plus :meth:`log_entry` /
        ``autolog`` additions); scopes without entries are left alone.
        """
        target = str(path) if path is not None else self._source_path
        if target is None:
            raise ValueError("No path to save to; pass one or load the bundle from a path.")
        return self._db.export_okf_bundle(
            target,
            uri_prefix=self._uri_prefix,
            write_index=write_index,
            write_viz=write_viz,
            write_log=write_log,
            prune=prune,
        )

    # --- graph escape hatch ------------------------------------------------

    @property
    def db(self) -> "GrafitoDatabase":
        """The underlying GrafitoDB — full Cypher / vector / FTS / traversal."""
        return self._db

    def execute(self, cypher: str, **params: Any) -> list[dict]:
        """Run a Cypher query against the underlying graph."""
        return self._db.execute(cypher, params or None)

    # --- concept access ----------------------------------------------------

    def concept(self, concept_id: str) -> Concept | None:
        """Look up a single concept by its ID."""
        for node in self._db.match_nodes(properties={"concept_id": concept_id}):
            if self._is_concept(node):
                return Concept(self, node)
        return None

    def concepts(
        self,
        *,
        type: str | None = None,
        layer: str | None = None,
        tag: str | None = None,
    ) -> list[Concept]:
        """List concepts (ordered by ID), optionally filtered by type/layer/tag.

        Filtering runs in SQL — only matching nodes are hydrated, so this stays
        fast on large bundles.
        """
        rows = self._concept_query("n.id", type=type, layer=layer, tag=tag)
        return [Concept(self, self._db.get_node(row["id"])) for row in rows]

    def __getitem__(self, concept_id: str) -> Concept:
        concept = self.concept(concept_id)
        if concept is None:
            raise KeyError(concept_id)
        return concept

    def __iter__(self) -> Iterator[Concept]:
        return iter(self.concepts())

    def __len__(self) -> int:
        row = self._db.conn.execute(
            f"SELECT COUNT(*) AS n FROM nodes n WHERE {_CONCEPT_SQL}"
        ).fetchone()
        return int(row["n"])

    # --- topology ----------------------------------------------------------

    def layers(self) -> dict[str, int]:
        """Top-level concept-id segments and their concept counts."""
        rows = self._db.conn.execute(
            f"""
            SELECT CASE WHEN instr(cid, '/') > 0
                        THEN substr(cid, 1, instr(cid, '/') - 1)
                        ELSE '.' END AS layer,
                   COUNT(*) AS n
            FROM (SELECT json_extract(n.properties, '$.concept_id') AS cid
                  FROM nodes n WHERE {_CONCEPT_SQL})
            GROUP BY layer ORDER BY layer
            """
        ).fetchall()
        return {row["layer"]: int(row["n"]) for row in rows}

    def index(self, layer: str | None = None) -> dict:
        """Reconstruct the OKF index view for a directory (progressive disclosure).

        The in-memory equivalent of an ``index.md``: child subdirectories and the
        concepts directly in this directory, with titles/descriptions but **no
        bodies** — so an agent can triage before opening any document.

        ``layer`` is a directory path (``None``/``""`` = bundle root, ``"decisions"``,
        ``"references/joins"``, ...). Returns
        ``{"layer", "subdirs": {name: count}, "concepts": [{"id","title","description","type"}]}``.
        """
        path = (layer or "").strip("/")
        prefix = f"{path}/" if path else ""
        params: list[Any] = []
        where = _CONCEPT_SQL
        if prefix:
            where += " AND json_extract(n.properties, '$.concept_id') LIKE ? ESCAPE '\\'"
            params.append(f"{_escape_like(prefix)}%")
        # Listing needs only id/title/description/type — bodies stay on disk.
        rows = self._db.conn.execute(
            f"""
            SELECT json_extract(n.properties, '$.concept_id') AS cid,
                   json_extract(n.properties, '$.title') AS title,
                   json_extract(n.properties, '$.description') AS description,
                   {_FIRST_LABEL_SQL} AS type
            FROM nodes n WHERE {where}
            ORDER BY cid
            """,
            params,
        ).fetchall()
        subdirs: dict[str, int] = {}
        concepts: list[dict] = []
        for row in rows:
            rest = row["cid"][len(prefix):]
            if "/" in rest:
                child = rest.split("/", 1)[0]
                subdirs[child] = subdirs.get(child, 0) + 1
            elif rest:
                concepts.append(
                    {
                        "id": row["cid"],
                        "title": row["title"] or rest,
                        "description": row["description"],
                        "type": row["type"] or "Concept",
                    }
                )
        return {"layer": path or None, "subdirs": subdirs, "concepts": concepts}

    def children(self, path: str | None = None) -> dict:
        """Immediate children of a directory via ``CONTAINS`` graph traversal.

        Requires ``directory_nodes=True`` at load time. Returns
        ``{"subdirs": [path, ...], "concepts": [Concept, ...]}``.
        """
        rows = self.execute(
            "MATCH (d)-[:CONTAINS]->(c) WHERE d.path = $p AND d.directory = true RETURN c",
            p=path or "",
        )
        subdirs: list[str] = []
        concepts: list[Concept] = []
        for row in rows:
            node = self._node_from_row(row["c"])
            if node.properties.get("directory"):
                subdirs.append(node.properties.get("path"))
            else:
                concepts.append(Concept(self, node))
        concepts.sort(key=lambda concept: concept.id)
        return {"subdirs": sorted(subdirs), "concepts": concepts}

    def log(self, concept_id: str | None = None) -> list[dict]:
        """Log entries (newest first); optionally only those mentioning a concept.

        Requires ``import_log=True`` at load time. Each entry is
        ``{"date", "kind", "text", "scope"}``.
        """
        if concept_id is None:
            rows = self.execute("MATCH (e) WHERE e.log = true RETURN e")
        else:
            rows = self.execute(
                "MATCH (e)-[:MENTIONS]->(c) WHERE e.log = true AND c.concept_id = $cid RETURN e",
                cid=concept_id,
            )
        entries = [
            {
                "date": row["e"]["properties"].get("date"),
                "kind": row["e"]["properties"].get("kind"),
                "text": row["e"]["properties"].get("text"),
                "scope": row["e"]["properties"].get("scope"),
            }
            for row in rows
        ]
        return sorted(entries, key=lambda entry: entry["date"] or "", reverse=True)

    def references(self) -> list[dict]:
        """External citation sources as ``[{"url", "title"}, ...]``."""
        refs = []
        for node in self._db.match_nodes(labels=["Reference"]):
            if node.properties.get("okf_auto"):
                refs.append(
                    {"url": node.properties.get("url"), "title": node.properties.get("title")}
                )
        return refs

    # --- search ------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        mode: str = "auto",
        type: str | None = None,
        layer: str | None = None,
        include_superseded: bool = False,
    ) -> list[Hit]:
        """Search concepts by text or meaning, with a unified result shape.

        ``mode``: ``"auto"`` (semantic if embeddings exist, else text),
        ``"semantic"``, ``"text"``, or ``"hybrid"`` (reciprocal-rank fusion).
        ``"hybrid"`` degrades to text-only when the bundle has no embeddings;
        ``"semantic"`` requires them (load with ``embed=``).

        Concepts marked ``status="superseded"`` (see :meth:`supersede`) are
        excluded by default, so retrieval never surfaces retracted claims as
        current truth; pass ``include_superseded=True`` to see them anyway.
        """
        if mode == "auto":
            mode = "semantic" if self._has_vector_index() else "text"
        elif mode == "hybrid" and not self._has_vector_index():
            mode = "text"

        # Post-filters (layer, superseded) need over-fetch to keep k results.
        fetch = k * 4 if (layer or not include_superseded) else k

        if mode == "semantic":
            hits = self._semantic(query, fetch, type)
        elif mode == "text":
            hits = self._text(query, fetch, type)
        elif mode == "hybrid":
            hits = self._hybrid(query, fetch, type)
        else:
            raise ValueError(f"Unknown search mode: {mode!r}")

        if layer is not None:
            prefix = f"{layer.strip('/')}/"
            hits = [h for h in hits if h.concept.id.startswith(prefix)]
        if not include_superseded:
            hits = [h for h in hits if not h.concept.is_superseded]
        return hits[:k]

    def context(
        self,
        query: str,
        *,
        budget_tokens: int = 2000,
        k: int = 8,
        mode: str = "auto",
        type: str | None = None,
        layer: str | None = None,
        expand_hops: int = 1,
        include_citations: bool = True,
        include_superseded: bool = False,
        token_counter: Callable[[str], int] | None = None,
        rerank: "Reranker | None" = None,
    ) -> ContextPack:
        """Retrieve, graph-expand, and pack grounded context within a token budget.

        Seeds with :meth:`search` (semantic/text/hybrid), then follows the graph
        — what each hit links to (any relationship type except ``CITES``) within
        ``expand_hops`` — so the pack carries context the embedding alone would
        miss. Concepts are rendered as titled,
        cited blocks and greedily added in priority order (seeds first, by score;
        then expanded neighbours) until the budget is reached. The top hit is
        always included, truncated if it alone exceeds the budget.

        ``rerank``: an optional :class:`~grafito.okf.rerank.Reranker` (any callable
        ``(query, candidates) -> [(concept, score), ...]``). When given, the seed +
        expanded pool is re-scored against the query text *before* budgeting — so
        graph-expanded neighbours compete on relevance instead of insertion order.

        ``include_superseded`` (default ``False``) also governs graph-expanded
        neighbours: superseded concepts pulled in via ``SUPERSEDES`` or a stale
        ``LINKS_TO`` edge are dropped unless requested, so a retracted claim
        doesn't leak into the pack through expansion even when the seed search
        excluded it.

        ``budget_tokens`` is measured with ``token_counter`` (default: a ~4
        chars/token heuristic — pass your model's tokenizer for exact budgeting).
        Returns a :class:`ContextPack`; ``str(pack)`` is the prompt-ready text.
        """
        count = token_counter or self._estimate_tokens
        hits = self.search(
            query, k=k, mode=mode, type=type, layer=layer, include_superseded=include_superseded
        )

        # Candidate order: seed concepts first (ranked), then graph-expanded
        # neighbours, de-duplicated by node identity.
        candidates: list[Concept] = []
        seen: set[int] = set()
        for hit in hits:
            if hit.concept.node.id not in seen:
                seen.add(hit.concept.node.id)
                candidates.append(hit.concept)
        if expand_hops > 0:
            for hit in hits:
                for nbr in hit.concept.neighbors(depth=expand_hops):
                    if nbr.node.id not in seen and (include_superseded or not nbr.is_superseded):
                        seen.add(nbr.node.id)
                        candidates.append(nbr)

        # Optional precision step: let an injected reranker decide the order (and,
        # if it has a top_n, the subset) the budget is then packed from.
        if rerank is not None and candidates:
            candidates = [concept for concept, _ in rerank(query, candidates)]

        # Greedy pack: whole blocks while they fit; force-truncate the first one
        # if even it overflows, so the top hit is never dropped silently.
        blocks: list[str] = []
        included: list[Concept] = []
        remaining = budget_tokens
        truncated = False
        for concept in candidates:
            block = self._render_block(concept, include_citations)
            cost = count(block)
            if cost <= remaining:
                blocks.append(block)
                included.append(concept)
                remaining -= cost
            elif not included:
                blocks.append(self._truncate(block, remaining, count))
                included.append(concept)
                remaining = 0
                truncated = True
                break
            else:
                truncated = True

        citations = self._collect_citations(included) if include_citations else []
        text = "\n\n".join(blocks)
        return ContextPack(
            text=text,
            citations=citations,
            concepts=included,
            hits=hits,
            tokens=count(text) if text else 0,
            truncated=truncated,
        )

    # --- mutation ----------------------------------------------------------

    def log_entry(
        self,
        text: str,
        *,
        kind: str = "Update",
        date: str | None = None,
        scope: str = "",
        concepts: "list[Concept | str] | None" = None,
    ) -> dict:
        """Append a changelog entry (a ``LogEntry`` node, SPEC sec. 7).

        ``save()`` serializes entries back to the scope's ``log.md``; entries
        mentioning ``concepts`` are linked via ``MENTIONS`` so ``log(cid)``
        finds them. ``date`` defaults to today (ISO ``YYYY-MM-DD``); ``kind``
        is the conventional leading bold word (``Creation``/``Update``/...).
        Set ``autolog=True`` at load time to have concept mutations call this
        automatically.

        Round-trip note: on re-import, ``MENTIONS`` edges are re-derived from
        markdown links in the entry *text* — include ``[Title](/id.md)`` in
        ``text`` (as autolog entries do) if the mention should survive
        markdown round-trips.
        """
        date = date or datetime.date.today().isoformat()
        entry_text = text if text.startswith("**") else f"**{kind}**: {text}"
        node = self._db.create_node(
            labels=["LogEntry"],
            properties={
                "date": date,
                "kind": kind,
                "text": entry_text,
                "scope": scope,
                "log": True,
            },
        )
        for target in concepts or []:
            self._db.create_relationship(node.id, self._resolve(target).node.id, "MENTIONS")
        return {"date": date, "kind": kind, "text": entry_text, "scope": scope}

    def add_concept(
        self,
        concept_id: str,
        *,
        type: str,
        title: str | None = None,
        body: str = "",
        description: str | None = None,
        tags: list[str] | None = None,
        **frontmatter: Any,
    ) -> Concept:
        """Create a new concept (and embed it if the bundle has a vector index).

        Raises ``ValueError`` if a concept with this ID already exists. The new
        node is auto-indexed for full-text search; ``save()`` persists it to
        markdown. Use ``link``/``cite`` to relate it to other concepts.
        """
        if self.concept(concept_id) is not None:
            raise ValueError(f"Concept already exists: {concept_id!r}")
        props: dict[str, Any] = {"concept_id": concept_id, "body": body or ""}
        if title is not None:
            props["title"] = title
        if description is not None:
            props["description"] = description
        if tags is not None:
            props["tags"] = list(tags)
        props.update(frontmatter)
        node = self._db.create_node(
            labels=[type], properties=props, uri=f"{self._uri_prefix}{concept_id}"
        )
        if self._has_vector_index():
            self._db.upsert_embeddings(
                [node.id], [concept_document(props, self._embed_fields)], index=self._embed_index
            )
        concept = Concept(self, node)
        if self.autolog:
            self.log_entry(
                f"Created [{title or concept_id}](/{concept_id}.md).",
                kind="Creation",
                concepts=[concept],
            )
        return concept

    def update_concept(
        self,
        concept_id: str,
        *,
        type: Any = _UNSET,
        title: Any = _UNSET,
        description: Any = _UNSET,
        body: Any = _UNSET,
        tags: Any = _UNSET,
        **frontmatter: Any,
    ) -> Concept:
        """Update a concept's fields in place (re-embedding and re-indexing it).

        Only the fields you pass change; passing ``None`` removes an optional
        field (``body=None`` clears it to ``""``). ``type`` relabels the node.
        Extra keyword arguments update producer-defined frontmatter keys the
        same way. Raises ``ValueError`` for an unknown concept.

        The full-text index follows automatically; the vector embedding is
        recomputed when the bundle has one. ``save()`` persists the new state.
        """
        concept = self._resolve(concept_id)
        node = concept.node
        props = dict(node.properties)

        named = {"title": title, "description": description, "tags": tags}
        for key, value in {**named, **frontmatter}.items():
            if value is _UNSET:
                continue
            if key in _PROTECTED_PROPS:
                raise ValueError(f"Cannot update reserved property: {key!r}")
            if value is None:
                props.pop(key, None)
            elif key == "tags":
                props[key] = list(value)
            else:
                props[key] = value
        if body is not _UNSET:
            props["body"] = body or ""

        self._db.replace_node_properties(node.id, props)

        if type is not _UNSET:
            if not isinstance(type, str) or not type:
                raise ValueError("type must be a non-empty string")
            if node.labels != [type]:
                if node.labels:
                    self._db.remove_labels(node.id, node.labels)
                self._db.add_labels(node.id, [type])

        if self._has_vector_index():
            self._db.upsert_embeddings(
                [node.id], [concept_document(props, self._embed_fields)], index=self._embed_index
            )
        updated = Concept(self, self._db.get_node(node.id))
        if self.autolog:
            changed = [k for k, v in {**named, **frontmatter}.items() if v is not _UNSET]
            if body is not _UNSET:
                changed.append("body")
            if type is not _UNSET:
                changed.append("type")
            self.log_entry(
                f"Updated [{updated.title or concept_id}](/{concept_id}.md)"
                f" ({', '.join(sorted(changed))}).",
                kind="Update",
                concepts=[updated],
            )
        return updated

    def link(
        self,
        src: "Concept | str",
        dst: "Concept | str",
        *,
        type: str = "LINKS_TO",
        anchor: str | None = None,
    ) -> None:
        """Create a relationship from ``src`` to ``dst`` (both concepts)."""
        source = self._resolve(src)
        target = self._resolve(dst)
        self._db.create_relationship(
            source.node.id, target.node.id, type, {"anchor": anchor} if anchor else {}
        )

    def cite(
        self, src: "Concept | str", target: "Concept | str", *, anchor: str | None = None
    ) -> None:
        """Create a ``CITES`` edge from ``src`` to a URL or another concept.

        A URL target resolves to a deduplicated ``Reference`` node.
        """
        source = self._resolve(src)
        if isinstance(target, str) and target.startswith(("http://", "https://")):
            target_id = self._reference_node(target, anchor).id
        else:
            target_id = self._resolve(target).node.id
        self._db.create_relationship(
            source.node.id, target_id, "CITES", {"anchor": anchor} if anchor else {}
        )

    def remove_concept(self, concept_id: str) -> bool:
        """Delete a concept (and its relationships/embedding). Returns success."""
        concept = self.concept(concept_id)
        if concept is None:
            return False
        node_id = concept.node.id
        if self._has_vector_index():
            try:
                self._db.remove_embedding(node_id, index=self._embed_index)
            except Exception:
                pass
        deleted = self._db.delete_node(node_id)
        if deleted and self.autolog:
            self.log_entry(f"Removed {concept_id}.", kind="Removal")
        return deleted

    def supersede(
        self, old: "Concept | str", new: "Concept | str", *, note: str | None = None
    ) -> Concept:
        """Mark ``old`` as superseded by ``new`` (SPEC trust model).

        Per **append-only-on-meaning**: a claim is never rewritten in place to
        change its meaning — a correction creates a new concept and links the
        old one to it. Sets ``status="superseded"``/``superseded_by`` on
        ``old`` and appends to ``supersedes`` on ``new``, and links
        ``new -[:SUPERSEDES]-> old`` so the relationship is graph-queryable
        and, with ``typed_links=True``, round-trips through markdown.
        Superseded concepts are excluded from :meth:`search`/:meth:`context`
        by default (pass ``include_superseded=True`` to include them).
        """
        old_c = self._resolve(old)
        new_c = self._resolve(new)
        if old_c.id == new_c.id:
            raise ValueError("A concept cannot supersede itself")
        # Re-fetch: a caller-held Concept may be stale (e.g. a second supersede()
        # onto the same `new` after the first already appended to `supersedes`).
        old_node = self._db.get_node(old_c.node.id)
        new_node = self._db.get_node(new_c.node.id)

        old_props = dict(old_node.properties)
        old_props["status"] = "superseded"
        old_props["superseded_by"] = new_c.id
        self._db.replace_node_properties(old_c.node.id, old_props)

        new_props = dict(new_node.properties)
        supersedes_value = new_props.get("supersedes")
        supersedes = list(supersedes_value) if isinstance(supersedes_value, list) else (
            [supersedes_value] if supersedes_value else []
        )
        if old_c.id not in supersedes:
            supersedes.append(old_c.id)
        new_props["supersedes"] = supersedes
        self._db.replace_node_properties(new_c.node.id, new_props)

        self._db.create_relationship(new_c.node.id, old_c.node.id, "SUPERSEDES")

        if self._has_vector_index():
            self._db.upsert_embeddings(
                [old_c.node.id],
                [concept_document(old_props, self._embed_fields)],
                index=self._embed_index,
            )

        if self.autolog:
            text = (
                f"[{new_c.title or new_c.id}](/{new_c.id}.md) supersedes"
                f" [{old_c.title or old_c.id}](/{old_c.id}.md)."
            )
            if note:
                text += f" {note}"
            self.log_entry(text, kind="Supersede", concepts=[old_c, new_c])

        return Concept(self, self._db.get_node(new_c.node.id))

    def conflicts_with(
        self, a: "Concept | str", b: "Concept | str", *, note: str | None = None
    ) -> None:
        """Flag ``a`` and ``b`` as contradicting each other (SPEC trust model).

        A conflict has no natural direction, so a ``CONFLICTS_WITH``
        relationship is created both ways, optionally carrying ``note``.
        Neither concept is changed or removed — resolving the conflict is a
        deliberate follow-up (:meth:`supersede` one side, or update both with
        clarifying context). This is the *default* when new information
        contradicts an existing concept without strong enough evidence to
        supersede it outright.
        """
        ca = self._resolve(a)
        cb = self._resolve(b)
        if ca.id == cb.id:
            raise ValueError("A concept cannot conflict with itself")
        props = {"anchor": note} if note else {}
        self._db.create_relationship(ca.node.id, cb.node.id, "CONFLICTS_WITH", props)
        self._db.create_relationship(cb.node.id, ca.node.id, "CONFLICTS_WITH", props)

        if self.autolog:
            text = (
                f"[{ca.title or ca.id}](/{ca.id}.md) conflicts with"
                f" [{cb.title or cb.id}](/{cb.id}.md)."
            )
            if note:
                text += f" {note}"
            self.log_entry(text, kind="Conflict", concepts=[ca, cb])

    def propose(
        self,
        concept_id: str,
        *,
        type: str,
        title: str | None = None,
        body: str = "",
        description: str | None = None,
        tags: list[str] | None = None,
        auto_approve: bool | None = None,
        similarity_threshold: float = _DEFAULT_REVIEW_THRESHOLD,
        similarity_k: int = 3,
        **frontmatter: Any,
    ) -> "Concept | Proposal":
        """Suggest a new concept, gated by a review queue (agent-write safety valve).

        Like :meth:`add_concept`, but instead of always writing immediately,
        checks whether the proposal looks like it might duplicate or overlap
        existing knowledge before deciding:

        - ``auto_approve=True`` always writes it now (equivalent to
          ``add_concept``) and returns the new :class:`Concept`.
        - ``auto_approve=False`` always stages it and returns a
          :class:`Proposal`, regardless of similarity.
        - ``auto_approve=None`` (default, *conditional* mode): searches the
          bundle for concepts similar to the proposal. With a vector index,
          it auto-approves unless a hit scores at or above
          ``similarity_threshold`` (cosine similarity, default 0.85);
          text-only search has no comparable numeric scale, so *any* hit at
          all triggers review. No hits (or nothing to compare — no
          title/description/body) auto-approves.

        A staged proposal is a real node in the graph (so ``pending_reviews()``
        survives process restarts) but is excluded from ``concept()``,
        ``concepts()``, ``search()``, iteration, and export until approved —
        it does not pollute retrieval or round-trip to markdown. It does not
        create any links/citations; wire those up with :meth:`link`/:meth:`cite`
        after :meth:`approve`.

        Raises ``ValueError`` if a concept with this ID already exists (an ID
        collision is a hard conflict, not a similarity judgment call).
        """
        if self.concept(concept_id) is not None:
            raise ValueError(f"Concept already exists: {concept_id!r}")

        props: dict[str, Any] = {"concept_id": concept_id, "body": body or ""}
        if title is not None:
            props["title"] = title
        if description is not None:
            props["description"] = description
        if tags is not None:
            props["tags"] = list(tags)
        props.update(frontmatter)

        similar: list[dict] = []
        if auto_approve is None:
            query_text = concept_document(props, self._embed_fields)
            if query_text.strip():
                semantic = self._has_vector_index()
                if semantic:
                    hits = self.search(query_text, k=similarity_k, mode="semantic")
                    hits = [h for h in hits if h.score >= similarity_threshold]
                else:
                    # Free text (markdown punctuation, quotes, hyphens...) is not
                    # a safe FTS5 MATCH query — reduce it to a bareword OR-query.
                    fts_query = " OR ".join(_FTS_TOKEN_RE.findall(query_text)[:32])
                    hits = self.search(fts_query, k=similarity_k, mode="text") if fts_query else []
                if hits:
                    similar = [
                        {
                            "concept_id": h.concept.id,
                            "title": h.concept.title,
                            "score": h.score,
                            "via": h.via,
                        }
                        for h in hits
                    ]
            auto_approve = not similar

        if auto_approve:
            return self.add_concept(
                concept_id,
                type=type,
                title=title,
                body=body,
                description=description,
                tags=tags,
                **frontmatter,
            )

        props["pending_review"] = True
        if similar:
            props["pending_similar"] = similar
        node = self._db.create_node(
            labels=[type], properties=props, uri=f"{self._uri_prefix}{concept_id}"
        )
        proposal = Proposal(self, node)
        if self.autolog:
            self.log_entry(
                f"Proposed {title or concept_id} ({concept_id}) — pending review.",
                kind="Proposal",
            )
        return proposal

    def pending_reviews(self) -> list[Proposal]:
        """List concepts staged by :meth:`propose`, ordered by concept ID."""
        nodes = list(self._db.match_nodes(properties={"pending_review": True}))
        nodes.sort(key=lambda n: n.properties.get("concept_id") or "")
        return [Proposal(self, n) for n in nodes]

    def approve(self, proposal: "Proposal | str") -> Concept:
        """Materialize a staged proposal into a real, retrievable concept.

        Clears the review markers (same node ID — any relationships added to
        the pending node, e.g. by a reviewer inspecting it, are preserved) and
        embeds it if the bundle has a vector index.
        """
        node = self._resolve_proposal(proposal)
        props = dict(node.properties)
        props.pop("pending_review", None)
        props.pop("pending_similar", None)
        self._db.replace_node_properties(node.id, props)
        if self._has_vector_index():
            self._db.upsert_embeddings(
                [node.id], [concept_document(props, self._embed_fields)], index=self._embed_index
            )
        concept = Concept(self, self._db.get_node(node.id))
        if self.autolog:
            self.log_entry(
                f"Approved [{concept.title or concept.id}](/{concept.id}.md).",
                kind="Creation",
                concepts=[concept],
            )
        return concept

    def reject(self, proposal: "Proposal | str", *, note: str | None = None) -> bool:
        """Discard a staged proposal. Returns whether a proposal was found."""
        try:
            node = self._resolve_proposal(proposal)
        except ValueError:
            return False
        concept_id = node.properties.get("concept_id")
        title = node.properties.get("title")
        deleted = self._db.delete_node(node.id)
        if deleted and self.autolog:
            text = f"Rejected proposal for {title or concept_id} ({concept_id})."
            if note:
                text += f" {note}"
            self.log_entry(text, kind="Rejection")
        return deleted

    def _resolve_proposal(self, value: "Proposal | str") -> Node:
        if isinstance(value, Proposal):
            node = self._db.get_node(value.node.id)
            if node is not None and node.properties.get("pending_review"):
                return node
            raise ValueError(f"Not a pending proposal: {value.id!r}")
        if not isinstance(value, str):
            raise TypeError(f"Expected a Proposal or concept ID string, got {type(value).__name__}")
        for node in self._db.match_nodes(properties={"concept_id": value, "pending_review": True}):
            return node
        raise ValueError(f"Unknown pending proposal: {value!r}")

    # --- metadata ----------------------------------------------------------

    @property
    def uri_prefix(self) -> str:
        return self._uri_prefix

    @property
    def okf_version(self) -> str | None:
        return self._okf_version

    @property
    def summary(self) -> dict:
        return dict(self._summary)

    def __repr__(self) -> str:
        n = self._summary.get("nodes", "?")
        return f"<OKFBundle {self._source_path!r} concepts={n}>"

    # --- internals ---------------------------------------------------------

    def _concept_query(
        self,
        columns: str,
        *,
        type: str | None = None,
        layer: str | None = None,
        tag: str | None = None,
    ) -> list:
        """Select concept rows with SQL-side type/layer/tag filtering."""
        sql = f"SELECT {columns} FROM nodes n"
        params: list[Any] = []
        conds = [_CONCEPT_SQL]
        if type is not None:
            sql += " JOIN node_labels nl ON nl.node_id = n.id JOIN labels l ON l.id = nl.label_id"
            conds.append("l.name = ?")
            params.append(type)
        if layer is not None:
            conds.append("json_extract(n.properties, '$.concept_id') LIKE ? ESCAPE '\\'")
            params.append(f"{_escape_like(layer)}/%")
        if tag is not None:
            conds.append(
                "EXISTS (SELECT 1 FROM json_each(n.properties, '$.tags') jt WHERE jt.value = ?)"
            )
            params.append(tag)
        sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY json_extract(n.properties, '$.concept_id')"
        return self._db.conn.execute(sql, params).fetchall()

    def _resolve(self, value: "Concept | str") -> Concept:
        if isinstance(value, Concept):
            return value
        concept = self.concept(value)
        if concept is None:
            raise ValueError(f"Unknown concept: {value!r}")
        return concept

    def _reference_node(self, url: str, anchor: str | None) -> Node:
        for node in self._db.match_nodes(properties={"url": url}):
            if node.properties.get("okf_auto"):
                return node
        return self._db.create_node(
            labels=[REFERENCE_LABEL],
            properties={"title": anchor or url, "url": url, "okf_auto": True},
            uri=url,
        )

    @staticmethod
    def _is_concept(node: Node) -> bool:
        """True for real concepts (not stubs, Reference, Directory, LogEntry, or a pending proposal)."""
        props = node.properties
        return not (
            props.get("stub")
            or props.get("okf_auto")
            or props.get("directory")
            or props.get("log")
            or props.get("pending_review")
        )

    def _has_vector_index(self) -> bool:
        try:
            return any(i.get("name") == self._embed_index for i in self._db.list_vector_indexes())
        except Exception:
            return False

    def _node_from_row(self, value: dict) -> Node:
        return Node(
            id=value["id"],
            labels=value.get("labels", []),
            properties=value.get("properties", {}),
            uri=value.get("uri"),
        )

    def _neighbors(
        self, concept_id: str, rel_type: str | None, direction: str, *, depth: int = 1
    ) -> list[Concept]:
        """Concepts within ``depth`` hops. ``rel_type=None`` follows any
        relationship type except ``CITES`` (citations have their own accessor)."""
        if rel_type is not None and not _IDENTIFIER_RE.fullmatch(rel_type):
            raise ValueError(f"Invalid relationship type: {rel_type!r}")
        start_rows = self._db.conn.execute(
            "SELECT n.id FROM nodes n WHERE json_extract(n.properties, '$.concept_id') = ?",
            (concept_id,),
        ).fetchall()
        frontier = {row["id"] for row in start_rows}
        if not frontier:
            return []
        src_col, dst_col = ("source_node_id", "target_node_id")
        if direction == "in":
            src_col, dst_col = dst_col, src_col
        seen: set[int] = set(frontier)
        reached: list[int] = []
        for _ in range(max(1, int(depth))):
            if not frontier:
                break
            placeholders = ",".join("?" * len(frontier))
            sql = (
                f"SELECT DISTINCT r.{dst_col} AS nid FROM relationships r"
                f" WHERE r.{src_col} IN ({placeholders})"
            )
            params: list[Any] = sorted(frontier)
            if rel_type is not None:
                sql += " AND r.type = ?"
                params.append(rel_type)
            else:
                sql += " AND r.type != 'CITES'"
            frontier = set()
            for row in self._db.conn.execute(sql + " ORDER BY nid", params).fetchall():
                if row["nid"] not in seen:
                    seen.add(row["nid"])
                    frontier.add(row["nid"])
                    reached.append(row["nid"])
        out: list[Concept] = []
        for node_id in reached:
            node = self._db.get_node(node_id)
            cid = node.properties.get("concept_id") if node else None
            if isinstance(cid, str) and cid != concept_id:
                out.append(Concept(self, node))
        return out

    def _citations_of(self, concept_id: str) -> list[dict]:
        rows = self.execute(
            "MATCH (a)-[r:CITES]->(t) WHERE a.concept_id = $cid "
            "RETURN t, r.anchor AS anchor",
            cid=concept_id,
        )
        cites: list[dict] = []
        for row in rows:
            target = self._node_from_row(row["t"])
            anchor = row.get("anchor")
            if target.properties.get("okf_auto"):
                cites.append({"url": target.properties.get("url"), "anchor": anchor})
            else:
                cites.append({"concept": target.properties.get("concept_id"), "anchor": anchor})
        return cites

    def _semantic(self, query: str, k: int, type: str | None) -> list[Hit]:
        results = self._db.semantic_search(
            query, k=k, index=self._embed_index, filter_labels=[type] if type else None
        )
        return [
            Hit(Concept(self, r["node"]), float(r["score"]), "semantic")
            for r in results
            if self._is_concept(r["node"])
        ]

    def _text(self, query: str, k: int, type: str | None) -> list[Hit]:
        results = self._db.text_search(query, k=k, labels=[type] if type else None)
        hits: list[Hit] = []
        for r in results:
            if r.get("entity_type") != "node":
                continue
            node = r["entity"]
            if self._is_concept(node):
                hits.append(Hit(Concept(self, node), float(r["score"]), "text"))
        return hits

    def _hybrid(self, query: str, k: int, type: str | None) -> list[Hit]:
        semantic = self._semantic(query, k, type)
        text = self._text(query, k, type)
        scores: dict[str, float] = {}
        concepts: dict[str, Concept] = {}
        for ranked in (semantic, text):
            for rank, hit in enumerate(ranked):
                cid = hit.concept.id
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
                concepts.setdefault(cid, hit.concept)
        fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [Hit(concepts[cid], score, "hybrid") for cid, score in fused]

    # --- context assembly --------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Default token estimate: ~4 characters per token (no tokenizer dep)."""
        return max(1, len(text) // 4)

    def _render_block(self, concept: Concept, include_citations: bool) -> str:
        """Render one concept as a titled, optionally-cited context block."""
        title = concept.title or concept.id
        lines = [f"### {title}  ·  {concept.type}  ·  {concept.id}"]
        if concept.description:
            lines.append(concept.description)
        body = concept.body.strip()
        if body:
            lines.append(body)
        if include_citations:
            sources = self._format_sources(concept.cites())
            if sources:
                lines.append(f"Sources: {sources}")
        return "\n".join(lines)

    @staticmethod
    def _format_sources(cites: list[dict]) -> str:
        parts: list[str] = []
        for c in cites:
            target = c.get("url") or c.get("concept")
            if not target:
                continue
            anchor = c.get("anchor")
            parts.append(f"{anchor} <{target}>" if anchor else str(target))
        return "; ".join(parts)

    @staticmethod
    def _truncate(text: str, budget_tokens: int, count: Callable[[str], int]) -> str:
        """Trim ``text`` to roughly ``budget_tokens``, by character ratio."""
        if budget_tokens <= 0:
            return ""
        total = count(text)
        if total <= budget_tokens:
            return text
        keep = max(1, int(len(text) * budget_tokens / total) - 1)
        return text[:keep].rstrip() + " …"

    def _collect_citations(self, concepts: list[Concept]) -> list[dict]:
        """Citations of the included concepts, de-duplicated, with ``cited_by``."""
        merged: dict[tuple, dict] = {}
        for concept in concepts:
            for c in concept.cites():
                key = ("url", c["url"]) if c.get("url") else ("concept", c.get("concept"))
                if key[1] is None:
                    continue
                entry = merged.setdefault(
                    key, {key[0]: key[1], "anchor": c.get("anchor"), "cited_by": []}
                )
                entry["cited_by"].append(concept.id)
        return list(merged.values())

    @staticmethod
    def _read_okf_version(path: str | Path, uri_prefix: str) -> str | None:
        import yaml

        from ..importers.okf import parse_frontmatter

        index = Path(path) / "index.md"
        if not index.exists():
            return None
        try:
            frontmatter, _ = parse_frontmatter(index.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            return None
        version = frontmatter.get("okf_version")
        return str(version) if version is not None else None
