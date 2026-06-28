"""High-level OKF façade over GrafitoDB.

:class:`OKFBundle` groups OKF operations (load, save, concept lookup, layer
navigation, search) behind an OKF-flavored API, while exposing the underlying
:class:`~grafito.database.GrafitoDatabase` via ``bundle.db`` for full graph
power. It *delegates* to the existing ``import_okf_bundle`` / ``export_okf_bundle``
implementations — it does not duplicate them.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from ..importers.okf import (
    DEFAULT_EMBED_FIELDS,
    REFERENCE_LABEL,
    concept_document,
)
from ..models import Node
from .concept import Concept, Hit

if TYPE_CHECKING:
    from ..database import GrafitoDatabase

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_RRF_K = 60  # reciprocal-rank-fusion constant for hybrid search


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
    ) -> None:
        self._db = db
        self._uri_prefix = uri_prefix
        self._embed_index = embed_index
        self._embed_fields = embed_fields
        self._source_path = source_path
        self._okf_version = okf_version
        self._summary = summary or {}

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
        **import_kw: Any,
    ) -> "OKFBundle":
        """Import an OKF bundle and return a façade over it.

        Creates an in-memory database when ``db`` is None.
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
        )

    def save(
        self,
        path: str | Path | None = None,
        *,
        write_index: bool = True,
        write_viz: bool = False,
    ) -> dict:
        """Export the graph back to an OKF bundle (defaults to the load path)."""
        target = str(path) if path is not None else self._source_path
        if target is None:
            raise ValueError("No path to save to; pass one or load the bundle from a path.")
        return self._db.export_okf_bundle(
            target,
            uri_prefix=self._uri_prefix,
            write_index=write_index,
            write_viz=write_viz,
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
        """List concepts, optionally filtered by type/layer/tag."""
        nodes = self._db.match_nodes(labels=[type] if type else None)
        out: list[Concept] = []
        for node in nodes:
            if not self._is_concept(node):
                continue
            cid = node.properties.get("concept_id")
            if not isinstance(cid, str):
                continue
            if layer is not None and self._layer_of(cid) != layer:
                continue
            if tag is not None and tag not in (node.properties.get("tags") or []):
                continue
            out.append(Concept(self, node))
        return out

    def __getitem__(self, concept_id: str) -> Concept:
        concept = self.concept(concept_id)
        if concept is None:
            raise KeyError(concept_id)
        return concept

    def __iter__(self) -> Iterator[Concept]:
        return iter(self.concepts())

    def __len__(self) -> int:
        return len(self.concepts())

    # --- topology ----------------------------------------------------------

    def layers(self) -> dict[str, int]:
        """Top-level concept-id segments and their concept counts."""
        counts: dict[str, int] = {}
        for concept in self.concepts():
            top = self._layer_of(concept.id) or "."
            counts[top] = counts.get(top, 0) + 1
        return counts

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
        subdirs: dict[str, int] = {}
        concepts: list[dict] = []
        for concept in self.concepts():
            cid = concept.id
            if prefix and not cid.startswith(prefix):
                continue
            rest = cid[len(prefix):]
            if "/" in rest:
                child = rest.split("/", 1)[0]
                subdirs[child] = subdirs.get(child, 0) + 1
            elif rest:
                concepts.append(
                    {
                        "id": cid,
                        "title": concept.title or rest,
                        "description": concept.description,
                        "type": concept.type,
                    }
                )
        concepts.sort(key=lambda entry: entry["id"])
        return {"layer": path or None, "subdirs": subdirs, "concepts": concepts}

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
    ) -> list[Hit]:
        """Search concepts by text or meaning, with a unified result shape.

        ``mode``: ``"auto"`` (semantic if embeddings exist, else text),
        ``"semantic"``, ``"text"``, or ``"hybrid"`` (reciprocal-rank fusion).
        """
        if mode == "auto":
            mode = "semantic" if self._has_vector_index() else "text"

        # When filtering by layer we post-filter, so over-fetch to keep k results.
        fetch = k * 4 if layer else k

        if mode == "semantic":
            hits = self._semantic(query, fetch, type)
        elif mode == "text":
            hits = self._text(query, fetch, type)
        elif mode == "hybrid":
            hits = self._hybrid(query, fetch, type)
        else:
            raise ValueError(f"Unknown search mode: {mode!r}")

        if layer is not None:
            hits = [h for h in hits if self._layer_of(h.concept.id) == layer]
        return hits[:k]

    # --- mutation ----------------------------------------------------------

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
        return Concept(self, node)

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
        return self._db.delete_node(node_id)

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
    def _layer_of(concept_id: str) -> str | None:
        return concept_id.split("/", 1)[0] if "/" in concept_id else None

    @staticmethod
    def _is_concept(node: Node) -> bool:
        """True for real concepts (not stubs or auto-created Reference nodes)."""
        return not node.properties.get("stub") and not node.properties.get("okf_auto")

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
        self, concept_id: str, rel_type: str, direction: str, *, depth: int = 1
    ) -> list[Concept]:
        if not _IDENTIFIER_RE.fullmatch(rel_type):
            raise ValueError(f"Invalid relationship type: {rel_type!r}")
        hops = f"*1..{int(depth)}" if depth > 1 else ""
        if direction == "in":
            pattern = f"(a)<-[:{rel_type}{hops}]-(b)"
        else:
            pattern = f"(a)-[:{rel_type}{hops}]->(b)"
        rows = self.execute(
            f"MATCH {pattern} WHERE a.concept_id = $cid AND b.concept_id IS NOT NULL "
            f"RETURN DISTINCT b",
            cid=concept_id,
        )
        out: list[Concept] = []
        seen: set[int] = set()
        for row in rows:
            node = self._node_from_row(row["b"])
            if node.id in seen or node.properties.get("concept_id") == concept_id:
                continue
            seen.add(node.id)
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

    @staticmethod
    def _read_okf_version(path: str | Path, uri_prefix: str) -> str | None:
        from ..importers.okf import parse_frontmatter

        index = Path(path) / "index.md"
        if not index.exists():
            return None
        frontmatter, _ = parse_frontmatter(index.read_text(encoding="utf-8"))
        version = frontmatter.get("okf_version")
        return str(version) if version is not None else None
