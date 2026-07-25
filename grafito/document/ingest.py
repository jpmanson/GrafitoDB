"""DocumentIngestor: document → managed passage graph + retrieval helpers."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Literal

from ..database import GrafitoDatabase
from ..exceptions import DatabaseError, NodeNotFoundError
from ..models import Node
from .chunkers.base import Chunker
from .chunkers.fixed import FixedChunker
from .chunkers.markdown import MarkdownChunker
from .tree import build_markdown_tree, flatten_chunks, sections_to_toc_dict
from .types import (
    ChunkSpec,
    ExpandResult,
    IngestResult,
    PackedContext,
    PackedSegment,
    SearchHit,
    SectionSpec,
)

MANAGED_BY = "grafito.document"
HELPER_SCHEMA_VERSION = 2

DEFAULT_DOCUMENT_LABEL = "Document"
DEFAULT_VERSION_LABEL = "DocumentVersion"
DEFAULT_PASSAGE_LABEL = "Chunk"
DEFAULT_SECTION_LABEL = "Section"
DEFAULT_HAS_VERSION = "HAS_VERSION"
DEFAULT_HAS_PASSAGE = "HAS_CHUNK"
DEFAULT_HAS_SECTION = "HAS_SECTION"


class DocumentIngestor:
    """Ingest text as owned passage nodes with optional vector index + expand/pack.

    Ownership: every managed node has ``managed_by``, ``ingestion_id``,
    ``generation``, ``owner_document_id``, ``role``. External ``parent_id`` is
    never deleted. Replace is generational (BUILDING → ACTIVE; previous STALE).

    External relationships to passages are treated as ephemeral (policy A).
    """

    def __init__(
        self,
        db: GrafitoDatabase,
        chunker: Chunker | None = None,
        *,
        embed_index: str | None = None,
        document_label: str = DEFAULT_DOCUMENT_LABEL,
        version_label: str = DEFAULT_VERSION_LABEL,
        passage_label: str = DEFAULT_PASSAGE_LABEL,
        section_label: str = DEFAULT_SECTION_LABEL,
        has_version_rel: str = DEFAULT_HAS_VERSION,
        has_passage_rel: str = DEFAULT_HAS_PASSAGE,
        has_section_rel: str = DEFAULT_HAS_SECTION,
        write_next_passage: bool = False,
        next_passage_rel: str = "NEXT_PASSAGE",
        corpus: str | None = None,
        store_full_text: bool = True,
        hierarchy: bool | Literal["auto"] = "auto",
        embed_section_summaries: bool = False,
    ) -> None:
        self.db = db
        self.chunker: Chunker = chunker or FixedChunker()
        self.embed_index = embed_index
        self.document_label = document_label
        self.version_label = version_label
        self.passage_label = passage_label
        self.section_label = section_label
        self.has_version_rel = has_version_rel
        self.has_passage_rel = has_passage_rel
        self.has_section_rel = has_section_rel
        self.write_next_passage = write_next_passage
        self.next_passage_rel = next_passage_rel
        self.corpus = corpus
        self.store_full_text = store_full_text
        self.hierarchy = hierarchy
        self.embed_section_summaries = embed_section_summaries

    # ------------------------------------------------------------------
    # Ingest / replace / delete
    # ------------------------------------------------------------------

    def ingest(
        self,
        text: str,
        *,
        document_key: str | None = None,
        title: str | None = None,
        source: str | None = None,
        parent_id: int | None = None,
        properties: dict[str, Any] | None = None,
        embed: bool = True,
        force: bool = False,
    ) -> IngestResult:
        """Create or update a document's active passage version.

        If ``document_key`` matches an existing parent and the fingerprint is
        unchanged (and ``force`` is False), returns ``skipped=True``.
        """
        if parent_id is not None and self.db.get_node(parent_id) is None:
            raise NodeNotFoundError(parent_id)

        parent = self._resolve_parent(
            parent_id=parent_id,
            document_key=document_key,
            title=title,
            source=source,
            properties=properties,
            text=text,
        )
        doc_key = document_key or parent.properties.get("document_key")
        fingerprint = self._fingerprint(text, embed=embed)
        active_gen = int(parent.properties.get("active_generation") or 0)

        if not force and active_gen > 0:
            active_fp = parent.properties.get("active_fingerprint")
            if active_fp == fingerprint:
                passages = self._list_passages(parent.id, active_gen)
                sections = self.db.match_nodes(
                    labels=[self.section_label],
                    properties={
                        "managed_by": MANAGED_BY,
                        "owner_document_id": parent.id,
                        "generation": active_gen,
                        "role": "section",
                    },
                )
                return IngestResult(
                    owner_document_id=parent.id,
                    version_id=self._active_version_id(parent.id),
                    generation=active_gen,
                    passage_ids=[p.id for p in passages],
                    n_passages=len(passages),
                    skipped=True,
                    fingerprint=fingerprint,
                    document_key=doc_key,
                    section_ids=[s.id for s in sections],
                    n_sections=len(sections),
                    hierarchy=bool(sections),
                )

        use_hierarchy = self._use_hierarchy()
        section_forest: list[SectionSpec] | None = None
        if use_hierarchy:
            max_chars = int(getattr(self.chunker, "max_chars", None) or getattr(self.chunker, "max_size", 1200))
            overlap = int(getattr(self.chunker, "overlap", 0) or 0)
            section_forest = build_markdown_tree(
                text,
                max_chars=max_chars,
                overlap=overlap,
                strategy=getattr(self.chunker, "name", "markdown-tree"),
            )
            specs = flatten_chunks(section_forest)
        else:
            specs = self.chunker.split(text)

        generation = active_gen + 1
        ingestion_id = uuid.uuid4().hex
        version = self._create_version(
            parent.id,
            generation=generation,
            ingestion_id=ingestion_id,
            fingerprint=fingerprint,
            status="BUILDING",
            n_passages=len(specs),
            hierarchy=use_hierarchy,
        )
        try:
            section_ids: list[int] = []
            if use_hierarchy and section_forest is not None:
                passage_ids, section_ids = self._write_hierarchy(
                    parent_id=parent.id,
                    version_id=version.id,
                    generation=generation,
                    ingestion_id=ingestion_id,
                    forest=section_forest,
                )
            else:
                passage_ids = self._write_passages(
                    parent_id=parent.id,
                    version_id=version.id,
                    generation=generation,
                    ingestion_id=ingestion_id,
                    specs=specs,
                )
            if embed and self.embed_index and passage_ids:
                # Re-read embed texts from specs order (matches passage creation order)
                docs = [s.text_for_embedding() for s in specs]
                self.db.upsert_embeddings(passage_ids, docs, index=self.embed_index)
            if (
                embed
                and self.embed_index
                and self.embed_section_summaries
                and section_ids
            ):
                self._embed_section_summaries(section_ids)
            self._activate_version(parent, version, fingerprint=fingerprint, generation=generation)
            self._gc_stale(parent.id, keep_generation=generation)
        except Exception:
            self._mark_version_failed(version.id)
            raise

        return IngestResult(
            owner_document_id=parent.id,
            version_id=version.id,
            generation=generation,
            passage_ids=passage_ids,
            n_passages=len(passage_ids),
            skipped=False,
            fingerprint=fingerprint,
            document_key=doc_key,
            section_ids=section_ids,
            n_sections=len(section_ids),
            hierarchy=use_hierarchy,
        )

    def replace(self, document_ref: int | str, text: str, *, embed: bool = True, force: bool = True) -> IngestResult:
        """Re-ingest by owner document id or ``document_key``."""
        parent = self._find_parent(document_ref)
        return self.ingest(
            text,
            parent_id=parent.id,
            document_key=parent.properties.get("document_key"),
            title=parent.properties.get("title"),
            source=parent.properties.get("source"),
            embed=embed,
            force=force,
        )

    def delete(self, document_ref: int | str, *, delete_owned_parent: bool = False) -> bool:
        """Delete helper-managed versions/passages for a document.

        Never deletes an external parent unless ``delete_owned_parent`` and the
        parent was created by this helper (``managed_by`` + role document).
        """
        parent = self._find_parent(document_ref)
        owner_id = parent.id
        managed = self._managed_descendants(owner_id)
        # Embeddings first (batch per index)
        if self.embed_index:
            passage_ids = [
                n.id
                for n in managed
                if n.properties.get("role") == "passage" or self.passage_label in (n.labels or [])
            ]
            if passage_ids:
                try:
                    self.db.remove_embeddings_batch(passage_ids, index=self.embed_index)
                except Exception:
                    for pid in passage_ids:
                        try:
                            self.db.remove_embedding(pid, index=self.embed_index)
                        except Exception:
                            pass
        for n in managed:
            self.db.delete_node(n.id)

        if delete_owned_parent and parent.properties.get("managed_by") == MANAGED_BY:
            if parent.properties.get("role") == "document":
                self.db.delete_node(parent.id)
                return True
        # Clear active generation markers on remaining parent
        if self.db.get_node(owner_id):
            props = dict(parent.properties)
            props.pop("active_generation", None)
            props.pop("active_fingerprint", None)
            props.pop("active_version_id", None)
            self.db.replace_node_properties(owner_id, props)
        return True

    # ------------------------------------------------------------------
    # Search / expand / pack
    # ------------------------------------------------------------------

    def search(
        self,
        query: str | list[float],
        *,
        k: int = 5,
        owner_document_id: int | None = None,
        embed_roles: list[str] | None = None,
        diversify_by_document: bool = False,
    ) -> list[SearchHit]:
        if not self.embed_index:
            raise DatabaseError("DocumentIngestor.embed_index is not set")
        roles = embed_roles or ["passage"]
        # semantic_search matches properties exactly; multi-role → prefilter managed only, then filter
        filter_props: dict[str, Any] = {"managed_by": MANAGED_BY}
        if len(roles) == 1:
            filter_props["embed_role"] = roles[0]
        hits_raw = self.db.semantic_search(
            query,
            k=k * (4 if diversify_by_document or len(roles) > 1 else 1),
            index=self.embed_index,
            filter_labels=[self.passage_label],
            filter_props=filter_props,
        )
        # Restrict to active generation per document
        results: list[SearchHit] = []
        seen_docs: set[int] = set()
        for row in hits_raw:
            node: Node = row["node"]
            props = node.properties or {}
            if props.get("managed_by") != MANAGED_BY:
                continue
            if props.get("embed_role") not in roles:
                continue
            oid = props.get("owner_document_id")
            gen = props.get("generation")
            if owner_document_id is not None and oid != owner_document_id:
                continue
            if oid is not None:
                parent = self.db.get_node(int(oid))
                active = parent.properties.get("active_generation") if parent else None
                if active is not None and gen != active:
                    continue
            if diversify_by_document and oid is not None:
                if int(oid) in seen_docs:
                    continue
                seen_docs.add(int(oid))
            results.append(
                SearchHit(
                    node=node,
                    score=float(row["score"]),
                    owner_document_id=int(oid) if oid is not None else None,
                    generation=int(gen) if gen is not None else None,
                    global_seq=int(props["global_seq"]) if props.get("global_seq") is not None else None,
                )
            )
            if len(results) >= k:
                break
        return results

    def expand(
        self,
        center: Node | int,
        *,
        window: int = 1,
        include_parent: bool = True,
        include_ancestors: bool = False,
    ) -> ExpandResult:
        node = center if isinstance(center, Node) else self.db.get_node(center)
        if node is None:
            raise NodeNotFoundError(center if isinstance(center, int) else getattr(center, "id", -1))
        props = node.properties or {}
        owner_id = props.get("owner_document_id")
        gen = props.get("generation")
        seq = props.get("global_seq")
        if owner_id is None or gen is None or seq is None:
            return ExpandResult(center=node, passages=[node])
        owner_id = int(owner_id)
        gen = int(gen)
        seq = int(seq)
        candidates = self._list_passages(owner_id, gen)
        lo, hi = seq - window, seq + window
        windowed = [
            p
            for p in candidates
            if p.properties.get("global_seq") is not None
            and lo <= int(p.properties["global_seq"]) <= hi
        ]
        windowed.sort(key=lambda p: int(p.properties.get("global_seq") or 0))
        parent = self.db.get_node(owner_id) if include_parent else None
        version = None
        if parent and parent.properties.get("active_version_id"):
            version = self.db.get_node(int(parent.properties["active_version_id"]))
        section = None
        ancestors: list[Node] = []
        section_id = props.get("section_node_id")
        if section_id is not None:
            section = self.db.get_node(int(section_id))
            if include_ancestors and section is not None:
                ancestors = self._section_ancestors(section)
        return ExpandResult(
            center=node,
            passages=windowed or [node],
            parent=parent,
            version=version,
            ancestors=ancestors,
            section=section,
        )

    def toc(self, document_ref: int | str, *, as_dict: bool = False) -> list[SectionSpec] | list[dict[str, Any]]:
        """Return the active version's section tree (titles/summaries, no bodies)."""
        parent = self._find_parent(document_ref)
        gen = int(parent.properties.get("active_generation") or 0)
        if gen <= 0:
            return []
        roots = self._list_root_sections(parent.id, gen)
        forest = [self._section_node_to_spec(s, parent.id, gen) for s in roots]
        if as_dict:
            return sections_to_toc_dict(forest)
        return forest

    def load_sections(
        self,
        document_ref: int | str,
        node_keys: list[str],
    ) -> list[Node]:
        """Load section nodes by per-document ``node_key`` (not global)."""
        parent = self._find_parent(document_ref)
        gen = int(parent.properties.get("active_generation") or 0)
        if gen <= 0 or not node_keys:
            return []
        wanted = set(node_keys)
        sections = self.db.match_nodes(
            labels=[self.section_label],
            properties={
                "managed_by": MANAGED_BY,
                "owner_document_id": parent.id,
                "generation": gen,
                "role": "section",
            },
        )
        by_key = {s.properties.get("node_key"): s for s in sections if s.properties.get("node_key")}
        return [by_key[k] for k in node_keys if k in by_key and k in wanted]

    def pack(
        self,
        nodes: list[Node] | ExpandResult,
        *,
        max_chars: int | None = None,
        max_tokens: int | None = None,
        order: Literal["reading", "score"] = "reading",
        deduplicate_overlap: bool = True,
        include_citations: bool = True,
        scores: dict[int, float] | None = None,
        token_counter: Any | None = None,
    ) -> PackedContext:
        """Pack passages into a budgeted context with optional overlap merge.

        Budget priority: ``token_counter`` + ``max_tokens`` (exact); else
        ``max_tokens`` alone uses a rough ``ceil(len/4)`` estimate; else
        ``max_chars``. Overlap merge prefers parent full text
        (``store_full_text=True``); without it, stitches from passage offsets.
        """
        if isinstance(nodes, ExpandResult):
            node_list = list(nodes.passages)
        else:
            node_list = list(nodes)
        if not node_list:
            return PackedContext(segments=[], text="", order=order)

        if order == "score" and scores:
            node_list = sorted(node_list, key=lambda n: scores.get(n.id, 0.0), reverse=True)
        else:
            node_list = sorted(
                node_list,
                key=lambda n: (
                    int(n.properties.get("global_seq") or 0),
                    n.id,
                ),
            )

        # Build segments (optionally merge overlapping char ranges within same document)
        segments: list[PackedSegment] = []
        if deduplicate_overlap:
            segments = self._merge_overlap_segments(node_list, include_citations=include_citations)
        else:
            for n in node_list:
                p = n.properties or {}
                segments.append(
                    PackedSegment(
                        text=str(p.get("text") or ""),
                        node_id=n.id,
                        document_id=int(p["owner_document_id"]) if p.get("owner_document_id") is not None else None,
                        char_start=p.get("char_start"),
                        char_end=p.get("char_end"),
                        section_path=p.get("section_path"),
                        global_seq=int(p["global_seq"]) if p.get("global_seq") is not None else None,
                    )
                )

        truncated = False
        # Budget: prefer exact token_counter; else max_tokens≈chars via 4 chars/token
        # (design §7.2 rough estimate); else max_chars.
        budget_chars = max_chars
        count_tokens = token_counter
        if max_tokens is not None and count_tokens is None:
            # Rough estimate when no counter is supplied (document which: chars≈4*tokens).
            count_tokens = lambda s: max(1, (len(s) + 3) // 4)  # noqa: E731
            if budget_chars is None:
                budget_chars = max_tokens * 4

        if max_tokens is not None and count_tokens is not None:
            kept, truncated = self._apply_token_budget(segments, max_tokens, count_tokens)
            segments = kept
        elif budget_chars is not None:
            kept, truncated = self._apply_char_budget(segments, budget_chars)
            segments = kept

        if include_citations:
            parts = []
            for seg in segments:
                header = f"[node={seg.node_id}"
                if seg.section_path:
                    header += f" section={seg.section_path}"
                if seg.global_seq is not None:
                    header += f" seq={seg.global_seq}"
                header += "]"
                parts.append(f"{header}\n{seg.text}")
            text = "\n\n".join(parts)
        else:
            text = "\n\n".join(seg.text for seg in segments)

        return PackedContext(segments=segments, text=text, truncated=truncated, order=order)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_parent(
        self,
        *,
        parent_id: int | None,
        document_key: str | None,
        title: str | None,
        source: str | None,
        properties: dict[str, Any] | None,
        text: str,
    ) -> Node:
        if parent_id is not None:
            node = self.db.get_node(parent_id)
            if node is None:
                raise NodeNotFoundError(parent_id)
            # Optionally stamp document_key
            if document_key and node.properties.get("document_key") != document_key:
                props = dict(node.properties)
                props["document_key"] = document_key
                self.db.replace_node_properties(node.id, props)
                node = self.db.get_node(node.id) or node
            return node

        if document_key:
            existing = self.db.match_nodes(
                labels=[self.document_label],
                properties={"document_key": document_key},
                limit=1,
            )
            if existing:
                return existing[0]

        props: dict[str, Any] = {
            "managed_by": MANAGED_BY,
            "role": "document",
        }
        if document_key:
            props["document_key"] = document_key
        if title is not None:
            props["title"] = title
        if source is not None:
            props["source"] = source
        if self.store_full_text:
            props["text"] = text
        if properties:
            props.update(properties)
        uri = f"grafito:document:{document_key}" if document_key else None
        return self.db.create_node(labels=[self.document_label], properties=props, uri=uri)

    def _find_parent(self, document_ref: int | str) -> Node:
        if isinstance(document_ref, int):
            node = self.db.get_node(document_ref)
            if node is None:
                raise NodeNotFoundError(document_ref)
            return node
        found = self.db.match_nodes(
            labels=[self.document_label],
            properties={"document_key": document_ref},
            limit=1,
        )
        if not found:
            raise DatabaseError(f"No document with document_key={document_ref!r}")
        return found[0]

    def _use_hierarchy(self) -> bool:
        if self.hierarchy is True:
            return True
        if self.hierarchy is False:
            return False
        # auto: hierarchical when markdown chunker is configured
        return isinstance(self.chunker, MarkdownChunker)

    def _create_version(
        self,
        owner_id: int,
        *,
        generation: int,
        ingestion_id: str,
        fingerprint: str,
        status: str,
        n_passages: int,
        hierarchy: bool = False,
    ) -> Node:
        props = {
            "managed_by": MANAGED_BY,
            "role": "version",
            "owner_document_id": owner_id,
            "generation": generation,
            "ingestion_id": ingestion_id,
            "fingerprint": fingerprint,
            "status": status,
            "n_passages": n_passages,
            "hierarchy": hierarchy,
            "embed_role": "none",
        }
        if self.corpus:
            props["corpus"] = self.corpus
        version = self.db.create_node(labels=[self.version_label], properties=props)
        self.db.create_relationship(owner_id, version.id, self.has_version_rel, {"generation": generation})
        return version

    def _write_hierarchy(
        self,
        *,
        parent_id: int,
        version_id: int,
        generation: int,
        ingestion_id: str,
        forest: list[SectionSpec],
    ) -> tuple[list[int], list[int]]:
        """Write Section tree + passages. Returns (passage_ids in global order, section_ids)."""
        passage_ids: list[int] = []
        section_ids: list[int] = []
        global_seq = 0

        def write_section(spec: SectionSpec, parent_section_id: int | None) -> int:
            nonlocal global_seq
            props: dict[str, Any] = {
                "title": spec.title,
                "level": spec.level,
                "local_ord": spec.local_ord,
                "node_key": spec.node_key,
                "managed_by": MANAGED_BY,
                "role": "section",
                "embed_role": "section_summary" if spec.summary else "none",
                "owner_document_id": parent_id,
                "generation": generation,
                "ingestion_id": ingestion_id,
                "version_id": version_id,
            }
            if spec.summary is not None:
                props["summary"] = spec.summary
            if spec.char_start is not None:
                props["char_start"] = spec.char_start
            if spec.char_end is not None:
                props["char_end"] = spec.char_end
            if self.corpus:
                props["corpus"] = self.corpus
            section_node = self.db.create_node(labels=[self.section_label], properties=props)
            section_ids.append(section_node.id)
            rel_source = version_id if parent_section_id is None else parent_section_id
            self.db.create_relationship(
                rel_source,
                section_node.id,
                self.has_section_rel,
                {"local_ord": spec.local_ord, "level": spec.level},
            )
            for chunk in spec.chunks:
                chunk.ord = global_seq
                pid = self._write_one_passage(
                    parent_id=parent_id,
                    version_id=version_id,
                    generation=generation,
                    ingestion_id=ingestion_id,
                    spec=chunk,
                    section_node_id=section_node.id,
                )
                passage_ids.append(pid)
                global_seq += 1
            for child in spec.children:
                write_section(child, section_node.id)
            return section_node.id

        for root in forest:
            write_section(root, None)

        if self.write_next_passage and len(passage_ids) > 1:
            for a, b in zip(passage_ids, passage_ids[1:]):
                self.db.create_relationship(a, b, self.next_passage_rel, {})
        return passage_ids, section_ids

    def _write_passages(
        self,
        *,
        parent_id: int,
        version_id: int,
        generation: int,
        ingestion_id: str,
        specs: list[ChunkSpec],
    ) -> list[int]:
        ids: list[int] = []
        for spec in specs:
            ids.append(
                self._write_one_passage(
                    parent_id=parent_id,
                    version_id=version_id,
                    generation=generation,
                    ingestion_id=ingestion_id,
                    spec=spec,
                    section_node_id=None,
                )
            )
        if self.write_next_passage and len(ids) > 1:
            for a, b in zip(ids, ids[1:]):
                self.db.create_relationship(a, b, self.next_passage_rel, {})
        return ids

    def _write_one_passage(
        self,
        *,
        parent_id: int,
        version_id: int,
        generation: int,
        ingestion_id: str,
        spec: ChunkSpec,
        section_node_id: int | None,
    ) -> int:
        props: dict[str, Any] = {
            "text": spec.text,
            "global_seq": spec.ord,
            "managed_by": MANAGED_BY,
            "role": "passage",
            "embed_role": "passage",
            "owner_document_id": parent_id,
            "generation": generation,
            "ingestion_id": ingestion_id,
            "version_id": version_id,
            "strategy": spec.strategy or getattr(self.chunker, "name", None),
        }
        if section_node_id is not None:
            props["section_node_id"] = section_node_id
        if spec.char_start is not None:
            props["char_start"] = spec.char_start
        if spec.char_end is not None:
            props["char_end"] = spec.char_end
        if spec.token_count is not None:
            props["token_count"] = spec.token_count
        if spec.heading is not None:
            props["heading"] = spec.heading
        if spec.section_path is not None:
            props["section_path"] = spec.section_path
        if spec.context is not None:
            props["context"] = spec.context
        if self.corpus:
            props["corpus"] = self.corpus
        node = self.db.create_node(labels=[self.passage_label], properties=props)
        # Link to version for membership
        self.db.create_relationship(
            version_id,
            node.id,
            self.has_passage_rel,
            {"global_seq": spec.ord},
        )
        # Also link to section when hierarchical
        if section_node_id is not None:
            self.db.create_relationship(
                section_node_id,
                node.id,
                self.has_passage_rel,
                {"global_seq": spec.ord},
            )
        return node.id

    def _embed_section_summaries(self, section_ids: list[int]) -> None:
        docs: list[str] = []
        ids: list[int] = []
        for sid in section_ids:
            node = self.db.get_node(sid)
            if not node:
                continue
            summary = node.properties.get("summary")
            title = node.properties.get("title") or ""
            if not summary:
                continue
            docs.append(f"{title}\n\n{summary}".strip())
            ids.append(sid)
        if ids:
            self.db.upsert_embeddings(ids, docs, index=self.embed_index)

    def _list_root_sections(self, owner_id: int, generation: int) -> list[Node]:
        version_id = self._active_version_id(owner_id)
        if version_id is None:
            return []
        # Roots: sections linked from version via HAS_SECTION
        neighbors = self.db.get_neighbors(version_id, direction="outgoing", rel_type=self.has_section_rel)
        roots = [
            n
            for n in neighbors
            if self.section_label in (n.labels or [])
            and n.properties.get("generation") == generation
        ]
        roots.sort(key=lambda n: int(n.properties.get("local_ord") or 0))
        return roots

    def _section_node_to_spec(self, node: Node, owner_id: int, generation: int) -> SectionSpec:
        children_nodes = self.db.get_neighbors(
            node.id, direction="outgoing", rel_type=self.has_section_rel
        )
        children_nodes = [
            c
            for c in children_nodes
            if self.section_label in (c.labels or [])
            and c.properties.get("generation") == generation
        ]
        children_nodes.sort(key=lambda n: int(n.properties.get("local_ord") or 0))
        # Count passages under this section only (direct)
        passage_neighbors = self.db.get_neighbors(
            node.id, direction="outgoing", rel_type=self.has_passage_rel
        )
        n_chunks = sum(1 for p in passage_neighbors if self.passage_label in (p.labels or []))
        spec = SectionSpec(
            title=str(node.properties.get("title") or ""),
            local_ord=int(node.properties.get("local_ord") or 0),
            level=int(node.properties.get("level") or 0),
            summary=node.properties.get("summary"),
            node_key=node.properties.get("node_key"),
            char_start=node.properties.get("char_start"),
            char_end=node.properties.get("char_end"),
            children=[self._section_node_to_spec(c, owner_id, generation) for c in children_nodes],
        )
        # Placeholder chunk count via empty list length not available; store in metadata
        spec.metadata = {"section_node_id": node.id, "n_chunks": n_chunks}
        return spec

    def _section_ancestors(self, section: Node) -> list[Node]:
        """Walk incoming HAS_SECTION until version (exclude version)."""
        chain: list[Node] = []
        current = section
        seen: set[int] = set()
        while current is not None and current.id not in seen:
            seen.add(current.id)
            parents = self.db.get_neighbors(
                current.id, direction="incoming", rel_type=self.has_section_rel
            )
            # Prefer section parent over version
            section_parents = [p for p in parents if self.section_label in (p.labels or [])]
            if not section_parents:
                break
            parent = section_parents[0]
            chain.append(parent)
            current = parent
        chain.reverse()  # root → … → parent of section
        return chain

    def _activate_version(
        self,
        parent: Node,
        version: Node,
        *,
        fingerprint: str,
        generation: int,
    ) -> None:
        # Mark previous active versions STALE
        prev_versions = self.db.match_nodes(
            labels=[self.version_label],
            properties={
                "managed_by": MANAGED_BY,
                "owner_document_id": parent.id,
                "status": "ACTIVE",
            },
        )
        for v in prev_versions:
            props = dict(v.properties)
            props["status"] = "STALE"
            self.db.replace_node_properties(v.id, props)

        vprops = dict(version.properties)
        vprops["status"] = "ACTIVE"
        self.db.replace_node_properties(version.id, vprops)

        pprops = dict(parent.properties)
        pprops["active_generation"] = generation
        pprops["active_fingerprint"] = fingerprint
        pprops["active_version_id"] = version.id
        self.db.replace_node_properties(parent.id, pprops)

    def _mark_version_failed(self, version_id: int) -> None:
        node = self.db.get_node(version_id)
        if not node:
            return
        props = dict(node.properties)
        props["status"] = "FAILED"
        try:
            self.db.replace_node_properties(version_id, props)
        except Exception:
            pass

    def _gc_stale(self, owner_id: int, *, keep_generation: int) -> None:
        managed = self._managed_descendants(owner_id)
        stale_ids = [
            n.id
            for n in managed
            if n.properties.get("generation") is not None
            and int(n.properties["generation"]) != keep_generation
        ]
        if not stale_ids:
            return
        if self.embed_index:
            try:
                self.db.remove_embeddings_batch(stale_ids, index=self.embed_index)
            except Exception:
                for sid in stale_ids:
                    try:
                        self.db.remove_embedding(sid, index=self.embed_index)
                    except Exception:
                        pass
        for sid in stale_ids:
            self.db.delete_node(sid)

    def _managed_descendants(self, owner_id: int) -> list[Node]:
        nodes = self.db.match_nodes(
            properties={"managed_by": MANAGED_BY, "owner_document_id": owner_id},
        )
        return nodes

    def _list_passages(self, owner_id: int, generation: int) -> list[Node]:
        nodes = self.db.match_nodes(
            labels=[self.passage_label],
            properties={
                "managed_by": MANAGED_BY,
                "owner_document_id": owner_id,
                "generation": generation,
                "role": "passage",
            },
        )
        nodes.sort(key=lambda n: int(n.properties.get("global_seq") or 0))
        return nodes

    def _active_version_id(self, owner_id: int) -> int | None:
        parent = self.db.get_node(owner_id)
        if not parent:
            return None
        vid = parent.properties.get("active_version_id")
        return int(vid) if vid is not None else None

    def _fingerprint(self, text: str, *, embed: bool) -> str:
        chunker_name = getattr(self.chunker, "name", type(self.chunker).__name__)
        chunker_params: dict[str, Any] = {}
        for attr in ("max_size", "max_chars", "overlap", "unit"):
            if hasattr(self.chunker, attr):
                chunker_params[attr] = getattr(self.chunker, attr)
        embed_meta: dict[str, Any] = {"embed": embed, "index": self.embed_index}
        if embed and self.embed_index:
            try:
                ef = self.db._get_embedding_function(self.embed_index)
                if ef is not None:
                    embed_meta["embedding_function"] = ef.name()
                    cfg = ef.get_config() if hasattr(ef, "get_config") else {}
                    embed_meta["embedding_config"] = cfg
            except Exception:
                pass
        payload = {
            "schema": HELPER_SCHEMA_VERSION,
            "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "chunker": chunker_name,
            "chunker_params": chunker_params,
            "hierarchy": self._use_hierarchy(),
            "embed_section_summaries": self.embed_section_summaries,
            "embed": embed_meta,
            "corpus": self.corpus,
        }
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _apply_token_budget(
        self,
        segments: list[PackedSegment],
        max_tokens: int,
        token_counter: Any,
    ) -> tuple[list[PackedSegment], bool]:
        kept: list[PackedSegment] = []
        used = 0
        truncated = False
        for seg in segments:
            tc = int(token_counter(seg.text))
            if kept and used + tc > max_tokens:
                truncated = True
                break
            if not kept and tc > max_tokens:
                ratio = max_tokens / max(tc, 1)
                cut = max(1, int(len(seg.text) * ratio))
                kept.append(
                    PackedSegment(
                        text=seg.text[:cut],
                        node_id=seg.node_id,
                        document_id=seg.document_id,
                        char_start=seg.char_start,
                        char_end=(seg.char_start + cut) if seg.char_start is not None else None,
                        section_path=seg.section_path,
                        global_seq=seg.global_seq,
                    )
                )
                truncated = True
                break
            kept.append(seg)
            used += tc
        return kept, truncated

    def _apply_char_budget(
        self,
        segments: list[PackedSegment],
        budget_chars: int,
    ) -> tuple[list[PackedSegment], bool]:
        kept: list[PackedSegment] = []
        used = 0
        truncated = False
        for seg in segments:
            if kept and used + len(seg.text) > budget_chars:
                truncated = True
                break
            if not kept and len(seg.text) > budget_chars:
                kept.append(
                    PackedSegment(
                        text=seg.text[:budget_chars],
                        node_id=seg.node_id,
                        document_id=seg.document_id,
                        char_start=seg.char_start,
                        char_end=(seg.char_start + budget_chars) if seg.char_start is not None else None,
                        section_path=seg.section_path,
                        global_seq=seg.global_seq,
                    )
                )
                truncated = True
                break
            kept.append(seg)
            used += len(seg.text)
        return kept, truncated

    def _merge_overlap_segments(
        self,
        nodes: list[Node],
        *,
        include_citations: bool,
    ) -> list[PackedSegment]:
        """Merge overlapping passage ranges for pack.

        Exact slice merge needs the parent document body (``store_full_text=True``,
        the default). With ``store_full_text=False``, text is reconstructed by
        stitching passage texts via ``char_start``/``char_end`` (works when offsets
        are consistent; still weaker than parent slicing if offsets drift).
        """
        # Group by owner_document_id; merge overlapping [char_start, char_end)
        by_doc: dict[int | None, list[Node]] = {}
        for n in nodes:
            oid = n.properties.get("owner_document_id")
            key = int(oid) if oid is not None else None
            by_doc.setdefault(key, []).append(n)

        segments: list[PackedSegment] = []
        for oid, group in by_doc.items():
            group = sorted(
                group,
                key=lambda n: (
                    int(n.properties.get("char_start") or 0),
                    int(n.properties.get("global_seq") or 0),
                ),
            )
            # If offsets missing, fall back to concat without merge
            if any(n.properties.get("char_start") is None for n in group):
                for n in group:
                    p = n.properties or {}
                    segments.append(
                        PackedSegment(
                            text=str(p.get("text") or ""),
                            node_id=n.id,
                            document_id=oid,
                            char_start=p.get("char_start"),
                            char_end=p.get("char_end"),
                            section_path=p.get("section_path"),
                            global_seq=int(p["global_seq"]) if p.get("global_seq") is not None else None,
                        )
                    )
                continue

            # Prefer parent full text if available for accurate slice merge
            full_text = None
            if oid is not None:
                parent = self.db.get_node(oid)
                if parent and isinstance(parent.properties.get("text"), str):
                    full_text = parent.properties["text"]

            ranges: list[tuple[int, int, int, int | None, str | None]] = []
            # (start, end, node_id, global_seq, section_path)
            for n in group:
                p = n.properties
                a, b = int(p["char_start"]), int(p["char_end"])
                if not ranges:
                    ranges.append((a, b, n.id, p.get("global_seq"), p.get("section_path")))
                    continue
                la, lb, lid, lseq, lpath = ranges[-1]
                if a <= lb:
                    ranges[-1] = (la, max(lb, b), lid, lseq, lpath or p.get("section_path"))
                else:
                    ranges.append((a, b, n.id, p.get("global_seq"), p.get("section_path")))

            for a, b, nid, gseq, spath in ranges:
                if full_text is not None and 0 <= a <= b <= len(full_text):
                    piece = full_text[a:b]
                else:
                    piece = self._stitch_range_from_passages(group, a, b)
                segments.append(
                    PackedSegment(
                        text=piece,
                        node_id=nid,
                        document_id=oid,
                        char_start=a,
                        char_end=b,
                        section_path=spath,
                        global_seq=int(gseq) if gseq is not None else None,
                    )
                )
        segments.sort(key=lambda s: (s.global_seq is None, s.global_seq or 0, s.node_id))
        return segments

    @staticmethod
    def _stitch_range_from_passages(group: list[Node], a: int, b: int) -> str:
        """Rebuild [a, b) by walking passage texts and offsets (no parent body)."""
        parts: list[str] = []
        cursor = a
        ordered = sorted(group, key=lambda n: int(n.properties.get("char_start") or 0))
        for n in ordered:
            p = n.properties or {}
            cs, ce = int(p["char_start"]), int(p["char_end"])
            if ce <= cursor or cs >= b:
                continue
            text = str(p.get("text") or "")
            take_start = max(cs, cursor)
            take_end = min(ce, b)
            if take_start >= take_end:
                continue
            local0 = take_start - cs
            local1 = take_end - cs
            if local0 < 0 or local1 > len(text):
                # Offsets inconsistent with stored text length — best effort
                local0 = max(0, min(local0, len(text)))
                local1 = max(local0, min(local1, len(text)))
            parts.append(text[local0:local1])
            cursor = take_end
            if cursor >= b:
                break
        return "".join(parts)
