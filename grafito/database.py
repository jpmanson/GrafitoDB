"""Core database class for Grafito graph database."""

import orjson
import os
import re
import sqlite3
from datetime import date, datetime, time, timedelta, timezone
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable

from .exceptions import (
    DatabaseError,
    ConstraintError,
    InvalidPropertyError,
    InvalidFilterError,
    NodeNotFoundError,
    RelationshipNotFoundError,
)
from .filters import PropertyFilter, PropertyFilterGroup, LabelFilter, SortOrder
from .models import Node, Relationship
from .vector_index import BruteForceIndex
from .indexers import Indexer
from .embedding_functions import create_embedding_function, EmbeddingFunction
from .query import PathFinder
from .schema import initialize_schema

#: Neighbour-score floor create_semantic_graph applies when none is given.
#: Lower than the SIMILAR() default (0.5) on purpose: SIMILAR() answers "is this
#: similar?", where a permissive threshold produces wrong answers, while here `k`
#: already bounds the result and min_score is a quality floor on the tail.
#: Matches txtai's minscore default.
DEFAULT_SEMANTIC_GRAPH_MIN_SCORE = 0.1

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .algorithms import Community
    from .ingest_report import IndexReport, SemanticGraphReport
    from .subgraph import Subgraph


class GrafitoDatabase:
    """SQLite-based property graph database.

    This class implements the Property Graph Model using SQLite as the storage backend.
    It supports nodes with multiple labels, directed relationships, and properties on both.

    Example:
        >>> db = GrafitoDatabase(':memory:')
        >>> person = db.create_node(labels=['Person'], properties={'name': 'Alice'})
        >>> company = db.create_node(labels=['Company'], properties={'name': 'TechCorp'})
        >>> rel = db.create_relationship(person.id, company.id, 'WORKS_AT')
    """

    def __init__(
        self,
        db_path: str = ':memory:',
        cypher_max_hops: int = 5,
        default_top_k: int = 10,
        sql_trace: bool = False,
    ):
        """Initialize the graph database.

        Args:
            db_path: Path to the SQLite database file. Use ':memory:' for in-memory database.
            cypher_max_hops: Default max hops for unbounded Cypher variable-length paths.

        Raises:
            DatabaseError: If database initialization fails
        """
        if cypher_max_hops <= 0:
            raise DatabaseError("cypher_max_hops must be a positive integer")
        if default_top_k <= 0:
            raise DatabaseError("default_top_k must be a positive integer")
        try:
            self.conn = sqlite3.connect(db_path)
            # Kept so vector-index persistence can tell a durable, file-backed
            # database (vectors must survive reopen) from an ephemeral one
            # (":memory:" / "" — a private temp db that cannot be reopened).
            self._db_path = db_path
            self.conn.row_factory = sqlite3.Row  # Access columns by name
            self.conn.execute("PRAGMA foreign_keys = ON")  # Enable CASCADE
            if sql_trace:
                self.conn.set_trace_callback(lambda stmt: print(f"[SQL] {stmt}", flush=True))
            initialize_schema(self.conn)
            self._in_transaction = False
            self.cypher_max_hops = cypher_max_hops
            self._vector_indexes: dict[str, BruteForceIndex] = {}
            self._vector_index_embeddings: dict[str, EmbeddingFunction] = {}
            self._embedding_functions: dict[str, EmbeddingFunction] = {}
            self.default_top_k = default_top_k
            self._rerankers: dict[str, Any] = {}
            self._text_indexes: dict[str, Any] = {}  # Custom text index backends

            # Register custom SQL functions
            self._register_custom_functions()
        except Exception as e:
            raise DatabaseError(f"Failed to initialize database: {e}", e)

    def _register_custom_functions(self) -> None:
        """Register custom SQLite functions for advanced queries."""

        def sqlite_regex(pattern: str, value: str) -> int:
            """Custom SQLite function for regex matching.

            Args:
                pattern: Regular expression pattern
                value: Value to match against

            Returns:
                1 if match, 0 otherwise
            """
            if value is None:
                return 0
            try:
                return 1 if re.search(pattern, str(value)) else 0
            except re.error:
                return 0

        self.conn.create_function('regex', 2, sqlite_regex)

    def has_fts5(self) -> bool:
        """Check whether SQLite FTS5 is available."""
        try:
            self.conn.execute("CREATE VIRTUAL TABLE fts5_check USING fts5(content)")
            self.conn.execute("DROP TABLE fts5_check")
            return True
        except sqlite3.OperationalError:
            return False

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            for vec_index in self._vector_indexes.values():
                try:
                    if hasattr(vec_index, "unload"):
                        vec_index.unload()
                except Exception:
                    pass
            self.conn.close()
        self._vector_indexes = {}

    # =========================================================================
    # Integrations
    # =========================================================================

    def to_networkx(self, directed: bool = True):
        """Export the graph to a NetworkX graph."""
        try:
            import networkx as nx
        except ImportError as exc:
            raise DatabaseError(
                "networkx is not installed. Install with `pip install networkx`."
            ) from exc

        graph = nx.MultiDiGraph() if directed else nx.MultiGraph()

        cursor = self.conn.execute("SELECT id, properties, uri FROM nodes ORDER BY id")
        for row in cursor.fetchall():
            node_id = int(row["id"])
            properties = orjson.loads(row["properties"])
            labels = self._get_node_labels(node_id)
            graph.add_node(
                node_id,
                labels=labels,
                properties=properties,
                uri=row["uri"],
            )

        cursor = self.conn.execute(
            """
            SELECT id, source_node_id, target_node_id, type, properties, uri
            FROM relationships
            ORDER BY id
            """
        )
        for row in cursor.fetchall():
            rel_id = int(row["id"])
            source_id = int(row["source_node_id"])
            target_id = int(row["target_node_id"])
            rel_type = row["type"]
            properties = orjson.loads(row["properties"])
            graph.add_edge(
                source_id,
                target_id,
                key=rel_id,
                id=rel_id,
                type=rel_type,
                properties=properties,
                uri=row["uri"],
            )

        return graph

    def to_analysis_graph(
        self,
        *,
        rel_types: list[str] | None = None,
        exclude_rel_types: list[str] | None = None,
        labels: list[str] | None = None,
        directed: bool = True,
        weight_property: str | None = None,
    ):
        """Export a filtered NetworkX graph for analysis.

        Unlike :meth:`to_networkx`, which mirrors the whole database, this
        selects the subgraph an analysis should actually run over. That
        distinction matters as soon as a database mixes edge kinds: derived or
        bulk-generated edges (similarity links, containment) will dominate any
        centrality or community result unless they are excluded, and the
        exclusion has to happen before the algorithm runs, not after.

        Args:
            rel_types: Keep only these relationship types.
            exclude_rel_types: Drop these relationship types.
            labels: Keep only nodes carrying at least one of these labels.
                Relationships with an endpoint outside the selection are dropped.
            directed: Whether to build a directed graph.
            weight_property: Relationship property to promote to the ``weight``
                edge attribute, for weighted algorithms. Edges missing it get
                weight 1.0.

        Returns:
            A NetworkX ``MultiDiGraph`` or ``MultiGraph``.
        """
        try:
            import networkx as nx
        except ImportError as exc:
            raise DatabaseError(
                "networkx is not installed. Install with `pip install networkx`."
            ) from exc

        if rel_types and exclude_rel_types:
            overlap = set(rel_types) & set(exclude_rel_types)
            if overlap:
                raise DatabaseError(
                    "rel_types and exclude_rel_types overlap on: "
                    + ", ".join(sorted(overlap))
                )

        graph = nx.MultiDiGraph() if directed else nx.MultiGraph()

        node_sql = "SELECT id, properties, uri FROM nodes"
        node_params: list[Any] = []
        if labels:
            placeholders = ", ".join("?" for _ in labels)
            node_sql += f"""
                WHERE id IN (
                    SELECT nl.node_id FROM node_labels nl
                    JOIN labels l ON l.id = nl.label_id
                    WHERE l.name IN ({placeholders})
                )
            """
            node_params.extend(labels)
        node_sql += " ORDER BY id"

        selected: set[int] = set()
        for row in self.conn.execute(node_sql, node_params).fetchall():
            node_id = int(row["id"])
            selected.add(node_id)
            graph.add_node(
                node_id,
                labels=self._get_node_labels(node_id),
                properties=orjson.loads(row["properties"]),
                uri=row["uri"],
            )

        rel_sql = """
            SELECT id, source_node_id, target_node_id, type, properties, uri
            FROM relationships
        """
        conditions: list[str] = []
        rel_params: list[Any] = []
        if rel_types:
            conditions.append(f"type IN ({', '.join('?' for _ in rel_types)})")
            rel_params.extend(rel_types)
        if exclude_rel_types:
            conditions.append(
                f"type NOT IN ({', '.join('?' for _ in exclude_rel_types)})"
            )
            rel_params.extend(exclude_rel_types)
        if conditions:
            rel_sql += " WHERE " + " AND ".join(conditions)
        rel_sql += " ORDER BY id"

        for row in self.conn.execute(rel_sql, rel_params).fetchall():
            source_id = int(row["source_node_id"])
            target_id = int(row["target_node_id"])
            if source_id not in selected or target_id not in selected:
                continue
            properties = orjson.loads(row["properties"])
            attrs: dict[str, Any] = {
                "id": int(row["id"]),
                "type": row["type"],
                "properties": properties,
                "uri": row["uri"],
            }
            if weight_property is not None:
                raw = properties.get(weight_property)
                try:
                    attrs["weight"] = 1.0 if raw is None else float(raw)
                except (TypeError, ValueError):
                    raise DatabaseError(
                        f"Relationship {row['id']} has a non-numeric "
                        f"'{weight_property}' property: {raw!r}"
                    ) from None
            graph.add_edge(source_id, target_id, key=int(row["id"]), **attrs)

        return graph

    def centrality(
        self,
        kind: str = "pagerank",
        *,
        rel_types: list[str] | None = None,
        exclude_rel_types: list[str] | None = None,
        labels: list[str] | None = None,
        directed: bool = True,
        weight_property: str | None = None,
        limit: int | None = None,
        graph: Any | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Rank nodes by a centrality measure.

        Supported ``kind`` values: ``pagerank``, ``degree``, ``in_degree``,
        ``out_degree``, ``betweenness``, ``closeness``, ``harmonic``,
        ``eigenvector``.

        ``betweenness`` is O(V·E) — on graphs beyond a few thousand nodes, pass
        NetworkX's ``k`` sampling parameter through ``**kwargs``, or prefer
        ``pagerank``.

        Args:
            kind: The centrality measure.
            rel_types, exclude_rel_types, labels, directed, weight_property:
                Passed to :meth:`to_analysis_graph` to scope the analysis.
            limit: Return only the top N nodes. ``None`` returns all of them,
                which materialises one :class:`~grafito.models.Node` per node in
                the graph.
            graph: Analyse this pre-built NetworkX graph instead of exporting
                one, e.g. a subgraph from :meth:`semantic_subgraph`.
            **kwargs: Passed to the underlying NetworkX function.

        Returns:
            ``[{"node": Node, "score": float}, ...]``, highest score first.
        """
        from .algorithms import compute_centrality

        if graph is None:
            graph = self.to_analysis_graph(
                rel_types=rel_types,
                exclude_rel_types=exclude_rel_types,
                labels=labels,
                directed=directed,
                weight_property=weight_property,
            )
        scores = compute_centrality(
            graph,
            kind,
            weight="weight" if weight_property else None,
            **kwargs,
        )
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if limit is not None:
            ranked = ranked[:limit]

        results = []
        for node_id, score in ranked:
            node = self.get_node(node_id)
            if node is not None:
                results.append({"node": node, "score": float(score)})
        return results

    def communities(
        self,
        algorithm: str = "louvain",
        *,
        rel_types: list[str] | None = None,
        exclude_rel_types: list[str] | None = None,
        labels: list[str] | None = None,
        weight_property: str | None = None,
        resolution: float = 1.0,
        seed: int | None = None,
        min_size: int = 1,
        graph: Any | None = None,
        **kwargs: Any,
    ) -> list["Community"]:
        """Partition the graph into communities, largest first.

        Supported ``algorithm`` values: ``louvain``, ``greedy``, ``lpa``
        (``label_propagation``).

        Direction is dropped — modularity is defined on undirected graphs — and
        parallel edges collapse into one weighted edge.

        ``louvain`` and ``lpa`` are randomised: pass ``seed`` for a reproducible
        partition. Community ``id`` values are positions in the returned list,
        so they are not stable across runs or across edits to the graph.

        Args:
            algorithm: The detection algorithm.
            rel_types, exclude_rel_types, labels, weight_property:
                Passed to :meth:`to_analysis_graph` to scope the analysis.
            resolution: Higher values produce more, smaller communities
                (``louvain`` and ``greedy`` only).
            seed: Seed for the randomised algorithms.
            min_size: Drop communities smaller than this. Singletons are noise
                in most datasets; raise it to skip them.
            graph: Analyse this pre-built NetworkX graph instead of exporting one.
            **kwargs: Passed to the underlying NetworkX function.

        Returns:
            A list of :class:`~grafito.algorithms.Community`.
        """
        from .algorithms import Community, detect_communities

        if graph is None:
            graph = self.to_analysis_graph(
                rel_types=rel_types,
                exclude_rel_types=exclude_rel_types,
                labels=labels,
                directed=False,
                weight_property=weight_property,
            )
        groups = detect_communities(
            graph,
            algorithm,
            weight="weight" if weight_property else None,
            resolution=resolution,
            seed=seed,
            **kwargs,
        )

        result: list[Community] = []
        for group in groups:
            if len(group) < min_size:
                continue
            nodes = [node for node in (self.get_node(nid) for nid in sorted(group)) if node]
            if not nodes:
                continue
            result.append(Community(id=len(result), nodes=nodes, size=len(nodes)))
        return result

    def subgraph(
        self,
        seeds: "list[int] | list[Node] | list[dict[str, Any]]",
        *,
        expand: int = 0,
        direction: str = "both",
        rel_types: list[str] | None = None,
        exclude_rel_types: list[str] | None = None,
        labels: list[str] | None = None,
        max_nodes: int | None = None,
        include_edges: bool = True,
    ) -> "Subgraph":
        """Build the induced subgraph around a set of seed nodes.

        Expansion is breadth-first: every node within ``expand`` hops of a seed
        is pulled in, and — when ``include_edges`` — *all* relationships between
        the selected nodes are returned, not only the ones traversed. That is
        what makes the result a graph rather than a tree: two seeds that link to
        each other directly show that link even if neither was reached from the
        other.

        Args:
            seeds: Node ids, :class:`~grafito.models.Node` objects, or search
                hits (``{"node": Node, "score": float}``). Scores, when present,
                are carried onto the result.
            expand: Hops of neighbourhood to include. ``0`` keeps only the seeds
                and the edges among them.
            direction: ``"both"`` (default), ``"out"``, or ``"in"`` — which way
                expansion follows relationships.
            rel_types: Traverse and return only these relationship types.
            exclude_rel_types: Never traverse or return these types. Use this to
                keep derived edges (similarity links) out of an expansion that
                would otherwise reach the whole database in one hop.
            labels: Only expand into nodes carrying one of these labels. Seeds
                are kept regardless.
            max_nodes: Stop expanding once the subgraph reaches this many nodes.
                Guards against a dense hub turning ``expand=2`` into a full
                table scan; the truncation is visible as missing hops.
            include_edges: Fetch the relationships. ``False`` returns nodes only.

        Returns:
            A :class:`~grafito.subgraph.Subgraph`.
        """
        from .subgraph import Subgraph

        if direction not in {"both", "out", "in"}:
            raise DatabaseError("direction must be 'both', 'out', or 'in'")
        if expand < 0:
            raise DatabaseError("expand must be >= 0")
        if max_nodes is not None and max_nodes <= 0:
            raise DatabaseError("max_nodes must be a positive integer")
        if rel_types and exclude_rel_types:
            overlap = set(rel_types) & set(exclude_rel_types)
            if overlap:
                raise DatabaseError(
                    "rel_types and exclude_rel_types overlap on: "
                    + ", ".join(sorted(overlap))
                )

        hits: list[dict[str, Any]] = []
        scores: dict[int, float] = {}
        seed_ids: list[int] = []
        for seed in seeds:
            if isinstance(seed, dict):
                node = seed.get("node") or seed.get("entity")
                score = seed.get("score")
            else:
                node, score = seed, None
            node_id = node.id if hasattr(node, "id") else int(node)
            if node_id in scores or node_id in seed_ids:
                continue
            resolved = node if hasattr(node, "id") else self.get_node(node_id)
            if resolved is None:
                continue
            seed_ids.append(node_id)
            hits.append({"node": resolved, "score": score})
            if score is not None:
                scores[node_id] = float(score)

        hops: dict[int, int] = {node_id: 0 for node_id in seed_ids}
        selected: list[int] = list(seed_ids)

        allowed_labels: set[int] | None = None
        if labels:
            placeholders = ", ".join("?" for _ in labels)
            rows = self.conn.execute(
                f"""
                SELECT nl.node_id FROM node_labels nl
                JOIN labels l ON l.id = nl.label_id
                WHERE l.name IN ({placeholders})
                """,
                labels,
            ).fetchall()
            allowed_labels = {int(row["node_id"]) for row in rows}

        frontier = list(seed_ids)
        for hop in range(1, expand + 1):
            if not frontier or (max_nodes is not None and len(selected) >= max_nodes):
                break
            neighbours = self._expand_frontier(
                frontier, direction, rel_types, exclude_rel_types
            )
            next_frontier: list[int] = []
            for node_id in neighbours:
                if node_id in hops:
                    continue
                if allowed_labels is not None and node_id not in allowed_labels:
                    continue
                if max_nodes is not None and len(selected) >= max_nodes:
                    break
                hops[node_id] = hop
                selected.append(node_id)
                next_frontier.append(node_id)
            frontier = next_frontier

        nodes = [node for node in (self.get_node(nid) for nid in selected) if node]
        present = {node.id for node in nodes}

        relationships: list[Relationship] = []
        if include_edges and present:
            relationships = self._relationships_within(
                present, rel_types, exclude_rel_types
            )

        return Subgraph(
            nodes=nodes,
            relationships=relationships,
            seeds=hits,
            scores=scores,
            hops={nid: hop for nid, hop in hops.items() if nid in present},
        )

    def _expand_frontier(
        self,
        frontier: list[int],
        direction: str,
        rel_types: list[str] | None,
        exclude_rel_types: list[str] | None,
    ) -> list[int]:
        """One breadth-first step: the neighbours of ``frontier``, in id order."""
        conditions: list[str] = []
        params: list[Any] = []
        placeholders = ", ".join("?" for _ in frontier)
        if direction == "out":
            conditions.append(f"source_node_id IN ({placeholders})")
            params.extend(frontier)
        elif direction == "in":
            conditions.append(f"target_node_id IN ({placeholders})")
            params.extend(frontier)
        else:
            conditions.append(
                f"(source_node_id IN ({placeholders}) OR target_node_id IN ({placeholders}))"
            )
            params.extend(frontier)
            params.extend(frontier)
        if rel_types:
            conditions.append(f"type IN ({', '.join('?' for _ in rel_types)})")
            params.extend(rel_types)
        if exclude_rel_types:
            conditions.append(f"type NOT IN ({', '.join('?' for _ in exclude_rel_types)})")
            params.extend(exclude_rel_types)

        rows = self.conn.execute(
            "SELECT source_node_id, target_node_id FROM relationships "
            f"WHERE {' AND '.join(conditions)} ORDER BY id",
            params,
        ).fetchall()

        frontier_set = set(frontier)
        seen: dict[int, None] = {}
        for row in rows:
            source_id = int(row["source_node_id"])
            target_id = int(row["target_node_id"])
            if direction in {"both", "out"} and source_id in frontier_set:
                seen.setdefault(target_id)
            if direction in {"both", "in"} and target_id in frontier_set:
                seen.setdefault(source_id)
        return list(seen)

    def _relationships_within(
        self,
        node_ids: set[int],
        rel_types: list[str] | None,
        exclude_rel_types: list[str] | None,
    ) -> list[Relationship]:
        """Every relationship with both endpoints inside ``node_ids``."""
        ids = list(node_ids)
        placeholders = ", ".join("?" for _ in ids)
        conditions = [
            f"source_node_id IN ({placeholders})",
            f"target_node_id IN ({placeholders})",
        ]
        params: list[Any] = ids + ids
        if rel_types:
            conditions.append(f"type IN ({', '.join('?' for _ in rel_types)})")
            params.extend(rel_types)
        if exclude_rel_types:
            conditions.append(f"type NOT IN ({', '.join('?' for _ in exclude_rel_types)})")
            params.extend(exclude_rel_types)

        rows = self.conn.execute(
            """
            SELECT id, source_node_id, target_node_id, type, properties, uri
            FROM relationships
            """
            f" WHERE {' AND '.join(conditions)} ORDER BY id",
            params,
        ).fetchall()
        return [
            Relationship(
                id=int(row["id"]),
                source_id=int(row["source_node_id"]),
                target_id=int(row["target_node_id"]),
                type=row["type"],
                properties=orjson.loads(row["properties"]),
                uri=row["uri"],
            )
            for row in rows
        ]

    def semantic_subgraph(
        self,
        query: list[float] | str,
        k: int | None = None,
        index: str = "default",
        *,
        expand: int = 1,
        direction: str = "both",
        rel_types: list[str] | None = None,
        exclude_rel_types: list[str] | None = None,
        labels: list[str] | None = None,
        max_nodes: int | None = None,
        include_edges: bool = True,
        filter_labels: list[str] | LabelFilter | None = None,
        filter_props: dict[str, Any] | PropertyFilterGroup | None = None,
        **search_kwargs: Any,
    ) -> "Subgraph":
        """Search semantically, then return the hits *and how they connect*.

        The graph counterpart of :meth:`semantic_search`: same retrieval, but
        the result carries the relationships among the hits and their
        neighbourhood, ready for visualisation or graph analysis.

        Args:
            query: Query string (embedded via the index) or a vector.
            k: Number of seed hits.
            index: Vector index to search.
            expand: Hops of neighbourhood around the hits. ``0`` returns just
                the hits and the edges between them.
            filter_labels, filter_props: Constrain *retrieval*, as in
                :meth:`semantic_search`.
            labels, rel_types, exclude_rel_types, direction, max_nodes,
                include_edges: Constrain *expansion*, as in :meth:`subgraph`.
            **search_kwargs: Forwarded to :meth:`semantic_search` (``rerank``,
                ``reranker``, ``exact``, ``candidate_multiplier``).

        Returns:
            A :class:`~grafito.subgraph.Subgraph`.
        """
        hits = self.semantic_search(
            query,
            k=k,
            index=index,
            filter_labels=filter_labels,
            filter_props=filter_props,
            **search_kwargs,
        )
        return self.subgraph(
            hits,
            expand=expand,
            direction=direction,
            rel_types=rel_types,
            exclude_rel_types=exclude_rel_types,
            labels=labels,
            max_nodes=max_nodes,
            include_edges=include_edges,
        )

    def text_subgraph(
        self,
        query: str,
        k: int | None = None,
        *,
        expand: int = 1,
        direction: str = "both",
        rel_types: list[str] | None = None,
        exclude_rel_types: list[str] | None = None,
        labels: list[str] | None = None,
        max_nodes: int | None = None,
        include_edges: bool = True,
        search_labels: list[str] | None = None,
    ) -> "Subgraph":
        """Full-text search (FTS5/BM25), returned as an induced subgraph.

        The lexical counterpart of :meth:`semantic_subgraph`. Relationship hits
        from :meth:`text_search` are ignored — only matching nodes seed the
        subgraph.

        Args:
            query: FTS5 query string.
            k: Number of seed hits.
            search_labels: Restrict the text search to these labels.
            expand, direction, rel_types, exclude_rel_types, labels, max_nodes,
                include_edges: As in :meth:`subgraph`.

        Returns:
            A :class:`~grafito.subgraph.Subgraph`.
        """
        hits = self.text_search(query, k=k, labels=search_labels)
        seeds = [
            {"node": hit["entity"], "score": hit["score"]}
            for hit in hits
            if hit.get("entity_type") == "node"
        ]
        return self.subgraph(
            seeds,
            expand=expand,
            direction=direction,
            rel_types=rel_types,
            exclude_rel_types=exclude_rel_types,
            labels=labels,
            max_nodes=max_nodes,
            include_edges=include_edges,
        )

    def index_documents(
        self,
        rows: "Iterable[dict[str, Any]]",
        *,
        label: str = "Document",
        id_key: str | None = "id",
        text_key: str = "text",
        index: str | None = "default",
        relationships_key: str | None = None,
        default_rel_type: str = "RELATED_TO",
        copy_attributes: bool = True,
        properties: list[str] | None = None,
        batch_size: int = 256,
        upsert: bool = True,
        configure_fts: bool = False,
    ) -> "IndexReport":
        """Ingest a collection of documents as nodes, edges, and embeddings.

        This is the migration path from row-shaped data — a HuggingFace dataset,
        a DataFrame's ``to_dict("records")``, a JSONL file — into the graph. It
        does in one call what would otherwise be a loop of
        :meth:`create_node` / :meth:`create_relationship` /
        :meth:`upsert_embeddings_batch`, and it embeds in batches rather than
        one text at a time.

        Nodes are created first and relationships afterwards, in a second pass,
        so a row may reference a document that appears later in ``rows``.

        ``relationships_key`` accepts either form per row::

            {"id": "1", "text": "...", "links": ["2", "3"]}
            {"id": "1", "text": "...", "links": [{"id": "2", "type": "CITES"}]}

        The mapping form also takes ``properties``. Targets are resolved against
        ``id_key`` values; a reference to an id not present in ``rows`` (and not
        already in the database) is skipped and counted in
        :attr:`IndexReport.unresolved`.

        Args:
            rows: Iterable of dicts. Consumed once.
            label: Label applied to every created node.
            id_key: Key holding each row's external id. ``None`` means the rows
                have no identity: nodes are always created and relationships
                cannot be resolved.
            text_key: Key holding the text to embed. Rows missing it are stored
                but not embedded.
            index: Vector index to write embeddings to. ``None`` skips embedding
                entirely.
            relationships_key: Key holding this row's outgoing relationships.
            default_rel_type: Type used for bare-id references.
            copy_attributes: Copy every other key in the row onto the node's
                properties. When ``False``, only ``id_key``/``text_key`` (and
                anything in ``properties``) are kept.
            properties: Explicit allowlist of keys to copy. Overrides
                ``copy_attributes``.
            batch_size: Rows embedded per call to the embedding function.
            upsert: When an ``id_key`` value already exists under ``label``,
                update that node instead of creating a duplicate.
            configure_fts: Also register a full-text index over
                ``label``/``text_key``, so the documents are reachable by
                :meth:`text_search` and hybrid retrieval without a second call.
                Registered *before* loading, so rows are indexed as they are
                written. If this call registered it and the ingest then fails,
                the registration is undone; an index that already existed is
                left alone. Off by default: it creates an index you did not ask
                for.

        Returns:
            An :class:`~grafito.ingest_report.IndexReport`.

        Warning:
            This is **not atomic**. Rows are written as they are read, so a
            failure part-way through — a bad row, an embedding provider
            timing out — leaves the rows before it in the database. That is
            deliberate: the alternative is holding a write transaction open
            across every embedding call, which on a hosted embedder means
            blocking writers for minutes.

            To resume, re-run with the remaining rows; with ``upsert=True``
            (the default) re-feeding rows that already landed updates them
            instead of duplicating. Wrap the call in
            :meth:`begin_transaction`/:meth:`rollback` yourself if you need
            all-or-nothing and your embedder is local and fast.

        Note:
            Without ``configure_fts`` this populates the vector index only. You
            can call :meth:`create_text_index` yourself instead, but do it
            *before* loading: it registers the configuration and indexes rows as
            they are written, so documents already in the database need a
            :meth:`rebuild_text_index` to become searchable.
        """
        from .ingest_report import IndexReport

        if batch_size <= 0:
            raise DatabaseError("batch_size must be a positive integer")
        embedder = None
        if index is not None:
            embedder = self._get_embedding_function(index)

        # Registered up front so rows are indexed as they are written, rather
        # than needing a full rebuild afterwards. Undone below if the ingest
        # fails, so a failed call leaves no index for documents never loaded.
        fts_registered_here = False
        if configure_fts and not any(
            cfg["entity_type"] == "node"
            and cfg["label_or_type"] == label
            and cfg["property"] == text_key
            for cfg in self.list_text_indexes()
        ):
            self.create_text_index("node", label, [text_key])
            fts_registered_here = True

        try:
            return self._index_documents(
                rows,
                label=label,
                id_key=id_key,
                text_key=text_key,
                index=index,
                relationships_key=relationships_key,
                default_rel_type=default_rel_type,
                copy_attributes=copy_attributes,
                properties=properties,
                batch_size=batch_size,
                upsert=upsert,
                embedder=embedder,
                report=IndexReport(),
            )
        except Exception:
            if fts_registered_here:
                self.drop_text_index("node", label, [text_key])
            raise

    def _index_documents(
        self,
        rows: "Iterable[dict[str, Any]]",
        *,
        label: str,
        id_key: str | None,
        text_key: str,
        index: str | None,
        relationships_key: str | None,
        default_rel_type: str,
        copy_attributes: bool,
        properties: list[str] | None,
        batch_size: int,
        upsert: bool,
        embedder: Any,
        report: "IndexReport",
    ) -> "IndexReport":
        """Load rows into nodes, edges and embeddings. See :meth:`index_documents`."""
        pending_relationships: list[tuple[int, Any, str, dict]] = []
        existing: dict[Any, int] = {}
        if id_key and upsert:
            existing = self._external_id_map(label, id_key)

        pending_ids: list[int] = []
        pending_texts: list[str] = []

        def flush() -> None:
            if not pending_ids:
                return
            if embedder is not None:
                vectors = embedder(pending_texts)
                self.upsert_embeddings_batch(pending_ids, vectors, index=index)
                report.embedded += len(pending_ids)
            pending_ids.clear()
            pending_texts.clear()

        for row in rows:
            if not isinstance(row, dict):
                raise DatabaseError("index_documents expects an iterable of dicts")
            external_id = row.get(id_key) if id_key else None
            props = self._select_document_properties(
                row,
                id_key=id_key,
                text_key=text_key,
                relationships_key=relationships_key,
                copy_attributes=copy_attributes,
                allowlist=properties,
            )

            node_id = existing.get(external_id) if external_id is not None else None
            if node_id is not None:
                self.update_node_properties(node_id, props)
                report.nodes_updated += 1
            else:
                node = self.create_node(labels=[label], properties=props)
                node_id = node.id
                report.nodes_created += 1
                if external_id is not None:
                    existing[external_id] = node_id
            if external_id is not None:
                report.ids[external_id] = node_id

            text = row.get(text_key)
            if index is not None and isinstance(text, str) and text.strip():
                if embedder is None:
                    raise DatabaseError(
                        f"Vector index '{index}' has no embedding function; "
                        "pass index=None to skip embedding"
                    )
                pending_ids.append(node_id)
                pending_texts.append(text)
                if len(pending_ids) >= batch_size:
                    flush()

            if relationships_key:
                for spec in row.get(relationships_key) or []:
                    if isinstance(spec, dict):
                        target = spec.get("id")
                        rel_type = spec.get("type") or default_rel_type
                        rel_props = spec.get("properties") or {}
                    else:
                        target, rel_type, rel_props = spec, default_rel_type, {}
                    if target is None:
                        continue
                    pending_relationships.append((node_id, target, rel_type, rel_props))

        flush()

        for source_id, target_ref, rel_type, rel_props in pending_relationships:
            target_id = existing.get(target_ref)
            if target_id is None and id_key:
                matches = self.match_nodes(labels=[label], properties={id_key: target_ref})
                target_id = matches[0].id if matches else None
            if target_id is None:
                report.unresolved.append(target_ref)
                continue
            self.create_relationship(source_id, target_id, rel_type, rel_props)
            report.relationships_created += 1

        return report

    #: Marks relationships produced by create_semantic_graph, so that a rebuild
    #: can replace exactly those and leave hand-made edges of the same type alone.
    SEMANTIC_GRAPH_MARKER = "create_semantic_graph"

    def create_semantic_graph(
        self,
        index: str = "default",
        *,
        rel_type: str = "SEMANTIC_SIMILAR",
        k: int = 15,
        min_score: float | None = None,
        approximate: bool = False,
        labels: list[str] | None = None,
        undirected: bool = True,
        replace: bool = True,
        max_edges: int | None = None,
    ) -> "SemanticGraphReport":
        """Materialise each node's nearest neighbours as relationships.

        Turns the vector index into a navigable graph: every indexed node gets
        edges to its ``k`` closest neighbours, so similarity becomes traversable
        with Cypher and analysable with :meth:`communities` — which is the one
        thing the ANN index alone cannot give you, since community detection
        needs edges to exist.

        For everything else, prefer querying the index directly. ``CALL
        db.vector.search`` and ``SIMILAR()`` answer the same questions without
        storing anything, and they never go stale.

        Args:
            index: Vector index to read neighbours from.
            rel_type: Relationship type to create.
            k: Neighbours per node.
            min_score: Skip neighbours scoring below this — the single most
                effective control on how many edges you end up with. Defaults
                to ``0.1``; required on an ``l2`` index, whose scores are
                negative.
            approximate: Only process nodes that have no generated edge for this
                ``rel_type`` and ``index`` yet. Hand-made relationships of the
                same type do not count, so they never exclude a node from being
                processed. This is the incremental mode — see
                :meth:`refresh_semantic_graph`. Has no effect with
                ``replace=True``, which deletes exactly those edges.
            labels: Only link nodes carrying one of these labels.
            undirected: Emit one edge per pair rather than one per direction.
                Halves the edge count. Query it with an undirected pattern,
                ``MATCH (a)-[:SEMANTIC_SIMILAR]-(b)`` — every node still reaches
                its ``k`` neighbours that way, but its *outgoing* degree alone
                will be less than ``k``.
            replace: Delete previously generated edges of this type and index
                first. Only edges carrying this method's marker are removed, so
                hand-made relationships of the same type survive, as do edges
                generated from a different vector index.
            max_edges: Stop once this many edges have been created. Enforced
                *between* nodes, so the count can overshoot by up to ``k-1``:
                cutting a node off half way through its neighbours would leave
                it looking processed with an incomplete neighbourhood, and no
                later refresh would finish it. Because every node it does
                process is complete, repeated
                :meth:`refresh_semantic_graph` calls under the same cap
                eventually build the whole graph.

        Returns:
            A :class:`~grafito.ingest_report.SemanticGraphReport`.

        Warning:
            This writes ``k`` edges per node into the same table as your domain
            relationships. At 100k nodes and ``k=15`` that is ~750k rows with
            ``undirected=True``. They will dominate any unqualified traversal
            (``MATCH (a)-[]->(b)``) and any centrality or community result that
            does not exclude them — pass ``exclude_rel_types=[rel_type]`` to
            :meth:`centrality` and :meth:`communities`. They also go stale: the
            edges reflect the embeddings as of the build, and nothing updates
            them when vectors change. Re-run, or use
            :meth:`refresh_semantic_graph` for new nodes.

        Note:
            The rebuild is atomic. Neighbours are computed first — the slow
            part — and the old edges are swapped for the new ones in a single
            short transaction, so a failure mid-build leaves the previous graph
            intact and readers never see it empty.
        """
        from datetime import datetime, timezone

        from .ingest_report import SemanticGraphReport

        if k <= 0:
            raise DatabaseError("k must be a positive integer")
        if max_edges is not None and max_edges <= 0:
            raise DatabaseError("max_edges must be a positive integer")
        if not rel_type or not isinstance(rel_type, str):
            raise DatabaseError("rel_type must be a non-empty string")

        vec_index = self._get_vector_index(index)
        report = SemanticGraphReport()
        min_score = self._resolve_semantic_min_score(min_score, index)

        linked: set[int] = set()
        seen_pairs: set[tuple[int, int]] = set()
        if not replace:
            # With replace=True everything below is about to be deleted, so
            # neither the skip set nor the dedup set would mean anything.
            existing = self._generated_edge_pairs(rel_type, index)
            if approximate:
                # A node counts as processed only if it is the *source* of a
                # generated edge — that is the node whose neighbourhood was
                # searched. Counting targets too would permanently skip nodes
                # that merely received an edge before a truncated build stopped,
                # which is the same "excluded forever" failure the marker was
                # meant to prevent. The cost of being strict is re-searching a
                # node whose edges were all deduplicated away; `seen_pairs`
                # below keeps that from creating duplicates.
                linked = {source for source, _ in existing}
            seen_pairs = {
                (min(source, target), max(source, target)) if undirected
                else (source, target)
                for source, target in existing
            }

        candidates = self._indexed_node_ids(index, labels)
        candidate_set = set(candidates)
        generated_at = datetime.now(timezone.utc).isoformat()
        pending: list[tuple[int, int, str, str, None]] = []

        for node_id in candidates:
            if approximate and node_id in linked:
                report.nodes_skipped += 1
                continue
            # Stop between nodes, never inside one. A node cut off half way
            # through its neighbours would still count as processed — being the
            # source of an edge — and no later refresh would finish it. The cost
            # is that max_edges is approximate, overshooting by at most k-1.
            if max_edges is not None and len(pending) >= max_edges:
                report.truncated = True
                break
            vector = vec_index.get_vector(node_id)
            if vector is None:
                vector = self._collect_vectors(index, [node_id]).get(node_id)
            if vector is None:  # pragma: no cover - candidates come from vectors
                continue
            report.nodes_processed += 1

            # k+1 because the node itself is its own nearest neighbour.
            for neighbour_id, score in vec_index.search(vector, k + 1):
                neighbour_id = int(neighbour_id)
                if neighbour_id == node_id or score < min_score:
                    continue
                if labels and neighbour_id not in candidate_set:
                    continue
                # Deduplicate within this build and against edges already in the
                # database, so re-running never stacks a second copy of an edge
                # that is already there.
                pair = (
                    (min(node_id, neighbour_id), max(node_id, neighbour_id))
                    if undirected
                    else (node_id, neighbour_id)
                )
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                properties = {
                    "score": float(score),
                    "index": index,
                    "generated_by": self.SEMANTIC_GRAPH_MARKER,
                    "generated_at": generated_at,
                }
                pending.append(
                    (
                        node_id,
                        neighbour_id,
                        rel_type,
                        orjson.dumps(properties).decode("utf-8"),
                        None,
                    )
                )

        if pending:
            # Every edge shares one type and one property shape, so constraints
            # are checked once rather than once per row. Going through
            # create_relationship would re-validate both endpoints for each of
            # k*N edges. Do it before opening the transaction, so a constraint
            # violation never reaches the delete.
            self._validate_constraints_on_relationship(
                rel_type,
                {
                    "score": 0.0,
                    "index": index,
                    "generated_by": self.SEMANTIC_GRAPH_MARKER,
                    "generated_at": generated_at,
                },
            )

        # Swap the old graph for the new one atomically. The neighbour search
        # above is the slow part and runs entirely outside this transaction, so
        # readers never observe a window where the semantic graph is missing,
        # and a failure mid-build leaves the previous graph intact.
        owns_transaction = not self._in_transaction
        if owns_transaction:
            self.begin_transaction()
        try:
            if replace:
                report.edges_removed = self._delete_generated_relationships(
                    rel_type, index=index
                )
            if pending:
                self.conn.executemany(
                    """
                    INSERT INTO relationships (source_node_id, target_node_id, type, properties, uri)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    pending,
                )
                report.edges_created = len(pending)
            if owns_transaction:
                self.commit()
        except Exception as exc:
            if owns_transaction:
                self.rollback()
            if isinstance(exc, sqlite3.Error):
                raise DatabaseError(f"Failed to create semantic graph: {exc}") from exc
            raise

        return report

    def refresh_semantic_graph(
        self,
        index: str = "default",
        *,
        rel_type: str = "SEMANTIC_SIMILAR",
        **kwargs: Any,
    ) -> "SemanticGraphReport":
        """Link nodes that have no semantic edges yet, leaving existing ones alone.

        The incremental counterpart of :meth:`create_semantic_graph`: run it
        after adding documents to connect the new ones without rebuilding the
        whole graph.

        This does not re-examine nodes that already have edges, so it will not
        notice that an *existing* node's neighbourhood changed — only a full
        rebuild does that. Embeddings that are updated in place drift out of
        sync until then.
        """
        kwargs.setdefault("approximate", True)
        kwargs.setdefault("replace", False)
        return self.create_semantic_graph(index, rel_type=rel_type, **kwargs)

    def drop_semantic_graph(
        self,
        rel_type: str = "SEMANTIC_SIMILAR",
        *,
        index: str | None = None,
    ) -> int:
        """Delete the generated relationships of ``rel_type``.

        Only edges carrying the :meth:`create_semantic_graph` marker are
        removed; relationships of the same type created by hand are kept.

        Args:
            rel_type: Relationship type to clear.
            index: Only drop edges generated from this vector index. ``None``
                drops them regardless of source index.

        Returns:
            Number of relationships deleted.
        """
        return self._delete_generated_relationships(rel_type, index=index)

    def _delete_generated_relationships(self, rel_type: str, index: str | None = None) -> int:
        """Remove relationships this class generated, by marker.

        ``index`` narrows the deletion to edges generated from one vector index,
        so rebuilding one index does not wipe another's edges of the same type.
        """
        sql = """
            DELETE FROM relationships
            WHERE type = ?
              AND json_extract(properties, '$.generated_by') = ?
        """
        params: list[Any] = [rel_type, self.SEMANTIC_GRAPH_MARKER]
        if index is not None:
            sql += " AND json_extract(properties, '$.index') = ?"
            params.append(index)
        try:
            cursor = self.conn.execute(sql, params)
            if not self._in_transaction:
                self.conn.commit()
            return cursor.rowcount or 0
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to remove generated relationships: {exc}") from exc

    def _generated_edge_pairs(self, rel_type: str, index: str) -> list[tuple[int, int]]:
        """(source, target) of every edge this method generated for one index."""
        rows = self.conn.execute(
            """
            SELECT source_node_id, target_node_id FROM relationships
            WHERE type = ?
              AND json_extract(properties, '$.generated_by') = ?
              AND json_extract(properties, '$.index') = ?
            """,
            (rel_type, self.SEMANTIC_GRAPH_MARKER, index),
        ).fetchall()
        return [(int(row["source_node_id"]), int(row["target_node_id"])) for row in rows]

    def _resolve_semantic_min_score(self, min_score: float | None, index: str) -> float:
        """Pick a neighbour-score floor, refusing where no default is meaningful.

        Mirrors the Cypher ``SIMILAR()`` rule: a cosine-calibrated default is
        silently wrong on an ``l2`` index, whose scores are negated distances
        (``<= 0``), so require an explicit value there instead of building a
        graph with no edges.
        """
        if min_score is not None:
            # bool is a subclass of int, so float(True) would silently become a
            # threshold of 1.0. SIMILAR() already refuses this.
            if isinstance(min_score, bool) or not isinstance(min_score, (int, float)):
                raise DatabaseError("min_score must be a number")
            return float(min_score)
        metric = self.vector_metric(index)
        if metric == "l2":
            raise DatabaseError(
                f"create_semantic_graph requires an explicit min_score on the "
                f"'{index}' index because its 'l2' metric produces negative "
                "scores (e.g. min_score=-0.5)"
            )
        return DEFAULT_SEMANTIC_GRAPH_MIN_SCORE

    def _indexed_node_ids(self, index: str, labels: list[str] | None) -> list[int]:
        """Node ids that carry a vector in ``index``, optionally label-filtered.

        Enumerating from the node table rather than the backend keeps this
        agnostic: the VectorIndex interface exposes ``get_vector`` but no way to
        list what it holds.
        """
        vec_index = self._get_vector_index(index)
        sql = "SELECT n.id AS id FROM nodes n"
        params: list[Any] = []
        if labels:
            placeholders = ", ".join("?" for _ in labels)
            sql += f"""
                WHERE n.id IN (
                    SELECT nl.node_id FROM node_labels nl
                    JOIN labels l ON l.id = nl.label_id
                    WHERE l.name IN ({placeholders})
                )
            """
            params.extend(labels)
        sql += " ORDER BY n.id"

        stored: set[int] | None = None
        if vec_index.options.get("store_embeddings"):
            rows = self.conn.execute(
                "SELECT node_id FROM vector_entries WHERE index_name = ?", (index,)
            ).fetchall()
            stored = {int(row["node_id"]) for row in rows}

        result = []
        for row in self.conn.execute(sql, params).fetchall():
            node_id = int(row["id"])
            if stored is not None:
                if node_id in stored:
                    result.append(node_id)
                continue
            if vec_index.get_vector(node_id) is not None:
                result.append(node_id)
        return result

    def _external_id_map(self, label: str, id_key: str) -> dict[Any, int]:
        """Map existing external ids to node ids, for upsert."""
        rows = self.conn.execute(
            """
            SELECT n.id AS id, json_extract(n.properties, '$.' || ?) AS external_id
            FROM nodes n
            JOIN node_labels nl ON nl.node_id = n.id
            JOIN labels l ON l.id = nl.label_id
            WHERE l.name = ?
            """,
            (id_key, label),
        ).fetchall()
        return {
            row["external_id"]: int(row["id"])
            for row in rows
            if row["external_id"] is not None
        }

    @staticmethod
    def _select_document_properties(
        row: dict[str, Any],
        *,
        id_key: str | None,
        text_key: str,
        relationships_key: str | None,
        copy_attributes: bool,
        allowlist: list[str] | None,
    ) -> dict[str, Any]:
        """Decide which row keys become node properties."""
        if allowlist is not None:
            return {key: row[key] for key in allowlist if key in row}
        if copy_attributes:
            # The relationships key describes edges, not the node; keeping it
            # would duplicate the graph structure inside a property.
            return {
                key: value
                for key, value in row.items()
                if key != relationships_key
            }
        return {
            key: row[key]
            for key in (id_key, text_key)
            if key is not None and key in row
        }

    def from_networkx(
        self,
        graph,
        label_attr: str = "labels",
        property_attr: str = "properties",
        rel_type_attr: str = "type",
        rel_property_attr: str = "properties",
    ) -> dict[Any, int]:
        """Import nodes and relationships from a NetworkX graph.

        Returns a mapping of original node IDs to new Grafito node IDs.
        """
        node_map: dict[Any, int] = {}

        for node_id, attrs in graph.nodes(data=True):
            labels = attrs.get(label_attr, [])
            if not isinstance(labels, list):
                labels = [labels]
            properties = dict(attrs.get(property_attr, {}))
            node_uri = attrs.get("uri") or properties.pop("uri", None)
            for key, value in attrs.items():
                if key in (label_attr, property_attr, "uri"):
                    continue
                properties.setdefault(key, value)
            created = self.create_node(labels=labels, properties=properties, uri=node_uri)
            node_map[node_id] = created.id

        if hasattr(graph, "edges"):
            for source, target, key, attrs in graph.edges(keys=True, data=True):
                rel_type = attrs.get(rel_type_attr, "RELATED_TO")
                properties = dict(attrs.get(rel_property_attr, {}))
                rel_uri = attrs.get("uri") or properties.pop("uri", None)
                for key_name, value in attrs.items():
                    if key_name in (rel_type_attr, rel_property_attr, "uri"):
                        continue
                    properties.setdefault(key_name, value)
                self.create_relationship(
                    node_map[source],
                    node_map[target],
                    rel_type,
                    properties,
                    uri=rel_uri,
                )

        return node_map

    # =========================================================================
    # Reranker Registration
    # =========================================================================

    def register_reranker(self, name: str, reranker: Any) -> None:
        """Register a reranker callable for semantic search.

        Args:
            name: Reranker name (alphanumeric + underscore).
            reranker: Callable taking (query_vector, candidates) and returning
                a list of {id, score} (dict) or (id, score) tuples.
        """
        self._validate_index_identifier(name, "Reranker name")
        if not callable(reranker):
            raise DatabaseError("Reranker must be callable")
        self._rerankers[name] = reranker

    def register_embedding_function(self, name: str, embedding_function: EmbeddingFunction) -> None:
        """Register an embedding function instance by name."""
        self._validate_index_identifier(name, "Embedding function name")
        self._embedding_functions[name] = embedding_function

    # =========================================================================
    # Context Manager Support
    # =========================================================================

    def __enter__(self):
        """Enter context manager - begin transaction."""
        self.begin_transaction()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - commit on success, rollback on exception."""
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        return False  # Don't suppress exceptions

    # =========================================================================
    # Transaction Management
    # =========================================================================

    def begin_transaction(self) -> None:
        """Begin an explicit transaction."""
        if not self._in_transaction:
            self.conn.execute("BEGIN")
            self._in_transaction = True

    def commit(self) -> None:
        """Commit the current transaction."""
        if self._in_transaction:
            self.conn.commit()
            self._in_transaction = False

    def rollback(self) -> None:
        """Rollback the current transaction."""
        if self._in_transaction:
            self.conn.rollback()
            self._in_transaction = False

    # =========================================================================
    # Helper Methods (Private)
    # =========================================================================

    def _validate_properties(self, properties: dict) -> dict:
        """Validate that properties contain only supported types.

        Args:
            properties: Dictionary of properties to validate

        Returns:
            The validated properties dictionary

        Raises:
            InvalidPropertyError: If any property has an unsupported type
        """
        if not properties:
            return {}

        normalized = {}
        for key, value in properties.items():
            normalized[key] = self._normalize_property_value(key, value)
        return normalized

    def _format_datetime(self, value: datetime) -> str:
        """Format datetime for storage, preserving timezone offset when present."""
        offset = value.utcoffset()
        if value.tzinfo is None or offset is None:
            return value.isoformat()
        if offset == timedelta(0):
            return value.astimezone(timezone.utc).replace(tzinfo=None).isoformat() + "Z"
        return value.isoformat()

    def _format_time(self, value: time) -> str:
        """Format time for storage, preserving timezone offset when present."""
        offset = value.utcoffset()
        if value.tzinfo is None or offset is None:
            return value.isoformat()
        return value.isoformat()

    def _normalize_property_value(self, key_path: str, value: object) -> object:
        """Validate and normalize a single property value recursively."""
        allowed_types = (int, float, str, bool, type(None))
        if isinstance(value, allowed_types):
            return value
        if isinstance(value, datetime):
            return self._format_datetime(value)
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, time):
            return self._format_time(value)
        if isinstance(value, list):
            return [
                self._normalize_property_value(f"{key_path}[{index}]", item)
                for index, item in enumerate(value)
            ]
        if isinstance(value, dict):
            normalized = {}
            for sub_key, sub_value in value.items():
                if not isinstance(sub_key, str):
                    raise InvalidPropertyError(
                        f"Property '{key_path}' has invalid key type {type(sub_key).__name__}. "
                        "Only string keys are supported."
                    )
                normalized[sub_key] = self._normalize_property_value(
                    f"{key_path}.{sub_key}", sub_value
                )
            return normalized
        raise InvalidPropertyError(
            f"Property '{key_path}' has invalid type {type(value).__name__}. "
            "Only int, float, str, bool, list, dict, date, time, datetime, or null are supported."
        )

    def _validate_index_identifier(self, name: str, kind: str) -> None:
        """Validate identifiers used for index definitions."""
        if not name:
            raise DatabaseError(f"{kind} cannot be empty")
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise DatabaseError(
                f"{kind} '{name}' has invalid characters; use letters, digits, or '_' only."
            )

    def _make_index_name(self, entity: str, label_or_type: str | None, prop: str) -> str:
        """Build a deterministic index name."""
        suffix = label_or_type or "all"
        return f"idx_{entity}_{suffix}_{prop}".lower()

    def _make_constraint_name(self, entity: str, label_or_type: str, prop: str, kind: str) -> str:
        """Build a deterministic constraint name."""
        return f"constraint_{entity}_{label_or_type}_{prop}_{kind}".lower()

    def _create_property_index(
        self,
        entity: str,
        label_or_type: str | None,
        property_name: str,
        unique: bool = False,
        name: str | None = None,
    ) -> str:
        """Create a property index in SQLite and register metadata."""
        self._validate_index_identifier(entity, "Entity")
        if label_or_type is not None:
            self._validate_index_identifier(label_or_type, "Label/type")
        self._validate_index_identifier(property_name, "Property")

        index_name = name or self._make_index_name(entity, label_or_type, property_name)
        self._validate_index_identifier(index_name, "Index name")

        table = "nodes" if entity == "node" else "relationships"
        unique_sql = "UNIQUE " if unique else ""
        expr = f"json_extract(properties, '$.{property_name}')"
        sql = f"CREATE {unique_sql}INDEX IF NOT EXISTS {index_name} ON {table} ({expr})"

        try:
            self.conn.execute(sql)
            self.conn.execute(
                """
                INSERT OR REPLACE INTO property_indexes
                    (name, entity, label_or_type, property, unique_flag)
                VALUES (?, ?, ?, ?, ?)
                """,
                (index_name, entity, label_or_type, property_name, 1 if unique else 0),
            )
            self.conn.commit()
        except Exception as exc:
            raise DatabaseError(f"Failed to create index '{index_name}': {exc}", exc)

        return index_name

    def create_node_index(self, label: str | None, property_name: str, unique: bool = False) -> str:
        """Create a property index for nodes.

        Args:
            label: Optional label for metadata/naming.
            property_name: Property name to index.
            unique: Whether to enforce uniqueness (SQLite UNIQUE index).
        """
        return self._create_property_index("node", label, property_name, unique=unique)

    def create_relationship_index(self, rel_type: str | None, property_name: str, unique: bool = False) -> str:
        """Create a property index for relationships.

        Args:
            rel_type: Optional relationship type for metadata/naming.
            property_name: Property name to index.
            unique: Whether to enforce uniqueness (SQLite UNIQUE index).
        """
        return self._create_property_index("relationship", rel_type, property_name, unique=unique)

    def _create_uri_index(self, table: str, unique: bool = True, name: str | None = None) -> str:
        """Create an index on the uri column for nodes or relationships."""
        if table not in {"nodes", "relationships"}:
            raise DatabaseError(f"Invalid table for uri index: {table}")
        suffix = "unique" if unique else "idx"
        index_name = name or f"idx_{table}_uri_{suffix}"
        self._validate_index_identifier(index_name, "Index name")
        unique_sql = "UNIQUE " if unique else ""
        sql = f"CREATE {unique_sql}INDEX IF NOT EXISTS {index_name} ON {table} (uri)"
        try:
            self.conn.execute(sql)
            self.conn.commit()
        except Exception as exc:
            raise DatabaseError(f"Failed to create uri index '{index_name}': {exc}", exc)
        return index_name

    def create_node_uri_index(self, unique: bool = True, name: str | None = None) -> str:
        """Create a (unique) index on nodes.uri."""
        return self._create_uri_index("nodes", unique=unique, name=name)

    def create_relationship_uri_index(self, unique: bool = True, name: str | None = None) -> str:
        """Create a (unique) index on relationships.uri."""
        return self._create_uri_index("relationships", unique=unique, name=name)

    def drop_index(self, name: str) -> None:
        """Drop a property index by name."""
        self._validate_index_identifier(name, "Index name")
        try:
            self.conn.execute(f"DROP INDEX IF EXISTS {name}")
            self.conn.execute("DELETE FROM property_indexes WHERE name = ?", (name,))
            self.conn.commit()
        except Exception as exc:
            raise DatabaseError(f"Failed to drop index '{name}': {exc}", exc)

    def list_indexes(self) -> list[dict[str, Any]]:
        """List registered property indexes."""
        cursor = self.conn.execute(
            """
            SELECT name, entity, label_or_type, property, unique_flag
            FROM property_indexes
            ORDER BY name
            """
        )
        return [
            {
                "name": row["name"],
                "entity": row["entity"],
                "label_or_type": row["label_or_type"],
                "property": row["property"],
                "unique": bool(row["unique_flag"]),
            }
            for row in cursor.fetchall()
        ]

    # =========================================================================
    # Full-Text Search (FTS5)
    # =========================================================================

    def rebuild_text_index(self) -> None:
        """Rebuild the materialized FTS index from current data and config."""
        started_transaction = False
        try:
            if not self._in_transaction:
                self.begin_transaction()
                started_transaction = True

            self.conn.execute("DELETE FROM fts_index")

            self.conn.execute(
                """
                INSERT INTO fts_index (entity_type, entity_id, label_type, content)
                SELECT
                    'node',
                    n.id,
                    l.name,
                    (
                        SELECT group_concat(
                                   CAST(json_extract(n.properties, '$.' || c.property) AS TEXT),
                                   ' '
                               )
                        FROM fts_config c
                        WHERE c.entity_type = 'node'
                          AND (c.label_type IS NULL OR c.label_type = l.name)
                          AND json_extract(n.properties, '$.' || c.property) IS NOT NULL
                    )
                FROM nodes n
                JOIN node_labels nl ON nl.node_id = n.id
                JOIN labels l ON l.id = nl.label_id
                """
            )

            self.conn.execute(
                """
                INSERT INTO fts_index (entity_type, entity_id, label_type, content)
                SELECT
                    'relationship',
                    r.id,
                    r.type,
                    (
                        SELECT group_concat(
                                   CAST(json_extract(r.properties, '$.' || c.property) AS TEXT),
                                   ' '
                               )
                        FROM fts_config c
                        WHERE c.entity_type = 'relationship'
                          AND (c.label_type IS NULL OR c.label_type = r.type)
                          AND json_extract(r.properties, '$.' || c.property) IS NOT NULL
                    )
                FROM relationships r
                """
            )

            if started_transaction:
                self.commit()
        except Exception as exc:
            if started_transaction:
                self.rollback()
            raise DatabaseError(f"Failed to rebuild text index: {exc}", exc)

    def create_text_index(
        self,
        entity_type: str,
        label_or_type: str | None,
        properties: list[str] | str,
        weights: dict[str, float] | None = None,
    ) -> None:
        """Register properties for full-text indexing."""
        if not entity_type:
            raise DatabaseError("Entity type cannot be empty")
        entity = entity_type.lower()
        if entity not in ("node", "relationship"):
            raise DatabaseError("Entity type must be 'node' or 'relationship'")

        if isinstance(properties, str):
            properties = [properties]
        if not properties:
            raise DatabaseError("Properties cannot be empty")

        if label_or_type is not None:
            if not label_or_type:
                raise DatabaseError("Label/type cannot be empty")
            self._validate_index_identifier(label_or_type, "Label/type")

        for prop in properties:
            if not isinstance(prop, str) or not prop:
                raise DatabaseError("Property names must be non-empty strings")
            self._validate_index_identifier(prop, "Property")

        params: list[Any] = [entity]
        label_clause = "label_type IS NULL"
        if label_or_type is not None:
            label_clause = "label_type = ?"
            params.append(label_or_type)

        placeholders = ",".join("?" * len(properties))
        params.extend(properties)
        cursor = self.conn.execute(
            f"""
            SELECT property
            FROM fts_config
            WHERE entity_type = ?
              AND {label_clause}
              AND property IN ({placeholders})
            """,
            params,
        )
        existing = {row["property"] for row in cursor.fetchall()}

        rows = []
        for prop in properties:
            if prop in existing:
                continue
            weight = None
            if weights and prop in weights:
                try:
                    weight = float(weights[prop])
                except (TypeError, ValueError) as exc:
                    raise DatabaseError(f"Weight for property '{prop}' must be numeric") from exc
            rows.append((entity, label_or_type, prop, weight))

        if not rows:
            return

        try:
            self.conn.executemany(
                """
                INSERT INTO fts_config (entity_type, label_type, property, weight)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )
            if not self._in_transaction:
                self.conn.commit()
        except Exception as exc:
            if not self._in_transaction:
                self.conn.rollback()
            raise DatabaseError(f"Failed to create text index: {exc}", exc)

    def drop_text_index(
        self,
        entity_type: str,
        label_or_type: str | None,
        properties: list[str] | str | None = None,
    ) -> None:
        """Remove properties from the full-text index configuration."""
        if not entity_type:
            raise DatabaseError("Entity type cannot be empty")
        entity = entity_type.lower()
        if entity not in ("node", "relationship"):
            raise DatabaseError("Entity type must be 'node' or 'relationship'")

        if label_or_type is not None:
            if not label_or_type:
                raise DatabaseError("Label/type cannot be empty")
            self._validate_index_identifier(label_or_type, "Label/type")

        params: list[Any] = [entity]
        label_clause = "label_type IS NULL"
        if label_or_type is not None:
            label_clause = "label_type = ?"
            params.append(label_or_type)

        sql = f"DELETE FROM fts_config WHERE entity_type = ? AND {label_clause}"
        if properties is not None:
            if isinstance(properties, str):
                properties = [properties]
            if not properties:
                raise DatabaseError("Properties cannot be empty")
            for prop in properties:
                if not isinstance(prop, str) or not prop:
                    raise DatabaseError("Property names must be non-empty strings")
                self._validate_index_identifier(prop, "Property")
            placeholders = ",".join("?" * len(properties))
            sql += f" AND property IN ({placeholders})"
            params.extend(properties)

        try:
            self.conn.execute(sql, params)
            if not self._in_transaction:
                self.conn.commit()
        except Exception as exc:
            if not self._in_transaction:
                self.conn.rollback()
            raise DatabaseError(f"Failed to drop text index: {exc}", exc)

    def list_text_indexes(self) -> list[dict[str, Any]]:
        """List configured full-text index entries."""
        cursor = self.conn.execute(
            """
            SELECT entity_type, label_type, property, weight
            FROM fts_config
            ORDER BY entity_type, label_type, property
            """
        )
        return [
            {
                "entity_type": row["entity_type"],
                "label_or_type": row["label_type"],
                "property": row["property"],
                "weight": row["weight"],
            }
            for row in cursor.fetchall()
        ]

    def text_search(
        self,
        query: str,
        k: int | None = None,
        labels: list[str] | None = None,
        rel_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search nodes/relationships using FTS5 BM25."""
        if not query or not query.strip():
            raise DatabaseError("Query cannot be empty")
        if k is None:
            k = self.default_top_k
        if k <= 0:
            raise DatabaseError("k must be a positive integer")

        if labels is not None and not isinstance(labels, list):
            raise DatabaseError("labels must be a list of strings or None")
        if rel_types is not None and not isinstance(rel_types, list):
            raise DatabaseError("rel_types must be a list of strings or None")
        if labels:
            for label in labels:
                if not isinstance(label, str) or not label:
                    raise DatabaseError("labels must contain non-empty strings")
        if rel_types:
            for rel_type in rel_types:
                if not isinstance(rel_type, str) or not rel_type:
                    raise DatabaseError("rel_types must contain non-empty strings")

        if labels is not None and len(labels) == 0:
            labels = None
        if rel_types is not None and len(rel_types) == 0:
            rel_types = None

        if labels is None and rel_types is None:
            search_nodes = True
            search_rels = True
        else:
            search_nodes = labels is not None
            search_rels = rel_types is not None

        def build_query(entity: str, label_filter: list[str] | None) -> tuple[str, list[Any]]:
            sql = """
                SELECT entity_type, entity_id, bm25(fts_index) AS score
                FROM fts_index
                WHERE entity_type = ?
            """
            params: list[Any] = [entity]
            if label_filter:
                placeholders = ",".join("?" * len(label_filter))
                sql += f" AND label_type IN ({placeholders})"
                params.extend(label_filter)
            sql += " AND fts_index MATCH ?"
            params.append(query)
            return sql, params

        if search_nodes and search_rels:
            node_sql, node_params = build_query("node", labels)
            rel_sql, rel_params = build_query("relationship", rel_types)
            sql = f"""
                SELECT entity_type, entity_id, score
                FROM ({node_sql} UNION ALL {rel_sql})
                ORDER BY score ASC
                LIMIT ?
            """
            params = node_params + rel_params + [k]
        elif search_nodes:
            sql, params = build_query("node", labels)
            sql += " ORDER BY score ASC LIMIT ?"
            params.append(k)
        else:
            sql, params = build_query("relationship", rel_types)
            sql += " ORDER BY score ASC LIMIT ?"
            params.append(k)

        cursor = self.conn.execute(sql, params)
        results = []
        for row in cursor.fetchall():
            entity_type = row["entity_type"]
            entity_id = int(row["entity_id"])
            score = float(row["score"])
            if entity_type == "node":
                entity = self.get_node(entity_id)
            else:
                entity = self.get_relationship(entity_id)
            if entity is None:
                continue
            results.append(
                {
                    "entity": entity,
                    "entity_type": entity_type,
                    "score": score,
                }
            )
        return results

    # =========================================================================
    # Custom Text Indexes
    # =========================================================================

    def register_text_index(self, name: str, text_index: Any) -> None:
        """Register a custom text index backend.
        
        Args:
            name: Name for the text index.
            text_index: TextIndex instance (e.g., BM25SIndex).
        """
        self._validate_index_identifier(name, "Text index name")
        self._text_indexes[name] = text_index

    def list_text_index_backends(self) -> list[str]:
        """List registered custom text index backends."""
        return list(self._text_indexes.keys())

    def get_text_index(self, name: str) -> Any:
        """Get a registered text index by name.
        
        Args:
            name: Name of the text index.
            
        Returns:
            The TextIndex instance.
            
        Raises:
            DatabaseError: If the index is not found.
        """
        if name not in self._text_indexes:
            raise DatabaseError(f"Text index '{name}' not found")
        return self._text_indexes[name]

    def text_search_custom(
        self,
        query: str,
        index: str,
        k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search using a custom text index backend.
        
        Args:
            query: Search query string.
            index: Name of the registered text index.
            k: Number of results to return.
            
        Returns:
            List of dicts with 'node' and 'score' keys.
        """
        if not query or not query.strip():
            raise DatabaseError("Query cannot be empty")
        if k is None:
            k = self.default_top_k
        if k <= 0:
            raise DatabaseError("k must be a positive integer")
        
        text_idx = self.get_text_index(index)
        raw_results = text_idx.search(query, k)
        
        results = []
        for node_id, score in raw_results:
            node = self.get_node(node_id)
            if node is not None:
                results.append({"node": node, "score": score})
        return results

    # =========================================================================
    # Vector Indexes
    # =========================================================================

    def create_vector_index(
        self,
        name: str,
        dim: int | None = None,
        backend: str = "bruteforce",
        method: str = "flat",
        options: dict[str, Any] | None = None,
        indexer: Indexer | None = None,
        embedding_function: EmbeddingFunction | str | None = None,
        if_not_exists: bool = False,
    ) -> str:
        """Create a vector index registry entry and in-memory backend.
        
        Args:
            name: Name of the vector index.
            dim: Dimension of the vectors.
            backend: Backend to use ('bruteforce', 'faiss', 'annoy', 'leann').
            method: Index method (backend-specific).
            options: Backend-specific options.
            indexer: Indexer instance (overrides other parameters).
            embedding_function: Embedding function to use.
            if_not_exists: If True, don't raise error if index already exists.
        
        Returns:
            The name of the created index.
        """
        self._validate_index_identifier(name, "Vector index name")
        if indexer is not None:
            backend = indexer.backend
            method = indexer.method
            options = indexer.to_options()
            embedding_function = indexer.embedding_function
            dim = indexer.dim
        if isinstance(embedding_function, str):
            if embedding_function not in self._embedding_functions:
                raise DatabaseError(f"Unknown embedding function '{embedding_function}'")
            embedding_function = self._embedding_functions[embedding_function]
        if embedding_function is not None and dim is None:
            dim = embedding_function.dimension
        if dim is None or dim <= 0:
            raise DatabaseError("Vector index dim must be a positive integer")
        backend = backend.lower()
        method = method.lower()
        if backend not in ("bruteforce", "faiss", "annoy", "leann", "hnswlib", "usearch", "voyager"):
            raise DatabaseError(f"Unsupported vector backend: {backend}")

        cursor = self.conn.execute("SELECT name FROM vector_indexes WHERE name = ?", (name,))
        if cursor.fetchone():
            if if_not_exists:
                return name
            raise DatabaseError(f"Vector index '{name}' already exists")

        options = options or {}
        # The bruteforce backend has no native on-disk index (unlike faiss/annoy/
        # hnswlib, which get an ``index_path`` sidecar below). Left alone its
        # vectors live only in memory and are lost when the process exits, so a
        # semantic_search over a reopened file database silently returns nothing.
        # Persist them in the SQLite ``vector_entries`` table instead — co-located
        # with the graph, atomic with it, and portable with the .db file. Only for
        # a durable database (":memory:"/"" cannot be reopened, so it would be pure
        # overhead), and only as a default: an explicit ``store_embeddings`` wins.
        if (
            backend == "bruteforce"
            and "store_embeddings" not in options
            and self._db_path not in (":memory:", "")
        ):
            options["store_embeddings"] = True
        if embedding_function is not None:
            self._embedding_functions.setdefault(embedding_function.name(), embedding_function)
            options["embedding_function"] = {
                "name": embedding_function.name(),
                "config": embedding_function.get_config(),
            }
        if "store_embeddings" in options and not isinstance(options["store_embeddings"], bool):
            raise DatabaseError("Vector index option 'store_embeddings' must be a boolean")
        if "default_k" in options:
            if not isinstance(options["default_k"], int) or options["default_k"] <= 0:
                raise DatabaseError("Vector index option 'default_k' must be a positive integer")
        if backend in ("faiss", "annoy", "leann", "hnswlib"):
            options = self._ensure_vector_index_path(name, options, backend)
        try:
            self.conn.execute(
                """
                INSERT INTO vector_indexes (name, dim, backend, method, options)
                VALUES (?, ?, ?, ?, ?)
                """,
                (name, dim, backend, method, orjson.dumps(options).decode('utf-8')),
            )
            self.conn.commit()
        except Exception as exc:
            raise DatabaseError(f"Failed to create vector index '{name}': {exc}", exc)

        self._vector_indexes[name] = self._build_vector_backend(
            backend=backend,
            dim=dim,
            method=method,
            options=options,
        )
        if embedding_function is not None:
            self._vector_index_embeddings[name] = embedding_function
        return name

    def drop_vector_index(self, name: str) -> None:
        """Drop a vector index by name."""
        self._validate_index_identifier(name, "Vector index name")
        cursor = self.conn.execute("SELECT name FROM vector_indexes WHERE name = ?", (name,))
        if not cursor.fetchone():
            raise DatabaseError(f"Vector index '{name}' does not exist")
        cursor = self.conn.execute("SELECT options FROM vector_indexes WHERE name = ?", (name,))
        row = cursor.fetchone()
        if row and row["options"]:
            try:
                options = orjson.loads(row["options"])
                path = options.get("index_path")
                if path and os.path.exists(path):
                    os.remove(path)
            except (OSError, orjson.JSONDecodeError):
                pass
        self.conn.execute("DELETE FROM vector_indexes WHERE name = ?", (name,))
        self.conn.execute("DELETE FROM vector_entries WHERE index_name = ?", (name,))
        self.conn.commit()
        self._vector_indexes.pop(name, None)
        self._vector_index_embeddings.pop(name, None)

    def list_vector_indexes(self) -> list[dict[str, Any]]:
        """List vector index registry entries."""
        cursor = self.conn.execute(
            """
            SELECT name, dim, backend, method, options
            FROM vector_indexes
            ORDER BY name
            """
        )
        rows = []
        for row in cursor.fetchall():
            options = row["options"]
            rows.append(
                {
                    "name": row["name"],
                    "dim": row["dim"],
                    "backend": row["backend"],
                    "method": row["method"],
                    "options": orjson.loads(options) if options else {},
                }
            )
        return rows

    def _commit_if_needed(self) -> None:
        """Commit only when not inside an explicit transaction."""
        if not self._in_transaction:
            self.conn.commit()

    def upsert_embeddings_batch(
        self,
        node_ids: list[int],
        vectors: list[list[float]],
        index: str = "default",
    ) -> None:
        """Insert or update embeddings for many nodes with one ANN update and one persist.

        Unlike calling :meth:`upsert_embedding` in a loop, this mutates the in-memory
        index once (remove+add), writes ``vector_entries`` in bulk when configured,
        commits at most once (skipped inside an explicit transaction), and calls
        :meth:`_persist_vector_index` a single time.
        """
        if len(node_ids) != len(vectors):
            raise DatabaseError("node_ids and vectors length mismatch")
        if not node_ids:
            return
        for node_id in node_ids:
            if not self.get_node(node_id):
                raise NodeNotFoundError(node_id)
        vec_index = self._get_vector_index(index)
        # Remove existing first when the backend can (avoid duplicate FAISS ids).
        if vec_index.supports_remove():
            try:
                vec_index.remove(list(node_ids))
            except Exception:
                pass
        vec_index.add(list(node_ids), list(vectors))
        if vec_index.options.get("store_embeddings"):
            try:
                for node_id, vector in zip(node_ids, vectors):
                    self.conn.execute(
                        """
                        INSERT OR REPLACE INTO vector_entries (index_name, node_id, vector)
                        VALUES (?, ?, ?)
                        """,
                        (index, node_id, orjson.dumps(vector).decode("utf-8")),
                    )
                self._commit_if_needed()
            except Exception as exc:
                raise DatabaseError(f"Failed to persist embeddings: {exc}", exc)
        self._persist_vector_index(index, vec_index)

    def upsert_embedding(self, node_id: int, vector: list[float], index: str = "default") -> None:
        """Insert or update an embedding for a node."""
        self.upsert_embeddings_batch([node_id], [vector], index=index)

    def upsert_embeddings(
        self,
        node_ids: list[int],
        documents: list[str],
        index: str = "default",
    ) -> None:
        """Insert or update embeddings from documents (batch embed + batch upsert)."""
        if len(node_ids) != len(documents):
            raise DatabaseError("node_ids and documents length mismatch")
        if not node_ids:
            return
        embedder = self._get_embedding_function(index)
        if embedder is None:
            raise DatabaseError(f"Vector index '{index}' has no embedding function")
        vectors = embedder(list(documents))
        self.upsert_embeddings_batch(node_ids, vectors, index=index)

    def remove_embeddings_batch(self, node_ids: list[int], index: str = "default") -> None:
        """Remove embeddings for many nodes with one ANN update and one persist."""
        if not node_ids:
            return
        vec_index = self._get_vector_index(index)
        try:
            vec_index.remove(list(node_ids))
        except Exception as exc:
            # Soft-delete backends may still accept the call; hard failures propagate.
            raise DatabaseError(f"Failed to remove embeddings from index '{index}': {exc}", exc)
        if vec_index.options.get("store_embeddings"):
            placeholders = ",".join("?" for _ in node_ids)
            self.conn.execute(
                f"DELETE FROM vector_entries WHERE index_name = ? AND node_id IN ({placeholders})",
                (index, *node_ids),
            )
            self._commit_if_needed()
        self._persist_vector_index(index, vec_index)

    def remove_embedding(self, node_id: int, index: str = "default") -> None:
        """Remove an embedding for a node."""
        self.remove_embeddings_batch([node_id], index=index)

    def semantic_search(
        self,
        vector: list[float] | str,
        k: int | None = None,
        index: str = "default",
        filter_labels: list[str] | LabelFilter | None = None,
        filter_props: dict[str, Any] | PropertyFilterGroup | None = None,
        exact: bool = False,
        rerank: bool = False,
        reranker: Any | None = None,
        candidate_multiplier: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search for nearest nodes using a vector index."""
        if isinstance(vector, str):
            embedder = self._get_embedding_function(index)
            if embedder is None:
                raise DatabaseError(f"Vector index '{index}' has no embedding function")
            vector = embedder([vector])[0]
        vec_index = self._get_vector_index(index)
        if k is None:
            k = int(vec_index.options.get("default_k", self.default_top_k))

        candidate_ids = None
        if filter_labels or filter_props:
            nodes = self.match_nodes(labels=filter_labels or None, properties=filter_props or None)
            candidate_ids = {node.id for node in nodes}
            if not candidate_ids:
                return []

        results: list[tuple[int, float]]
        if candidate_ids and hasattr(vec_index, "search_ids"):
            results = vec_index.search_ids(vector, list(candidate_ids), k)
        else:
            effective_k = k
            if candidate_ids and not exact:
                if candidate_multiplier is not None:
                    effective_k = min(len(candidate_ids), max(k, k * candidate_multiplier))
                else:
                    effective_k = max(k, len(candidate_ids))
            results = vec_index.search(vector, effective_k)

        if candidate_ids:
            results = [(idx, score) for idx, score in results if idx in candidate_ids][:k]

        resolved_reranker = self._resolve_reranker(reranker)
        if resolved_reranker and results:
            reranked = self._apply_custom_reranker(index, vector, results, resolved_reranker)
            if reranked:
                results = reranked[:k]
        elif rerank and results:
            rerank_ids = [idx for idx, _ in results]
            reranked = self._rerank_vectors(index, vector, rerank_ids)
            if reranked:
                results = reranked[:k]

        output = []
        for idx, score in results:
            node = self.get_node(idx)
            if node:
                output.append({"node": node, "score": score})
        return output

    def vector_score(
        self,
        node: "Node | int",
        query: list[float] | str,
        index: str = "default",
    ) -> float | None:
        """Score one node's stored embedding against a query.

        This is the pointwise counterpart of :meth:`semantic_search`: instead of
        asking the index "which nodes are closest?", it asks "how close is *this*
        node?". It backs the Cypher ``VECTOR_SCORE()``/``SIMILAR()`` functions,
        where the candidate set already comes from a pattern match.

        Returns ``None`` when the node has no embedding in ``index`` — an absent
        vector is unknown, not distant, so it must not score as far away.

        Scores follow the index metric and are always "higher is better":
        cosine similarity in ``[-1, 1]``, inner product unbounded, and negated
        squared distance (``<= 0``) for ``l2``.
        """
        node_id = node.id if hasattr(node, "id") else int(node)
        if isinstance(query, str):
            embedder = self._get_embedding_function(index)
            if embedder is None:
                raise DatabaseError(f"Vector index '{index}' has no embedding function")
            query = embedder([query])[0]
        vectors = self._collect_vectors(index, [node_id])
        vector = vectors.get(node_id)
        if vector is None:
            return None
        metric = self._get_vector_index(index).options.get("metric") or "cosine"
        return self._vector_score(metric, query, vector)

    def vector_metric(self, index: str = "default") -> str:
        """Return the similarity metric configured for a vector index."""
        return self._get_vector_index(index).options.get("metric") or "cosine"

    def _resolve_reranker(self, reranker: Any | None) -> Any | None:
        """Resolve reranker name to callable if needed."""
        if reranker is None:
            return None
        if callable(reranker):
            return reranker
        if isinstance(reranker, str):
            if reranker not in self._rerankers:
                raise DatabaseError(f"Unknown reranker '{reranker}'")
            return self._rerankers[reranker]
        raise DatabaseError("Reranker must be a callable or registered name")

    def _apply_custom_reranker(
        self,
        index: str,
        query_vector: list[float],
        results: list[tuple[int, float]],
        reranker: Any,
    ) -> list[tuple[int, float]]:
        """Apply a custom reranker to candidate results."""
        ids = [idx for idx, _ in results]
        vectors = self._collect_vectors(index, ids)
        candidates = []
        for idx, score in results:
            candidates.append(
                {
                    "id": idx,
                    "score": score,
                    "vector": vectors.get(idx),
                    "node": self.get_node(idx),
                }
            )

        reranked = reranker(query_vector, candidates)
        if reranked is None:
            return []
        output = []
        for item in reranked:
            if isinstance(item, dict):
                idx = item.get("id")
                score = item.get("score")
            else:
                idx, score = item
            if idx is None or score is None:
                continue
            output.append((int(idx), float(score)))
        return output

    def _collect_vectors(self, index: str, ids: list[int]) -> dict[int, list[float]]:
        """Collect vectors from the index and optional persisted storage."""
        vec_index = self._get_vector_index(index)
        vectors = {}
        for idx in ids:
            vector = vec_index.get_vector(idx)
            if vector is not None:
                vectors[idx] = vector
        if vec_index.options.get("store_embeddings"):
            missing = [idx for idx in ids if idx not in vectors]
            if missing:
                vectors.update(self._load_vectors_for_ids(index, missing))
        return vectors

    def _get_embedding_function(self, index: str) -> EmbeddingFunction | None:
        embedder = self._vector_index_embeddings.get(index)
        if embedder is not None:
            return embedder
        cursor = self.conn.execute("SELECT options FROM vector_indexes WHERE name = ?", (index,))
        row = cursor.fetchone()
        if not row or not row["options"]:
            return None
        try:
            options = orjson.loads(row["options"])
        except orjson.JSONDecodeError:
            return None
        embedding_config = options.get("embedding_function")
        if not embedding_config:
            return None
        name = embedding_config.get("name")
        if not name:
            return None
        embedder = self._embedding_functions.get(name)
        if embedder is None:
            embedder = create_embedding_function(name, embedding_config.get("config") or {})
        self._vector_index_embeddings[index] = embedder
        return embedder

    def _get_vector_index(self, name: str) -> BruteForceIndex:
        """Get or initialize a vector index backend."""
        if name in self._vector_indexes:
            return self._vector_indexes[name]

        cursor = self.conn.execute(
            "SELECT dim, backend, method, options FROM vector_indexes WHERE name = ?",
            (name,),
        )
        row = cursor.fetchone()
        if not row:
            raise DatabaseError(f"Vector index '{name}' does not exist")
        options = orjson.loads(row["options"]) if row["options"] else {}
        embedding_config = options.get("embedding_function")
        if embedding_config and name not in self._vector_index_embeddings:
            embedder_name = embedding_config.get("name")
            if embedder_name:
                embedder = self._embedding_functions.get(embedder_name)
                if embedder is None:
                    try:
                        embedder = create_embedding_function(
                            embedder_name, embedding_config.get("config") or {}
                        )
                    except Exception:
                        embedder = None
                if embedder is not None:
                    self._vector_index_embeddings[name] = embedder
        if row["backend"] in ("faiss", "annoy", "leann", "hnswlib"):
            options = self._ensure_vector_index_path(name, options, row["backend"])
        vec_index = self._build_vector_backend(
            backend=row["backend"],
            dim=row["dim"],
            method=row["method"],
            options=options,
        )
        loaded_from_disk = self._load_vector_index(name, vec_index)
        # Only load from vector_entries if index wasn't loaded from disk
        # to avoid duplicates (disk file already has the vectors)
        # EXCEPTION: Always load vectors for LEANN backend since it needs them
        # in memory for fallback manual search when index is pruned
        is_leann = row["method"] == "leann" or row["backend"] == "leann"
        if options.get("store_embeddings") and (not loaded_from_disk or is_leann):
            cursor = self.conn.execute(
                "SELECT node_id, vector FROM vector_entries WHERE index_name = ?",
                (name,),
            )
            ids = []
            vectors = []
            for entry in cursor.fetchall():
                ids.append(entry["node_id"])
                vectors.append(orjson.loads(entry["vector"]))
            if ids:
                vec_index.add(ids, vectors)
        self._vector_indexes[name] = vec_index
        return vec_index

    def _build_vector_backend(
        self,
        backend: str,
        dim: int,
        method: str,
        options: dict[str, Any],
    ):
        if backend == "bruteforce":
            return BruteForceIndex(dim=dim, method=method, options=options)
        if backend == "faiss":
            try:
                from .vector_index.faiss import FaissIndex
            except Exception as exc:
                raise DatabaseError(
                    "FAISS backend not available. Install with `pip install grafito[faiss]` "
                    "or `uv pip install grafito[faiss]`."
                ) from exc
            return FaissIndex(dim=dim, method=method, options=options)
        if backend == "annoy":
            try:
                from .vector_index.annoy import AnnoyIndexBackend
            except Exception as exc:
                raise DatabaseError(
                    "Annoy backend not available. Install with `pip install grafito[annoy]` "
                    "or `uv pip install grafito[annoy]`."
                ) from exc
            return AnnoyIndexBackend(dim=dim, method=method, options=options)
        if backend == "hnswlib":
            try:
                from .vector_index.hnswlib import HNSWlibIndexBackend
            except Exception as exc:
                raise DatabaseError(
                    "hnswlib backend not available. Install with `pip install grafito[hnswlib]` "
                    "or `uv pip install grafito[hnswlib]`."
                ) from exc
            return HNSWlibIndexBackend(dim=dim, method=method, options=options)
        if backend == "leann":
            try:
                from .vector_index.leann import LeannIndexBackend
            except Exception as exc:
                raise DatabaseError(
                    "LEANN backend not available. Install with `pip install grafito[leann]` "
                    "or `uv pip install grafito[leann]`."
                ) from exc
            return LeannIndexBackend(dim=dim, method=method, options=options)
        if backend == "usearch":
            try:
                from .vector_index.usearch import USearchIndexBackend
            except Exception as exc:
                raise DatabaseError(
                    "USearch backend not available. Install with `pip install grafito[usearch]` "
                    "or `uv pip install grafito[usearch]`."
                ) from exc
            return USearchIndexBackend(dim=dim, method=method, options=options)
        if backend == "voyager":
            try:
                from .vector_index.voyager import VoyagerIndexBackend
            except Exception as exc:
                raise DatabaseError(
                    "Voyager backend not available. Install with `pip install grafito[voyager]` "
                    "or `uv pip install grafito[voyager]`."
                ) from exc
            return VoyagerIndexBackend(dim=dim, method=method, options=options)
        raise DatabaseError(f"Unsupported vector backend: {backend}")

    def _ensure_vector_index_path(
        self,
        name: str,
        options: dict[str, Any],
        backend: str,
    ) -> dict[str, Any]:
        """Ensure vector index path is set and directory exists.

        The derived sidecar lives in ``.grafito/indexes`` **next to the database
        file**, not under the current working directory, so it travels with the
        ``.db`` when copied or moved and reopens regardless of where the process
        runs. An explicit ``index_path`` always wins, and an already-created index
        keeps the absolute path persisted in its options (this only derives one for
        a new index). ``:memory:``/``""`` has no file to anchor to, so it falls back
        to the working directory — unchanged from before.
        """
        if options.get("index_path"):
            return options
        if self._db_path not in (":memory:", ""):
            anchor = os.path.dirname(os.path.abspath(self._db_path))
        else:
            anchor = os.getcwd()
        base_dir = os.path.join(anchor, ".grafito", "indexes")
        os.makedirs(base_dir, exist_ok=True)
        options = dict(options)
        suffix = "idx"
        if backend == "faiss":
            suffix = "faiss.idx"
        elif backend == "annoy":
            suffix = "annoy"
        elif backend == "leann":
            suffix = "leann"
        elif backend == "hnswlib":
            suffix = "hnswlib"
        elif backend == "usearch":
            suffix = "usearch"
        elif backend == "voyager":
            suffix = "voyager"
        options["index_path"] = os.path.join(base_dir, f"{name}.{suffix}")
        return options

    def _persist_vector_index(self, name: str, vec_index) -> None:
        """Persist vector index to disk when configured."""
        path = vec_index.options.get("index_path")
        if not path:
            return
        try:
            vec_index.save(path)
        except Exception as exc:
            raise DatabaseError(f"Failed to persist vector index '{name}': {exc}", exc)

    def rebuild_vector_index(self, name: str) -> None:
        """Force rebuild/persist of a vector index."""
        vec_index = self._get_vector_index(name)
        # Call rebuild() if the index supports it (for backends with auto_build option)
        if hasattr(vec_index, 'rebuild'):
            vec_index.rebuild()
        self._persist_vector_index(name, vec_index)

    def _load_vector_index(self, name: str, vec_index) -> bool:
        """Load vector index from disk when configured.
        
        Returns:
            True if index was loaded from disk, False otherwise.
        """
        path = vec_index.options.get("index_path")
        if not path or not os.path.exists(path):
            return False
        try:
            vec_index.load(path)
            return True
        except Exception as exc:
            raise DatabaseError(f"Failed to load vector index '{name}': {exc}", exc)

    def _rerank_vectors(
        self,
        index: str,
        query_vector: list[float],
        ids: list[int],
    ) -> list[tuple[int, float]]:
        """Re-rank candidates using exact scoring when embeddings are available."""
        vec_index = self._get_vector_index(index)
        vectors = {}

        for idx in ids:
            vector = vec_index.get_vector(idx)
            if vector is not None:
                vectors[idx] = vector

        if vec_index.options.get("store_embeddings"):
            missing = [idx for idx in ids if idx not in vectors]
            if missing:
                vectors.update(self._load_vectors_for_ids(index, missing))

        if not vectors:
            return []

        metric = vec_index.options.get("metric") or "cosine"
        scored = []
        for idx in ids:
            vector = vectors.get(idx)
            if vector is None:
                continue
            score = self._vector_score(metric, query_vector, vector)
            scored.append((idx, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def _load_vectors_for_ids(self, index: str, ids: list[int]) -> dict[int, list[float]]:
        """Load persisted embeddings for specific node ids."""
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        params = [index] + ids
        cursor = self.conn.execute(
            f"""
            SELECT node_id, vector
            FROM vector_entries
            WHERE index_name = ? AND node_id IN ({placeholders})
            """,
            params,
        )
        return {row["node_id"]: orjson.loads(row["vector"]) for row in cursor.fetchall()}

    def _vector_score(self, metric: str, left: list[float], right: list[float]) -> float:
        """Compute similarity score for two vectors."""
        if metric == "l2":
            total = 0.0
            for a, b in zip(left, right):
                diff = a - b
                total += diff * diff
            return -total
        if metric == "ip":
            return sum(a * b for a, b in zip(left, right))
        # cosine
        dot = 0.0
        left_norm = 0.0
        right_norm = 0.0
        for a, b in zip(left, right):
            dot += a * b
            left_norm += a * a
            right_norm += b * b
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / (left_norm ** 0.5 * right_norm ** 0.5)

    def create_node_uniqueness_constraint(
        self, label: str, property_name: str, name: str | None = None, if_not_exists: bool = False
    ) -> str:
        """Create a uniqueness constraint for nodes."""
        return self._create_constraint(
            "node", label, property_name, "UNIQUE", None, name, if_not_exists
        )

    def create_node_existence_constraint(
        self, label: str, property_name: str, name: str | None = None, if_not_exists: bool = False
    ) -> str:
        """Create a property existence constraint for nodes."""
        return self._create_constraint(
            "node", label, property_name, "EXISTS", None, name, if_not_exists
        )

    def create_node_type_constraint(
        self,
        label: str,
        property_name: str,
        type_name: str,
        name: str | None = None,
        if_not_exists: bool = False,
    ) -> str:
        """Create a property type constraint for nodes."""
        return self._create_constraint(
            "node", label, property_name, "TYPE", type_name, name, if_not_exists
        )

    def create_relationship_uniqueness_constraint(
        self, rel_type: str, property_name: str, name: str | None = None, if_not_exists: bool = False
    ) -> str:
        """Create a uniqueness constraint for relationships."""
        return self._create_constraint(
            "relationship", rel_type, property_name, "UNIQUE", None, name, if_not_exists
        )

    def create_relationship_existence_constraint(
        self, rel_type: str, property_name: str, name: str | None = None, if_not_exists: bool = False
    ) -> str:
        """Create a property existence constraint for relationships."""
        return self._create_constraint(
            "relationship", rel_type, property_name, "EXISTS", None, name, if_not_exists
        )

    def create_relationship_type_constraint(
        self,
        rel_type: str,
        property_name: str,
        type_name: str,
        name: str | None = None,
        if_not_exists: bool = False,
    ) -> str:
        """Create a property type constraint for relationships."""
        return self._create_constraint(
            "relationship", rel_type, property_name, "TYPE", type_name, name, if_not_exists
        )

    def drop_constraint(self, name: str, if_exists: bool = False) -> None:
        """Drop a constraint by name."""
        self._validate_index_identifier(name, "Constraint name")
        cursor = self.conn.execute("SELECT name FROM property_constraints WHERE name = ?", (name,))
        row = cursor.fetchone()
        if not row and not if_exists:
            raise ConstraintError(f"Constraint '{name}' does not exist")
        self.conn.execute("DELETE FROM property_constraints WHERE name = ?", (name,))
        self.conn.commit()

    def list_constraints(self) -> list[dict[str, Any]]:
        """List registered property constraints."""
        cursor = self.conn.execute(
            """
            SELECT name, entity, label_or_type, property, constraint_type, type_name
            FROM property_constraints
            ORDER BY name
            """
        )
        return [
            {
                "name": row["name"],
                "entity": row["entity"],
                "label_or_type": row["label_or_type"],
                "property": row["property"],
                "type": row["constraint_type"],
                "type_name": row["type_name"],
            }
            for row in cursor.fetchall()
        ]

    def _create_constraint(
        self,
        entity: str,
        label_or_type: str,
        property_name: str,
        constraint_type: str,
        type_name: str | None,
        name: str | None,
        if_not_exists: bool,
    ) -> str:
        """Create a constraint and validate existing data."""
        self._validate_index_identifier(entity, "Entity")
        self._validate_index_identifier(label_or_type, "Label/type")
        self._validate_index_identifier(property_name, "Property")
        if type_name:
            self._validate_index_identifier(type_name, "Type")

        constraint_name = name or self._make_constraint_name(entity, label_or_type, property_name, constraint_type)
        self._validate_index_identifier(constraint_name, "Constraint name")

        cursor = self.conn.execute(
            "SELECT name FROM property_constraints WHERE name = ?",
            (constraint_name,),
        )
        if cursor.fetchone():
            if if_not_exists:
                return constraint_name
            raise ConstraintError(f"Constraint '{constraint_name}' already exists")

        # Validate existing data
        if entity == "node":
            nodes = self.match_nodes(labels=[label_or_type])
            self._validate_constraint_rows(
                constraint_name, constraint_type, property_name, type_name, nodes, entity
            )
        else:
            rels = self.match_relationships(rel_type=label_or_type)
            self._validate_constraint_rows(
                constraint_name, constraint_type, property_name, type_name, rels, entity
            )

        self.conn.execute(
            """
            INSERT INTO property_constraints
                (name, entity, label_or_type, property, constraint_type, type_name)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (constraint_name, entity, label_or_type, property_name, constraint_type, type_name),
        )
        self.conn.commit()
        return constraint_name

    def _validate_constraint_rows(
        self,
        constraint_name: str,
        constraint_type: str,
        property_name: str,
        type_name: str | None,
        rows: list[Any],
        entity: str,
        existing_id: int | None = None,
    ) -> None:
        """Validate existing rows for a constraint."""
        seen = {}
        for row in rows:
            value = row.properties.get(property_name)
            if constraint_type == "EXISTS":
                if value is None:
                    raise ConstraintError(
                        f"Constraint '{constraint_name}' violated: {entity} missing '{property_name}'"
                    )
            elif constraint_type == "TYPE":
                if value is None:
                    raise ConstraintError(
                        f"Constraint '{constraint_name}' violated: {entity} missing '{property_name}'"
                    )
                if not self._value_matches_type(value, type_name or ""):
                    raise ConstraintError(
                        f"Constraint '{constraint_name}' violated: {entity} '{property_name}' has wrong type"
                    )
            elif constraint_type == "UNIQUE":
                if value is None:
                    continue
                if row.id == existing_id:
                    continue
                key = orjson.dumps(value, option=orjson.OPT_SORT_KEYS).decode('utf-8')
                if key in seen:
                    raise ConstraintError(
                        f"Constraint '{constraint_name}' violated: duplicate '{property_name}'"
                    )
                seen[key] = row.id

    def _value_matches_type(self, value: Any, type_name: str) -> bool:
        """Check if a value matches a constraint type name."""
        type_name = type_name.upper()
        if type_name == "STRING":
            return isinstance(value, str)
        if type_name == "INTEGER":
            return isinstance(value, int) and not isinstance(value, bool)
        if type_name == "FLOAT":
            return isinstance(value, float)
        if type_name == "BOOLEAN":
            return isinstance(value, bool)
        if type_name == "LIST":
            return isinstance(value, list)
        if type_name == "MAP":
            return isinstance(value, dict)
        return False

    def _validate_constraints_on_node(self, labels: list[str], properties: dict, node_id: int | None = None) -> None:
        """Validate node constraints for the given labels and properties."""
        for label in labels:
            constraints = self._get_constraints("node", label)
            for constraint in constraints:
                self._validate_single_constraint(
                    constraint, properties, node_id=node_id, entity="node", label_or_type=label
                )

    def _validate_constraints_on_relationship(self, rel_type: str, properties: dict, rel_id: int | None = None) -> None:
        """Validate relationship constraints."""
        constraints = self._get_constraints("relationship", rel_type)
        for constraint in constraints:
            self._validate_single_constraint(
                constraint, properties, rel_id=rel_id, entity="relationship", label_or_type=rel_type
            )

    def _get_constraints(self, entity: str, label_or_type: str) -> list[dict[str, Any]]:
        """Load constraints for an entity and label/type."""
        cursor = self.conn.execute(
            """
            SELECT name, property, constraint_type, type_name
            FROM property_constraints
            WHERE entity = ? AND label_or_type = ?
            """,
            (entity, label_or_type),
        )
        return [
            {
                "name": row["name"],
                "property": row["property"],
                "type": row["constraint_type"],
                "type_name": row["type_name"],
            }
            for row in cursor.fetchall()
        ]

    def _validate_single_constraint(
        self,
        constraint: dict[str, Any],
        properties: dict,
        node_id: int | None = None,
        rel_id: int | None = None,
        entity: str = "node",
        label_or_type: str = "",
    ) -> None:
        """Validate a constraint against current properties."""
        prop = constraint["property"]
        ctype = constraint["type"]
        type_name = constraint["type_name"]
        value = properties.get(prop)

        if ctype == "EXISTS":
            if value is None:
                raise ConstraintError(
                    f"Constraint '{constraint['name']}' violated: missing '{prop}'"
                )
            return
        if ctype == "TYPE":
            if value is None:
                raise ConstraintError(
                    f"Constraint '{constraint['name']}' violated: missing '{prop}'"
                )
            if not self._value_matches_type(value, type_name or ""):
                raise ConstraintError(
                    f"Constraint '{constraint['name']}' violated: '{prop}' has wrong type"
                )
            return
        if ctype == "UNIQUE":
            if value is None:
                return
            existing_id = node_id if entity == "node" else rel_id
            if self._has_duplicate_property(entity, label_or_type, prop, value, existing_id):
                raise ConstraintError(
                    f"Constraint '{constraint['name']}' violated: duplicate '{prop}'"
                )

    def _has_duplicate_property(
        self,
        entity: str,
        label_or_type: str,
        property_name: str,
        value: Any,
        existing_id: int | None,
    ) -> bool:
        """Check for duplicate property values for uniqueness constraints."""
        if isinstance(value, (list, dict)):
            if entity == "node":
                nodes = self.match_nodes(labels=[label_or_type])
                for node in nodes:
                    if existing_id is not None and node.id == existing_id:
                        continue
                    if node.properties.get(property_name) == value:
                        return True
                return False
            rels = self.match_relationships(rel_type=label_or_type)
            for rel in rels:
                if existing_id is not None and rel.id == existing_id:
                    continue
                if rel.properties.get(property_name) == value:
                    return True
            return False

        if entity == "node":
            sql = """
                SELECT n.id
                FROM nodes n
                JOIN node_labels nl ON n.id = nl.node_id
                JOIN labels l ON l.id = nl.label_id
                WHERE l.name = ?
                  AND json_extract(n.properties, ?) = ?
            """
            params = [label_or_type, f"$.{property_name}", value]
            if existing_id is not None:
                sql += " AND n.id != ?"
                params.append(existing_id)
        else:
            sql = """
                SELECT id
                FROM relationships
                WHERE type = ?
                  AND json_extract(properties, ?) = ?
            """
            params = [label_or_type, f"$.{property_name}", value]
            if existing_id is not None:
                sql += " AND id != ?"
                params.append(existing_id)

        cursor = self.conn.execute(sql, params)
        return cursor.fetchone() is not None

    def _get_node_labels(self, node_id: int) -> list[str]:
        """Fetch all labels for a node.

        Args:
            node_id: ID of the node

        Returns:
            List of label names (sorted alphabetically)
        """
        cursor = self.conn.execute(
            """
            SELECT l.name
            FROM labels l
            JOIN node_labels nl ON l.id = nl.label_id
            WHERE nl.node_id = ?
            ORDER BY l.name
            """,
            (node_id,),
        )
        return [row['name'] for row in cursor.fetchall()]

    def _ensure_label_exists(self, label: str) -> int:
        """Get or create a label, returning its ID.

        Args:
            label: Label name

        Returns:
            Label ID
        """
        # Try to get existing label
        cursor = self.conn.execute("SELECT id FROM labels WHERE name = ?", (label,))
        row = cursor.fetchone()
        if row:
            return row['id']

        # Create new label
        cursor = self.conn.execute("INSERT INTO labels (name) VALUES (?)", (label,))
        return cursor.lastrowid

    def _escape_like_pattern(self, value: str) -> str:
        """Escape LIKE special characters in a string.

        Args:
            value: String to escape

        Returns:
            Escaped string safe for LIKE patterns
        """
        return value.replace('\\', '\\\\').replace('%', r'\%').replace('_', r'\_')

    def _build_property_conditions(
        self, properties: dict | PropertyFilterGroup, table_alias: str = 'n'
    ) -> tuple[list[str], list[Any]]:
        """Build WHERE conditions for property filtering.

        Handles:
        - Exact matching (backward compatible)
        - Comparison operators (>, <, >=, <=, !=, BETWEEN)
        - String pattern matching (CONTAINS, STARTS_WITH, ENDS_WITH, REGEX)
        - OR/AND logic via PropertyFilterGroup

        Args:
            properties: Property filter dictionary or PropertyFilterGroup
            table_alias: SQL table alias (default 'n' for nodes, 'r' for relationships)

        Returns:
            Tuple of (conditions_list, params_list)

        Raises:
            InvalidFilterError: If filter specification is invalid
        """
        conditions = []
        params = []

        if not properties:
            return conditions, params

        # Handle PropertyFilterGroup at top level
        if isinstance(properties, PropertyFilterGroup):
            group_conditions = []
            for filter_dict in properties.filters:
                sub_conds, sub_params = self._build_property_conditions(
                    filter_dict, table_alias
                )
                if sub_conds:
                    group_conditions.append(f"({' AND '.join(sub_conds)})")
                    params.extend(sub_params)

            if group_conditions:
                operator = ' OR ' if properties.operator == 'OR' else ' AND '
                conditions.append(f"({operator.join(group_conditions)})")

            return conditions, params

        # Regular dict processing
        for key, value in properties.items():
            # Validate property key (prevent SQL injection)
            if not re.match(r'^[a-zA-Z0-9_\.]+$', key):
                raise InvalidFilterError(
                    f"Invalid property key '{key}'. "
                    "Only alphanumeric, underscore, and dot characters allowed."
                )

            json_path = f"$.{key}"
            json_expr = f"json_extract({table_alias}.properties, '{json_path}')"

            # PropertyFilterGroup (OR/AND combinations)
            if isinstance(value, PropertyFilterGroup):
                group_conditions = []
                for filter_dict in value.filters:
                    sub_conds, sub_params = self._build_property_conditions(
                        filter_dict, table_alias
                    )
                    if sub_conds:
                        group_conditions.append(f"({' AND '.join(sub_conds)})")
                        params.extend(sub_params)

                if group_conditions:
                    operator = ' OR ' if value.operator == 'OR' else ' AND '
                    conditions.append(f"({operator.join(group_conditions)})")

            # PropertyFilter (comparison/pattern matching)
            elif isinstance(value, PropertyFilter):
                normalized_value = self._normalize_property_value(key, value.value)
                normalized_value2 = None
                if value.value2 is not None:
                    normalized_value2 = self._normalize_property_value(key, value.value2)
                op = value.operator

                if op == 'BETWEEN':
                    if isinstance(normalized_value, str) and isinstance(normalized_value2, str):
                        conditions.append(
                            f"(typeof({json_expr}) = 'text' AND {json_expr} BETWEEN ? AND ?)"
                        )
                    else:
                        conditions.append(
                            f"(typeof({json_expr}) IN ('integer','real') AND {json_expr} BETWEEN ? AND ?)"
                        )
                    params.extend([normalized_value, normalized_value2])

                elif op == '!=':
                    # Handle NULL: NULL != X is NULL (not TRUE)
                    conditions.append(
                        f"({json_expr} != ? OR {json_expr} IS NULL)"
                    )
                    params.append(normalized_value)

                elif op in ('>', '<', '>=', '<='):
                    if isinstance(normalized_value, str):
                        conditions.append(
                            f"(typeof({json_expr}) = 'text' AND {json_expr} {op} ?)"
                        )
                    else:
                        conditions.append(
                            f"(typeof({json_expr}) IN ('integer','real') AND {json_expr} {op} ?)"
                        )
                    params.append(normalized_value)

                elif op in ('CONTAINS', 'STARTS_WITH', 'ENDS_WITH'):
                    if not value.case_sensitive:
                        json_expr = f"LOWER({json_expr})"
                        search_value = str(normalized_value).lower()
                    else:
                        search_value = str(normalized_value)

                    if value.case_sensitive:
                        if op == 'CONTAINS':
                            conditions.append(f"instr({json_expr}, ?) > 0")
                            params.append(search_value)
                        elif op == 'STARTS_WITH':
                            conditions.append(f"substr({json_expr}, 1, ?) = ?")
                            params.extend([len(search_value), search_value])
                        else:  # ENDS_WITH
                            conditions.append(f"substr({json_expr}, -?) = ?")
                            params.extend([len(search_value), search_value])
                    else:
                        # Escape LIKE wildcards for case-insensitive matching
                        search_value = self._escape_like_pattern(search_value)
                        if op == 'CONTAINS':
                            pattern = f"%{search_value}%"
                        elif op == 'STARTS_WITH':
                            pattern = f"{search_value}%"
                        else:  # ENDS_WITH
                            pattern = f"%{search_value}"
                        conditions.append(f"{json_expr} LIKE ? ESCAPE '\\'")
                        params.append(pattern)

                elif op == 'REGEX':
                    # Regular expression matching (uses custom SQLite function)
                    conditions.append(f"regex(?, {json_expr})")
                    params.append(normalized_value)

                else:
                    raise InvalidFilterError(f"Unknown operator '{op}'")

            # Exact match (backward compatible)
            elif value is None:
                conditions.append(f"{json_expr} IS NULL")
            else:
                normalized_value = self._normalize_property_value(key, value)
                # For list/dict, we need to compare as JSON strings
                if isinstance(normalized_value, (list, dict)):
                    # Compare the JSON representation
                    conditions.append(f"{json_expr} = json(?)")
                    params.append(
                        orjson.dumps(normalized_value, option=orjson.OPT_SORT_KEYS).decode('utf-8')
                    )
                else:
                    conditions.append(f"{json_expr} = ?")
                    params.append(normalized_value)

        return conditions, params

    def _build_order_clause(
        self,
        order_by: str | list[str] | list[SortOrder],
        ascending: bool,
        table_alias: str = 'n'
    ) -> str:
        """Build ORDER BY clause from various input formats.

        Args:
            order_by: Property name, list of names, or list of SortOrder objects
            ascending: Default direction (ignored if SortOrder objects provided)
            table_alias: SQL table alias

        Returns:
            SQL ORDER BY clause (empty string if no ordering)

        Raises:
            InvalidFilterError: If property name is invalid
        """
        if not order_by:
            return ""

        order_parts = []

        if isinstance(order_by, str):
            # Single property name
            if not re.match(r'^[a-zA-Z0-9_\.]+$', order_by):
                raise InvalidFilterError(f"Invalid property name '{order_by}'")

            direction = 'ASC' if ascending else 'DESC'
            expr = f"json_extract({table_alias}.properties, '$.{order_by}')"
            order_parts.append(f"{expr} IS NULL, {expr} {direction}")

        elif isinstance(order_by, list):
            if not order_by:
                return ""

            if isinstance(order_by[0], SortOrder):
                # List of SortOrder objects
                for sort in order_by:
                    if not re.match(r'^[a-zA-Z0-9_\.]+$', sort.property):
                        raise InvalidFilterError(f"Invalid property name '{sort.property}'")
                    direction = 'ASC' if sort.ascending else 'DESC'
                    expr = f"json_extract({table_alias}.properties, '$.{sort.property}')"
                    order_parts.append(f"{expr} IS NULL, {expr} {direction}")
            else:
                # List of property names
                direction = 'ASC' if ascending else 'DESC'
                for prop in order_by:
                    if not re.match(r'^[a-zA-Z0-9_\.]+$', prop):
                        raise InvalidFilterError(f"Invalid property name '{prop}'")
                    expr = f"json_extract({table_alias}.properties, '$.{prop}')"
                    order_parts.append(f"{expr} IS NULL, {expr} {direction}")

        if order_parts:
            return f"ORDER BY {', '.join(order_parts)}"
        return ""

    # =========================================================================
    # Node Operations (to be implemented in Phase 2)
    # =========================================================================

    def create_node(
        self, labels: list[str] = None, properties: dict = None, uri: str | None = None
    ) -> Node:
        """Create a new node in the graph.

        Args:
            labels: List of labels to assign to the node
            properties: Dictionary of properties (key-value pairs)
            uri: Optional URI for RDF export or external identity

        Returns:
            Created Node object

        Raises:
            InvalidPropertyError: If properties contain unsupported types
            DatabaseError: If node creation fails
        """
        labels = labels or []
        properties = properties or {}

        # Validate properties
        properties = self._validate_properties(properties)
        self._validate_constraints_on_node(labels, properties)

        try:
            # Serialize properties to JSON
            properties_json = orjson.dumps(properties).decode('utf-8')

            # Insert node
            cursor = self.conn.execute(
                "INSERT INTO nodes (properties, uri) VALUES (?, ?)",
                (properties_json, uri),
            )
            node_id = cursor.lastrowid

            # Insert labels
            for label in labels:
                label_id = self._ensure_label_exists(label)
                self.conn.execute(
                    "INSERT INTO node_labels (node_id, label_id) VALUES (?, ?)",
                    (node_id, label_id),
                )

            # Commit if not in transaction
            if not self._in_transaction:
                self.conn.commit()

            # Return Node object
            return Node(
                id=node_id,
                labels=labels.copy(),
                properties=properties.copy(),
                uri=uri,
            )

        except Exception as e:
            if not self._in_transaction:
                self.conn.rollback()
            raise DatabaseError(f"Failed to create node: {e}", e)

    def merge_node(
        self,
        labels: list[str] = None,
        match_properties: dict = None,
        on_create: dict = None,
        on_match: dict = None,
    ) -> tuple[Node, bool]:
        """Find or create a node based on match criteria.
        
        Similar to Cypher's MERGE: if a node with the given labels and
        match_properties exists, return it (optionally updating with on_match).
        Otherwise, create a new node with the combined properties.
        
        Args:
            labels: Labels the node must have (for matching and creation).
            match_properties: Properties to match on. All must match exactly.
            on_create: Additional properties to set only when creating a new node.
            on_match: Additional properties to set only when matching existing node.
        
        Returns:
            Tuple of (Node, created) where created is True if node was created.
        
        Example:
            # Idempotent node creation
            node, created = db.merge_node(
                labels=["Person"],
                match_properties={"email": "user@example.com"},
                on_create={"created_at": "2024-01-01"},
                on_match={"last_seen": "2024-01-15"},
            )
        """
        labels = labels or []
        match_properties = match_properties or {}
        on_create = on_create or {}
        on_match = on_match or {}
        
        # Try to find existing node
        existing = self.match_nodes(labels=labels, properties=match_properties, limit=1)
        
        if existing:
            node = existing[0]
            # Apply on_match properties if any
            if on_match:
                self.update_node_properties(node.id, on_match)
                # Refresh node to get updated properties
                node = self.get_node(node.id)
            return (node, False)
        
        # Create new node with all properties
        all_properties = {**match_properties, **on_create}
        node = self.create_node(labels=labels, properties=all_properties)
        return (node, True)

    def get_node(self, node_id: int) -> Node | None:
        """Get a node by its ID.

        Args:
            node_id: Node ID

        Returns:
            Node object if found, None otherwise
        """
        # Get node data
        cursor = self.conn.execute(
            "SELECT id, properties, uri FROM nodes WHERE id = ?", (node_id,)
        )
        row = cursor.fetchone()

        if not row:
            return None

        # Deserialize properties
        properties = orjson.loads(row['properties'])

        # Get labels
        labels = self._get_node_labels(node_id)

        return Node(id=row['id'], labels=labels, properties=properties, uri=row['uri'])

    def update_node_properties(self, node_id: int, properties: dict) -> bool:
        """Update node properties (merges with existing).

        Args:
            node_id: Node ID
            properties: Properties to update/add

        Returns:
            True if successful

        Raises:
            NodeNotFoundError: If node doesn't exist
            InvalidPropertyError: If properties contain unsupported types
        """
        # Validate new properties
        properties = self._validate_properties(properties)

        # Get existing node
        node = self.get_node(node_id)
        if not node:
            raise NodeNotFoundError(node_id)

        # Merge properties
        merged_properties = {**node.properties, **properties}
        self._validate_constraints_on_node(node.labels, merged_properties, node_id=node_id)

        # Serialize and update
        properties_json = orjson.dumps(merged_properties).decode('utf-8')
        self.conn.execute(
            "UPDATE nodes SET properties = ? WHERE id = ?",
            (properties_json, node_id),
        )

        # Commit if not in transaction
        if not self._in_transaction:
            self.conn.commit()

        return True

    def replace_node_properties(self, node_id: int, properties: dict) -> bool:
        """Replace node properties.

        Args:
            node_id: Node ID
            properties: Properties to set

        Returns:
            True if successful
        """
        properties = self._validate_properties(properties)
        node = self.get_node(node_id)
        if not node:
            raise NodeNotFoundError(node_id)

        self._validate_constraints_on_node(node.labels, properties, node_id=node_id)
        properties_json = orjson.dumps(properties).decode('utf-8')
        self.conn.execute(
            "UPDATE nodes SET properties = ? WHERE id = ?",
            (properties_json, node_id),
        )
        if not self._in_transaction:
            self.conn.commit()
        return True

    def add_labels(self, node_id: int, labels: list[str]) -> bool:
        """Add labels to a node.

        Args:
            node_id: Node ID
            labels: Labels to add

        Returns:
            True if successful

        Raises:
            NodeNotFoundError: If node doesn't exist
        """
        # Check node exists
        if not self.get_node(node_id):
            raise NodeNotFoundError(node_id)

        # Add each label
        for label in labels:
            label_id = self._ensure_label_exists(label)

            # Insert if not already exists (ignore if duplicate)
            try:
                self.conn.execute(
                    "INSERT INTO node_labels (node_id, label_id) VALUES (?, ?)",
                    (node_id, label_id),
                )
            except sqlite3.IntegrityError:
                # Label already exists on node, skip
                pass

        # Commit if not in transaction
        if not self._in_transaction:
            self.conn.commit()

        return True

    def remove_labels(self, node_id: int, labels: list[str]) -> bool:
        """Remove labels from a node.

        Args:
            node_id: Node ID
            labels: Labels to remove

        Returns:
            True if successful

        Raises:
            NodeNotFoundError: If node doesn't exist
        """
        # Check node exists
        if not self.get_node(node_id):
            raise NodeNotFoundError(node_id)

        # Remove each label
        for label in labels:
            # Get label ID
            cursor = self.conn.execute(
                "SELECT id FROM labels WHERE name = ?", (label,)
            )
            row = cursor.fetchone()

            if row:
                label_id = row['id']
                # Delete from junction table
                self.conn.execute(
                    "DELETE FROM node_labels WHERE node_id = ? AND label_id = ?",
                    (node_id, label_id),
                )

        # Commit if not in transaction
        if not self._in_transaction:
            self.conn.commit()

        return True

    def delete_node(self, node_id: int) -> bool:
        """Delete a node (cascades to relationships).

        Best-effort: also removes the node's embeddings from all registered
        vector indexes so ANN backends do not retain ghost ids. Backends that
        only soft-delete may still need :meth:`rebuild_vector_index`.

        Args:
            node_id: Node ID

        Returns:
            True if node was deleted, False if node didn't exist
        """
        if self.get_node(node_id) is None:
            return False

        for idx in self.list_vector_indexes():
            name = idx.get("name")
            if not name:
                continue
            try:
                self.remove_embedding(node_id, index=name)
            except Exception:
                # Index may not contain the id or backend may lack durable remove.
                pass

        # Delete node (CASCADE will handle relationships and labels)
        cursor = self.conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))

        # Commit if not in transaction
        if not self._in_transaction:
            self.conn.commit()

        # Return True if a row was deleted
        return cursor.rowcount > 0

    # =========================================================================
    # Relationship Operations (to be implemented in Phase 3)
    # =========================================================================

    def create_relationship(
        self,
        source_id: int,
        target_id: int,
        rel_type: str,
        properties: dict = None,
        uri: str | None = None,
    ) -> Relationship:
        """Create a directed relationship between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            rel_type: Relationship type (e.g., 'WORKS_AT', 'KNOWS')
            properties: Dictionary of properties
            uri: Optional URI for RDF export or external identity

        Returns:
            Created Relationship object

        Raises:
            NodeNotFoundError: If source or target node doesn't exist
            InvalidPropertyError: If properties contain unsupported types
            DatabaseError: If relationship creation fails
        """
        properties = properties or {}

        # Validate that both nodes exist
        source_node = self.get_node(source_id)
        if not source_node:
            raise NodeNotFoundError(source_id)

        target_node = self.get_node(target_id)
        if not target_node:
            raise NodeNotFoundError(target_id)

        # Validate properties
        properties = self._validate_properties(properties)
        self._validate_constraints_on_relationship(rel_type, properties)

        try:
            # Serialize properties to JSON
            properties_json = orjson.dumps(properties).decode('utf-8')

            # Insert relationship
            cursor = self.conn.execute(
                """
                INSERT INTO relationships (source_node_id, target_node_id, type, properties, uri)
                VALUES (?, ?, ?, ?, ?)
                """,
                (source_id, target_id, rel_type, properties_json, uri),
            )
            rel_id = cursor.lastrowid

            # Commit if not in transaction
            if not self._in_transaction:
                self.conn.commit()

            # Return Relationship object
            return Relationship(
                id=rel_id,
                source_id=source_id,
                target_id=target_id,
                type=rel_type,
                properties=properties.copy(),
                uri=uri,
            )

        except Exception as e:
            if not self._in_transaction:
                self.conn.rollback()
            raise DatabaseError(f"Failed to create relationship: {e}", e)

    def get_relationship(self, rel_id: int) -> Relationship | None:
        """Get a relationship by its ID.

        Args:
            rel_id: Relationship ID

        Returns:
            Relationship object if found, None otherwise
        """
        # Get relationship data
        cursor = self.conn.execute(
            """
            SELECT id, source_node_id, target_node_id, type, properties, uri
            FROM relationships
            WHERE id = ?
            """,
            (rel_id,),
        )
        row = cursor.fetchone()

        if not row:
            return None

        # Deserialize properties
        properties = orjson.loads(row['properties'])

        return Relationship(
            id=row['id'],
            source_id=row['source_node_id'],
            target_id=row['target_node_id'],
            type=row['type'],
            properties=properties,
            uri=row['uri'],
        )

    def update_relationship_properties(self, rel_id: int, properties: dict) -> bool:
        """Update relationship properties (merges with existing)."""
        properties = self._validate_properties(properties)
        rel = self.get_relationship(rel_id)
        if not rel:
            raise RelationshipNotFoundError(rel_id)

        merged_properties = {**rel.properties, **properties}
        self._validate_constraints_on_relationship(rel.type, merged_properties, rel_id=rel_id)

        properties_json = orjson.dumps(merged_properties).decode('utf-8')
        self.conn.execute(
            "UPDATE relationships SET properties = ? WHERE id = ?",
            (properties_json, rel_id),
        )
        if not self._in_transaction:
            self.conn.commit()
        return True

    def replace_relationship_properties(self, rel_id: int, properties: dict) -> bool:
        """Replace relationship properties."""
        properties = self._validate_properties(properties)
        rel = self.get_relationship(rel_id)
        if not rel:
            raise RelationshipNotFoundError(rel_id)

        self._validate_constraints_on_relationship(rel.type, properties, rel_id=rel_id)

        properties_json = orjson.dumps(properties).decode('utf-8')
        self.conn.execute(
            "UPDATE relationships SET properties = ? WHERE id = ?",
            (properties_json, rel_id),
        )
        if not self._in_transaction:
            self.conn.commit()
        return True

    def delete_relationship(self, rel_id: int) -> bool:
        """Delete a relationship.

        Args:
            rel_id: Relationship ID

        Returns:
            True if relationship was deleted, False if it didn't exist
        """
        # Delete relationship
        cursor = self.conn.execute(
            "DELETE FROM relationships WHERE id = ?", (rel_id,)
        )

        # Commit if not in transaction
        if not self._in_transaction:
            self.conn.commit()

        # Return True if a row was deleted
        return cursor.rowcount > 0

    # =========================================================================
    # Pattern Matching (to be implemented in Phase 4)
    # =========================================================================

    def match_nodes(
        self,
        labels: list[str] | LabelFilter = None,
        properties: dict = None,
        order_by: str | list[str] | list[SortOrder] = None,
        ascending: bool = True,
        limit: int = None,
        offset: int = None,
    ) -> list[Node]:
        """Find nodes matching a pattern with advanced filtering, ordering, and pagination.

        Args:
            labels: Labels filter - can be:
                - list[str]: Nodes must have ALL labels (AND logic, backward compatible)
                - LabelFilter.any(): Nodes with ANY of the labels (OR logic)
                - LabelFilter.all(): Nodes with ALL labels (explicit AND)
            properties: Property filters - can be:
                - dict with exact values: {'name': 'Alice', 'age': 30} (backward compatible)
                - dict with PropertyFilter objects: {'age': PropertyFilter.gt(30)}
                - PropertyFilterGroup for OR/AND combinations
            order_by: Ordering specification:
                - str: Single property name
                - list[str]: Multiple properties (same direction)
                - list[SortOrder]: Multiple properties with individual directions
            ascending: Default sort direction (ignored if using SortOrder objects)
            limit: Maximum number of results to return
            offset: Number of results to skip (for pagination)

        Returns:
            List of matching Node objects

        Raises:
            InvalidFilterError: If filter specification is invalid

        Examples:
            Basic (backward compatible):
                >>> db.match_nodes(labels=['Person'], properties={'age': 30})

            Comparison operators:
                >>> db.match_nodes(properties={'age': PropertyFilter.gt(30)})
                >>> db.match_nodes(properties={'age': PropertyFilter.between(25, 35)})

            String matching:
                >>> db.match_nodes(properties={
                ...     'name': PropertyFilter.contains('alice', case_sensitive=False)
                ... })

            OR logic:
                >>> db.match_nodes(labels=LabelFilter.any(['Person', 'Company']))
                >>> db.match_nodes(properties=PropertyFilterGroup.or_(
                ...     {'city': 'NYC'},
                ...     {'city': 'LA'}
                ... ))

            Ordering and pagination:
                >>> db.match_nodes(
                ...     labels=['Person'],
                ...     order_by='age',
                ...     ascending=False,
                ...     limit=10
                ... )
        """
        properties = properties or {}

        # Build query components separately so that property predicates land in the
        # WHERE clause (before GROUP BY). Appending them after a HAVING clause would
        # push them into HAVING, which is evaluated post-aggregation and prevents
        # SQLite from using property indexes (forcing a full scan). See #match_nodes.
        from_join = "FROM nodes n"
        where_conditions: list[str] = []
        where_params: list[Any] = []
        group_having = ""
        having_params: list[Any] = []

        if isinstance(labels, LabelFilter):
            from_join = (
                "FROM nodes n\n"
                "            JOIN node_labels nl ON n.id = nl.node_id\n"
                "            JOIN labels l ON nl.label_id = l.id"
            )
            placeholders = ','.join('?' * len(labels.labels))
            where_conditions.append(f"l.name IN ({placeholders})")
            where_params.extend(labels.labels)
            if labels.operator != 'OR':
                # AND: Node must have ALL labels (with HAVING COUNT)
                group_having = "GROUP BY n.id\n            HAVING COUNT(DISTINCT l.name) = ?"
                having_params.append(len(labels.labels))

        elif labels:
            # Plain list: AND logic (backward compatible)
            from_join = (
                "FROM nodes n\n"
                "            JOIN node_labels nl ON n.id = nl.node_id\n"
                "            JOIN labels l ON nl.label_id = l.id"
            )
            placeholders = ','.join('?' * len(labels))
            where_conditions.append(f"l.name IN ({placeholders})")
            where_params.extend(labels)
            group_having = "GROUP BY n.id\n            HAVING COUNT(DISTINCT l.name) = ?"
            having_params.append(len(labels))

        # Build property conditions (kept in WHERE, before any GROUP BY/HAVING)
        prop_conditions, prop_params = self._build_property_conditions(properties, 'n')
        where_conditions.extend(prop_conditions)

        # Params must follow placeholder order in the final SQL:
        # WHERE (labels, then properties) -> HAVING (label count)
        params: list[Any] = [*where_params, *prop_params, *having_params]

        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)

        # Build ORDER BY clause
        order_clause = self._build_order_clause(order_by, ascending, 'n')

        # Build LIMIT/OFFSET clause
        limit_clause = ""
        if limit is not None:
            if limit < 0:
                raise InvalidFilterError("Limit must be non-negative")
            limit_clause = f" LIMIT {limit}"
            if offset is not None:
                if offset < 0:
                    raise InvalidFilterError("Offset must be non-negative")
                limit_clause += f" OFFSET {offset}"

        # Construct final query
        query = f"""
            SELECT DISTINCT n.id
            {from_join}
            {where_clause}
            {group_having}
            {order_clause}
            {limit_clause}
        """

        cursor = self.conn.execute(query, params)
        node_ids = [row['id'] for row in cursor.fetchall()]

        # Fetch full node objects
        matching_nodes = []
        for node_id in node_ids:
            node = self.get_node(node_id)
            if node:
                matching_nodes.append(node)

        return matching_nodes

    def match_relationships(
        self,
        source_id: int = None,
        target_id: int = None,
        rel_type: str = None,
        properties: dict = None,
        order_by: str | list[str] | list[SortOrder] = None,
        ascending: bool = True,
        limit: int = None,
        offset: int = None,
    ) -> list[Relationship]:
        """Find relationships matching criteria with advanced filtering, ordering, and pagination.

        Args:
            source_id: Filter by source node ID
            target_id: Filter by target node ID
            rel_type: Filter by relationship type
            properties: Property filters - can be:
                - dict with exact values: {'since': 2020} (backward compatible)
                - dict with PropertyFilter objects: {'since': PropertyFilter.gt(2020)}
                - PropertyFilterGroup for OR/AND combinations
            order_by: Ordering specification:
                - str: Single property name
                - list[str]: Multiple properties (same direction)
                - list[SortOrder]: Multiple properties with individual directions
            ascending: Default sort direction (ignored if using SortOrder objects)
            limit: Maximum number of results to return
            offset: Number of results to skip (for pagination)

        Returns:
            List of matching Relationship objects

        Raises:
            InvalidFilterError: If filter specification is invalid

        Examples:
            Basic (backward compatible):
                >>> db.match_relationships(source_id=1, rel_type='KNOWS')

            Comparison operators:
                >>> db.match_relationships(properties={'since': PropertyFilter.gte(2020)})

            Ordering and pagination:
                >>> db.match_relationships(
                ...     rel_type='WORKS_AT',
                ...     order_by='since',
                ...     ascending=False,
                ...     limit=10
                ... )
        """
        properties = properties or {}

        # Build WHERE clause dynamically
        conditions = []
        params = []

        if source_id is not None:
            conditions.append("r.source_node_id = ?")
            params.append(source_id)

        if target_id is not None:
            conditions.append("r.target_node_id = ?")
            params.append(target_id)

        if rel_type is not None:
            conditions.append("r.type = ?")
            params.append(rel_type)

        # Build property conditions
        prop_conditions, prop_params = self._build_property_conditions(properties, 'r')
        conditions.extend(prop_conditions)
        params.extend(prop_params)

        # Construct WHERE clause
        where_clause = ""
        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)

        # Build ORDER BY clause
        order_clause = self._build_order_clause(order_by, ascending, 'r')

        # Build LIMIT/OFFSET clause
        limit_clause = ""
        if limit is not None:
            if limit < 0:
                raise InvalidFilterError("Limit must be non-negative")
            limit_clause = f" LIMIT {limit}"
            if offset is not None:
                if offset < 0:
                    raise InvalidFilterError("Offset must be non-negative")
                limit_clause += f" OFFSET {offset}"

        # Construct query
        query = f"""
            SELECT r.id, r.source_node_id, r.target_node_id, r.type, r.properties, r.uri
            FROM relationships r
            {where_clause}
            {order_clause}
            {limit_clause}
        """

        # Execute query
        cursor = self.conn.execute(query, params)
        rows = cursor.fetchall()

        # Build Relationship objects
        relationships = []
        for row in rows:
            rel_properties = orjson.loads(row['properties'])
            rel = Relationship(
                id=row['id'],
                source_id=row['source_node_id'],
                target_id=row['target_node_id'],
                type=row['type'],
                properties=rel_properties,
                uri=row['uri'],
            )
            relationships.append(rel)

        return relationships

    def get_neighbors(
        self,
        node_id: int,
        direction: str = 'outgoing',
        rel_type: str = None,
    ) -> list[Node]:
        """Get neighboring nodes via relationships.

        Args:
            node_id: Node ID
            direction: 'outgoing', 'incoming', or 'both'
            rel_type: Optional filter by relationship type

        Returns:
            List of neighbor Node objects

        Raises:
            NodeNotFoundError: If node doesn't exist
            ValueError: If direction is invalid
        """
        # Validate that node exists
        if not self.get_node(node_id):
            raise NodeNotFoundError(node_id)

        # Validate direction
        if direction not in ('outgoing', 'incoming', 'both'):
            raise ValueError(
                f"Invalid direction '{direction}'. Must be 'outgoing', 'incoming', or 'both'"
            )

        neighbor_ids = set()

        # Get outgoing neighbors (nodes this node points to)
        if direction in ('outgoing', 'both'):
            query = "SELECT target_node_id FROM relationships WHERE source_node_id = ?"
            params = [node_id]

            if rel_type:
                query += " AND type = ?"
                params.append(rel_type)

            cursor = self.conn.execute(query, params)
            neighbor_ids.update(row['target_node_id'] for row in cursor.fetchall())

        # Get incoming neighbors (nodes that point to this node)
        if direction in ('incoming', 'both'):
            query = "SELECT source_node_id FROM relationships WHERE target_node_id = ?"
            params = [node_id]

            if rel_type:
                query += " AND type = ?"
                params.append(rel_type)

            cursor = self.conn.execute(query, params)
            neighbor_ids.update(row['source_node_id'] for row in cursor.fetchall())

        # Get full node objects for all neighbors
        neighbors = []
        for neighbor_id in neighbor_ids:
            node = self.get_node(neighbor_id)
            if node:
                neighbors.append(node)

        return neighbors

    # =========================================================================
    # Graph Traversal (to be implemented in Phase 5)
    # =========================================================================

    def find_path(
        self, source_id: int, target_id: int, max_depth: int = None
    ) -> list[Node] | None:
        """Find any path between two nodes (DFS).

        Args:
            source_id: Source node ID
            target_id: Target node ID
            max_depth: Maximum path length (number of relationships)

        Returns:
            List of Node objects in the path, or None if no path exists

        Raises:
            NodeNotFoundError: If source or target node doesn't exist
        """
        # Validate that both nodes exist
        if not self.get_node(source_id):
            raise NodeNotFoundError(source_id)
        if not self.get_node(target_id):
            raise NodeNotFoundError(target_id)

        # Use PathFinder to find a path
        path_finder = PathFinder(self)
        return path_finder.dfs_find_path(source_id, target_id, max_depth)

    def find_shortest_path(
        self, source_id: int, target_id: int
    ) -> list[Node] | None:
        """Find shortest path between two nodes (BFS).

        Args:
            source_id: Source node ID
            target_id: Target node ID

        Returns:
            List of Node objects in the shortest path, or None if no path exists

        Raises:
            NodeNotFoundError: If source or target node doesn't exist
        """
        # Validate that both nodes exist
        if not self.get_node(source_id):
            raise NodeNotFoundError(source_id)
        if not self.get_node(target_id):
            raise NodeNotFoundError(target_id)

        # Use PathFinder to find shortest path
        path_finder = PathFinder(self)
        return path_finder.bfs_shortest_path(source_id, target_id)

    # =========================================================================
    # Metadata Queries (to be implemented in Phase 6)
    # =========================================================================

    def get_all_labels(self) -> list[str]:
        """Get all labels in the database.

        Returns:
            List of label names (sorted alphabetically)
        """
        cursor = self.conn.execute("SELECT name FROM labels ORDER BY name")
        return [row['name'] for row in cursor.fetchall()]

    def get_all_relationship_types(self) -> list[str]:
        """Get all relationship types in the database.

        Returns:
            List of relationship types (sorted alphabetically)
        """
        cursor = self.conn.execute(
            "SELECT DISTINCT type FROM relationships ORDER BY type"
        )
        return [row['type'] for row in cursor.fetchall()]

    def get_node_count(self, label: str = None) -> int:
        """Count nodes in the database.

        Args:
            label: Optional filter by label

        Returns:
            Number of nodes
        """
        if label is None:
            # Count all nodes
            cursor = self.conn.execute("SELECT COUNT(*) as count FROM nodes")
        else:
            # Count nodes with specific label
            cursor = self.conn.execute(
                """
                SELECT COUNT(DISTINCT nl.node_id) as count
                FROM node_labels nl
                JOIN labels l ON nl.label_id = l.id
                WHERE l.name = ?
                """,
                (label,),
            )
        return cursor.fetchone()['count']

    def get_relationship_count(self, rel_type: str = None) -> int:
        """Count relationships in the database.

        Args:
            rel_type: Optional filter by relationship type

        Returns:
            Number of relationships
        """
        if rel_type is None:
            # Count all relationships
            cursor = self.conn.execute(
                "SELECT COUNT(*) as count FROM relationships"
            )
        else:
            # Count relationships of specific type
            cursor = self.conn.execute(
                "SELECT COUNT(*) as count FROM relationships WHERE type = ?",
                (rel_type,),
            )
        return cursor.fetchone()['count']

    def execute(self, cypher_query: str, parameters: dict | None = None) -> list[dict]:
        """Execute a Cypher query and return results.

        This method provides Cypher query language support for Grafito.
        It translates Cypher queries to the programmatic API.

        Supported subset:
        - CREATE (n:Label {props})
        - MATCH (n:Label) WHERE condition RETURN projection
        - Relationship patterns: (a)-[r:TYPE]->(b)
        - WHERE expressions: =, !=, <, >, <=, >=, AND, OR, NOT

        Args:
            cypher_query: Cypher query string
            parameters: Optional mapping of names to values, bound to ``$name``
                references in the query (avoids string interpolation).

        Returns:
            List of result dictionaries

        Raises:
            CypherSyntaxError: If query has invalid syntax
            CypherExecutionError: If query execution fails

        Examples:
            >>> db.execute("CREATE (n:Person {name: 'Alice', age: 30})")
            >>> db.execute("MATCH (n:Person) WHERE n.age > 25 RETURN n.name")
            [{'n.name': 'Alice'}]
            >>> db.execute("MATCH (n:Person {name: $name}) RETURN n.age AS age",
            ...            {"name": "Alice"})
            [{'age': 30}]
        """
        from .cypher.lexer import Lexer
        from .cypher.parser import Parser
        from .cypher.executor import CypherExecutor

        # 1. Tokenize
        lexer = Lexer(cypher_query)
        tokens = lexer.tokenize()

        # 2. Parse
        parser = Parser(tokens)
        ast = parser.parse()

        # 3. Execute
        executor = CypherExecutor(self, parameters=parameters)
        return executor.execute(ast)

    def execute_script(self, cypher_script: str) -> list[list[dict]]:
        """Execute a Cypher script with semicolon-separated statements."""
        results = []
        for statement in self._split_cypher_statements(cypher_script):
            if statement.strip():
                results.append(self.execute(statement))
        return results

    def execute_script_file(self, path: str) -> list[list[dict]]:
        """Execute a .cypher file with semicolon-separated statements."""
        with open(path, "r", encoding="utf-8") as handle:
            script = handle.read()
        return self.execute_script(script)

    def import_neo4j_dump(
        self,
        dump_path: str,
        temp_dir: str | None = None,
        cleanup: bool = True,
        endian: str = ">",
        progress_every: int | None = None,
        node_limit: int | None = None,
        rel_limit: int | None = None,
    ) -> None:
        """Import a Neo4j .dump file into this database."""
        from .importers.neo4j_dump import import_dump

        import_dump(
            self,
            dump_path,
            temp_dir=temp_dir,
            cleanup=cleanup,
            endian=endian,
            progress_every=progress_every,
            node_limit=node_limit,
            rel_limit=rel_limit,
        )

    def import_okf_bundle(
        self,
        path: str,
        *,
        link_type: str = "LINKS_TO",
        typed_links: bool = False,
        citations: bool = True,
        citation_type: str = "CITES",
        configure_fts: bool = True,
        embed: "EmbeddingFunction | None" = None,
        embed_index: str = "okf",
        embed_fields: tuple[str, ...] = ("title", "description", "body"),
        embed_backend: str = "bruteforce",
        embed_options: dict | None = None,
        progress_every: int | None = None,
        progress: "Callable[[str, int], None] | None" = None,
        directory_nodes: bool = False,
        import_log: bool = False,
        uri_prefix: str = "okf:",
        incremental: bool = False,
        prune: bool = False,
        wikilinks: bool = False,
    ) -> dict:
        """Import an Open Knowledge Format (OKF) bundle into this database.

        A bundle is a directory tree of markdown files with YAML frontmatter.
        Each concept becomes a node (label from ``type``, frontmatter as
        properties, body as the ``body`` property), and intra-bundle markdown
        links become relationships. The ``sources`` frontmatter (SPEC sec. 5.1)
        — and a legacy ``# Citations`` body list — become ``CITES``
        relationships to concepts or to auto-created ``Reference`` nodes.

        Pass ``embed=<EmbeddingFunction>`` to also embed each concept into a
        vector index (``embed_index``) for semantic search; query it later with
        ``db.semantic_search("text", index=embed_index)``. ``embed_options``
        (e.g. ``{"store_embeddings": True}``) is forwarded to
        ``create_vector_index`` for durable reuse across sessions. Use
        ``progress_every=N`` (prints) or ``progress=callback`` for progress
        reporting on large bundles.

        Pass ``incremental=True`` to re-import a bundle cheaply: concept files
        whose content hasn't changed since the last import (tracked via a
        content hash) are skipped entirely (no re-parsing, no re-embedding);
        changed files are updated in place (same node ID); new files are
        added. Add ``prune=True`` (requires ``incremental=True``) to also
        delete nodes for concept files removed from the bundle.

        Pass ``wikilinks=True`` to also resolve Obsidian-style ``[[Note]]``
        links in concept bodies (an Obsidian vault is OKF-compatible without a
        plugin already; this adds its native link syntax).

        Returns a summary dict with
        node/relationship/citation/reference/stub/skipped/embedded/unchanged/
        updated/pruned counts.
        """
        from .importers.okf import import_bundle

        return import_bundle(
            self,
            path,
            link_type=link_type,
            typed_links=typed_links,
            citations=citations,
            citation_type=citation_type,
            configure_fts=configure_fts,
            embed=embed,
            embed_index=embed_index,
            embed_fields=embed_fields,
            embed_backend=embed_backend,
            embed_options=embed_options,
            progress_every=progress_every,
            progress=progress,
            directory_nodes=directory_nodes,
            import_log=import_log,
            uri_prefix=uri_prefix,
            incremental=incremental,
            prune=prune,
            wikilinks=wikilinks,
        )

    def export_okf_bundle(
        self,
        path: str,
        *,
        uri_prefix: str = "okf:",
        write_index: bool = True,
        write_viz: bool = False,
        write_log: bool = True,
        prune: bool = False,
        okf_version: str | None = None,
    ) -> dict:
        """Export this database to an Open Knowledge Format (OKF) bundle.

        Writes a directory tree of markdown files with YAML frontmatter (the
        inverse of :meth:`import_okf_bundle`), with per-directory ``index.md``
        files, per-scope ``log.md`` files regenerated from ``LogEntry`` nodes
        (``write_log``), and an optional self-contained ``viz.html``.
        ``prune=True`` deletes concept ``.md`` files that no longer correspond
        to a node. ``okf_version`` declares the format version in the root
        ``index.md`` (SPEC sec. 12). Returns a summary dict with
        concept/skipped/pruned/logs/viz counts.
        """
        from .integrations.okf import export_bundle

        return export_bundle(
            self,
            path,
            uri_prefix=uri_prefix,
            write_index=write_index,
            write_viz=write_viz,
            write_log=write_log,
            prune=prune,
            okf_version=okf_version,
        )

    def _split_cypher_statements(self, script: str) -> list[str]:
        """Split Cypher script into statements, respecting string literals and comments."""
        statements = []
        current = []
        in_single = False
        in_double = False
        escape = False
        in_line_comment = False
        in_block_comment = False
        i = 0
        length = len(script)
        while i < length:
            ch = script[i]
            nxt = script[i + 1] if i + 1 < length else ""
            if escape:
                current.append(ch)
                escape = False
                i += 1
                continue
            if in_line_comment:
                if ch == "\n":
                    in_line_comment = False
                    current.append(ch)
                i += 1
                continue
            if in_block_comment:
                if ch == "*" and nxt == "/":
                    in_block_comment = False
                    i += 2
                else:
                    i += 1
                continue
            if ch == "\\":
                escape = True
                current.append(ch)
                i += 1
                continue
            if ch == "'" and not in_double:
                in_single = not in_single
                current.append(ch)
                i += 1
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                current.append(ch)
                i += 1
                continue
            if not in_single and not in_double:
                if ch == "-" and nxt == "-":
                    in_line_comment = True
                    i += 2
                    continue
                if ch == "/" and nxt == "/":
                    in_line_comment = True
                    i += 2
                    continue
                if ch == "/" and nxt == "*":
                    in_block_comment = True
                    i += 2
                    continue
            if ch == ";" and not in_single and not in_double:
                statements.append("".join(current).strip())
                current = []
                i += 1
                continue
            current.append(ch)
            i += 1

        tail = "".join(current).strip()
        if tail:
            statements.append(tail)
        return statements
