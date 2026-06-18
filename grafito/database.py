"""Core database class for Grafito graph database."""

import orjson
import os
import re
import sqlite3
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

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

    def upsert_embedding(self, node_id: int, vector: list[float], index: str = "default") -> None:
        """Insert or update an embedding for a node."""
        if not self.get_node(node_id):
            raise NodeNotFoundError(node_id)
        vec_index = self._get_vector_index(index)
        # Remove existing embedding first to avoid duplicates (FAISS add_with_ids doesn't update)
        if vec_index.supports_remove():
            try:
                vec_index.remove([node_id])
            except Exception:
                pass  # ID may not exist yet
        vec_index.add([node_id], [vector])
        if vec_index.options.get("store_embeddings"):
            try:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO vector_entries (index_name, node_id, vector)
                    VALUES (?, ?, ?)
                    """,
                    (index, node_id, orjson.dumps(vector).decode('utf-8')),
                )
                self.conn.commit()
            except Exception as exc:
                raise DatabaseError(f"Failed to persist embedding: {exc}", exc)
        self._persist_vector_index(index, vec_index)

    def upsert_embeddings(
        self,
        node_ids: list[int],
        documents: list[str],
        index: str = "default",
    ) -> None:
        """Insert or update embeddings from documents."""
        if len(node_ids) != len(documents):
            raise DatabaseError("node_ids and documents length mismatch")
        embedder = self._get_embedding_function(index)
        if embedder is None:
            raise DatabaseError(f"Vector index '{index}' has no embedding function")
        vectors = embedder(list(documents))
        for node_id, vector in zip(node_ids, vectors):
            self.upsert_embedding(node_id, vector, index=index)

    def remove_embedding(self, node_id: int, index: str = "default") -> None:
        """Remove an embedding for a node."""
        vec_index = self._get_vector_index(index)
        vec_index.remove([node_id])
        if vec_index.options.get("store_embeddings"):
            self.conn.execute(
                "DELETE FROM vector_entries WHERE index_name = ? AND node_id = ?",
                (index, node_id),
            )
            self.conn.commit()
        self._persist_vector_index(index, vec_index)

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
        """Ensure vector index path is set and directory exists."""
        if options.get("index_path"):
            return options
        base_dir = os.path.join(os.getcwd(), ".grafito", "indexes")
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

        Args:
            node_id: Node ID

        Returns:
            True if node was deleted, False if node didn't exist
        """
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
        params: list[Any] = []

        # Build label conditions
        label_query = ""
        has_labels = False

        if isinstance(labels, LabelFilter):
            # LabelFilter with OR/AND logic
            has_labels = True
            if labels.operator == 'OR':
                # OR: Node must have ANY of the labels (no HAVING COUNT)
                placeholders = ','.join('?' * len(labels.labels))
                label_query = f"""
                    FROM nodes n
                    JOIN node_labels nl ON n.id = nl.node_id
                    JOIN labels l ON nl.label_id = l.id
                    WHERE l.name IN ({placeholders})
                """
                params.extend(labels.labels)
            else:
                # AND: Node must have ALL labels (with HAVING COUNT)
                placeholders = ','.join('?' * len(labels.labels))
                label_query = f"""
                    FROM nodes n
                    JOIN node_labels nl ON n.id = nl.node_id
                    JOIN labels l ON nl.label_id = l.id
                    WHERE l.name IN ({placeholders})
                    GROUP BY n.id
                    HAVING COUNT(DISTINCT l.name) = ?
                """
                params.extend(labels.labels)
                params.append(len(labels.labels))

        elif labels:
            # Plain list: AND logic (backward compatible)
            has_labels = True
            placeholders = ','.join('?' * len(labels))
            label_query = f"""
                FROM nodes n
                JOIN node_labels nl ON n.id = nl.node_id
                JOIN labels l ON nl.label_id = l.id
                WHERE l.name IN ({placeholders})
                GROUP BY n.id
                HAVING COUNT(DISTINCT l.name) = ?
            """
            params.extend(labels)
            params.append(len(labels))
        else:
            # No label filter
            label_query = "FROM nodes n"

        # Build property conditions
        prop_conditions, prop_params = self._build_property_conditions(properties, 'n')

        # Construct WHERE clause for properties
        where_clause = ""
        if prop_conditions:
            if has_labels and 'WHERE' in label_query:
                # Labels already have WHERE, use AND
                where_clause = " AND " + " AND ".join(prop_conditions)
            else:
                # No WHERE yet or no labels, add WHERE
                where_clause = " WHERE " + " AND ".join(prop_conditions)

        params.extend(prop_params)

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
            {label_query}
            {where_clause}
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

    def execute(self, cypher_query: str) -> list[dict]:
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

        Returns:
            List of result dictionaries

        Raises:
            CypherSyntaxError: If query has invalid syntax
            CypherExecutionError: If query execution fails

        Examples:
            >>> db.execute("CREATE (n:Person {name: 'Alice', age: 30})")
            >>> db.execute("MATCH (n:Person) WHERE n.age > 25 RETURN n.name")
            [{'n.name': 'Alice'}]
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
        executor = CypherExecutor(self)
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
        citations: bool = True,
        citation_type: str = "CITES",
        configure_fts: bool = True,
        embed: "EmbeddingFunction | None" = None,
        embed_index: str = "okf",
        embed_fields: tuple[str, ...] = ("title", "description", "body"),
        embed_backend: str = "bruteforce",
        uri_prefix: str = "okf:",
    ) -> dict:
        """Import an Open Knowledge Format (OKF) bundle into this database.

        A bundle is a directory tree of markdown files with YAML frontmatter.
        Each concept becomes a node (label from ``type``, frontmatter as
        properties, body as the ``body`` property), and intra-bundle markdown
        links become relationships. Links under a ``# Citations`` heading become
        ``CITES`` relationships to concepts or to auto-created ``Reference``
        nodes (external URLs).

        Pass ``embed=<EmbeddingFunction>`` to also embed each concept into a
        vector index (``embed_index``) for semantic search; query it later with
        ``db.semantic_search("text", index=embed_index)``.

        Returns a summary dict with
        node/relationship/citation/reference/stub/skipped/embedded counts.
        """
        from .importers.okf import import_bundle

        return import_bundle(
            self,
            path,
            link_type=link_type,
            citations=citations,
            citation_type=citation_type,
            configure_fts=configure_fts,
            embed=embed,
            embed_index=embed_index,
            embed_fields=embed_fields,
            embed_backend=embed_backend,
            uri_prefix=uri_prefix,
        )

    def export_okf_bundle(
        self,
        path: str,
        *,
        uri_prefix: str = "okf:",
        write_index: bool = True,
        write_viz: bool = False,
    ) -> dict:
        """Export this database to an Open Knowledge Format (OKF) bundle.

        Writes a directory tree of markdown files with YAML frontmatter (the
        inverse of :meth:`import_okf_bundle`), with per-directory ``index.md``
        files and an optional self-contained ``viz.html``. Returns a summary
        dict with concept/skipped/viz counts.
        """
        from .integrations.okf import export_bundle

        return export_bundle(
            self,
            path,
            uri_prefix=uri_prefix,
            write_index=write_index,
            write_viz=write_viz,
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
