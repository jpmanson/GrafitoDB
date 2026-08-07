"""SQLite FTS5 text index backend."""

from __future__ import annotations

import sqlite3
from typing import Any

from ..exceptions import DatabaseError
from .base import TextIndex


class SQLiteFTSIndex(TextIndex):
    """Text index backend using SQLite FTS5.
    
    This backend leverages SQLite's built-in full-text search capabilities.
    It requires access to the database connection to operate on the FTS tables.
    
    Note: This backend is tightly coupled to the database and uses shared tables
    (`fts_index`, `fts_config`). It's designed to work within GrafitoDatabase.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        name: str,
        options: dict[str, Any] | None = None,
    ) -> None:
        """Initialize SQLite FTS index.
        
        Args:
            conn: SQLite database connection.
            name: Name of this text index (used for identification).
            options: Optional configuration (e.g., weights).
        """
        super().__init__(options)
        self.conn = conn
        self.name = name

    def add(self, ids: list[int], contents: list[str]) -> None:
        """Add documents to the FTS index.
        
        Args:
            ids: List of entity IDs.
            contents: List of text contents to index.
        """
        if len(ids) != len(contents):
            raise ValueError("ids and contents length mismatch")
        
        rows = [(self.name, entity_id, content) for entity_id, content in zip(ids, contents)]
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO fts_index (label_type, entity_id, content, entity_type)
            VALUES (?, ?, ?, 'custom')
            """,
            rows,
        )

    def remove(self, ids: list[int]) -> None:
        """Remove documents from the FTS index.
        
        Args:
            ids: List of entity IDs to remove.
        """
        placeholders = ",".join("?" * len(ids))
        self.conn.execute(
            f"""
            DELETE FROM fts_index
            WHERE label_type = ? AND entity_id IN ({placeholders})
            """,
            [self.name] + list(ids),
        )

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Search for documents matching the query.
        
        Args:
            query: Search query string.
            k: Number of results to return.
            
        Returns:
            List of (id, score) tuples, ordered by relevance.

        Raises:
            DatabaseError: If the query is not valid FTS5 syntax.
        """
        if not query or not query.strip():
            return []

        try:
            cursor = self.conn.execute(
                """
                SELECT entity_id, bm25(fts_index) AS score
                FROM fts_index
                WHERE label_type = ? AND fts_index MATCH ?
                ORDER BY score ASC
                LIMIT ?
                """,
                (self.name, query, k),
            )
            rows = cursor.fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to search text index: {exc}", exc) from exc

        return [(int(row[0]), float(row[1])) for row in rows]

    def save(self, path: str) -> None:
        """Save is a no-op for SQLite FTS (data is in the database)."""
        pass

    def load(self, path: str) -> None:
        """Load is a no-op for SQLite FTS (data is in the database)."""
        pass

    def rebuild(self) -> None:
        """Rebuild the FTS index by re-running optimize."""
        self.conn.execute("INSERT INTO fts_index(fts_index) VALUES('optimize')")

    def supports_remove(self) -> bool:
        return True
