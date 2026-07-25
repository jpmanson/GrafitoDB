"""GrafitoDB: SQLite-based Property Graph Database.

GrafitoDB implements the Property Graph Model (used by Neo4j and Cypher) using SQLite
as the storage backend. It provides a Pythonic API for creating and querying graphs
with nodes, relationships, labels, and properties.

Example:
    >>> from grafito import GrafitoDatabase
    >>> db = GrafitoDatabase()
    >>> person = db.create_node(labels=['Person'], properties={'name': 'Alice', 'age': 30})
    >>> company = db.create_node(labels=['Company'], properties={'name': 'TechCorp'})
    >>> rel = db.create_relationship(person.id, company.id, 'WORKS_AT', {'since': 2020})
"""

from .database import GrafitoDatabase
from .exceptions import (
    DatabaseError,
    GrafitoError,
    InvalidLabelError,
    InvalidPropertyError,
    InvalidFilterError,
    NodeNotFoundError,
    RelationshipNotFoundError,
)
from .filters import (
    PropertyFilter,
    PropertyFilterGroup,
    LabelFilter,
    SortOrder,
)
from .models import Node, Relationship, Point
from .tools import CypherTools, GraphTools, ToolRegistry, ToolSet

def _resolve_version() -> str:
    """Resolve the package version without hardcoding it (avoids drift).

    Uses installed distribution metadata when available, and falls back to
    reading pyproject.toml when running from an uninstalled source tree.
    """
    from importlib.metadata import version as _pkg_version, PackageNotFoundError

    try:
        return _pkg_version('grafitodb')
    except PackageNotFoundError:
        import tomllib
        from pathlib import Path

        pyproject = Path(__file__).resolve().parent.parent / 'pyproject.toml'
        try:
            return tomllib.loads(pyproject.read_text(encoding='utf-8'))['project']['version']
        except (OSError, KeyError, tomllib.TOMLDecodeError):
            return '0.0.0+unknown'


__version__ = _resolve_version()
__all__ = [
    'GrafitoDatabase',
    'Node',
    'Relationship',
    'Point',
    'GrafitoError',
    'NodeNotFoundError',
    'RelationshipNotFoundError',
    'InvalidLabelError',
    'InvalidPropertyError',
    'InvalidFilterError',
    'DatabaseError',
    'PropertyFilter',
    'PropertyFilterGroup',
    'LabelFilter',
    'SortOrder',
    'GraphTools',
    'CypherTools',
    'ToolSet',
    'ToolRegistry',
]
