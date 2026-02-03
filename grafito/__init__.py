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

__version__ = '0.1.0'
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
]
