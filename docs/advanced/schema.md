# Advanced: SQLite Schema

GrafitoDB stores the property graph in SQLite using a normalized schema.
See `grafito/schema.py` for the source of truth.

## Core Tables

### nodes

Stores graph nodes.

- `id` (INTEGER PRIMARY KEY)
- `created_at` (REAL, julian day)
- `properties` (TEXT, JSON)
- `uri` (TEXT, optional)

### labels

Normalized label names.

- `id` (INTEGER PRIMARY KEY)
- `name` (TEXT UNIQUE, case-insensitive)

### node_labels

Many-to-many table between nodes and labels.

- `node_id` (INTEGER, FK -> nodes)
- `label_id` (INTEGER, FK -> labels)

### relationships

Directed edges between nodes.

- `id` (INTEGER PRIMARY KEY)
- `source_node_id` (INTEGER, FK -> nodes)
- `target_node_id` (INTEGER, FK -> nodes)
- `type` (TEXT)
- `created_at` (REAL)
- `properties` (TEXT, JSON)
- `uri` (TEXT, optional)

## Index Metadata

### property_indexes

Metadata for property indexes and uniqueness.

- `name` (TEXT PRIMARY KEY)
- `entity` (TEXT)
- `label_or_type` (TEXT)
- `property` (TEXT)
- `unique_flag` (INTEGER)

### property_constraints

Schema constraints registry.

- `name` (TEXT PRIMARY KEY)
- `entity` (TEXT)
- `label_or_type` (TEXT)
- `property` (TEXT)
- `constraint_type` (TEXT)
- `type_name` (TEXT, optional)

## Vector Indexing

### vector_indexes

Metadata for vector indexes.

- `name` (TEXT PRIMARY KEY)
- `dim` (INTEGER)
- `backend` (TEXT)
- `method` (TEXT)
- `options` (TEXT, JSON)

### vector_entries

Optional persisted vectors.

- `index_name` (TEXT)
- `node_id` (INTEGER)
- `vector` (TEXT, JSON)
- `updated_at` (REAL)

## Full-Text Search (FTS5)

### fts_index (virtual table)

- `entity_type` (`node` or `relationship`)
- `entity_id` (INTEGER)
- `label_type` (label or relationship type)
- `content` (TEXT)

### fts_config

Defines which properties are indexed.

- `entity_type` (TEXT)
- `label_type` (TEXT, nullable)
- `property` (TEXT)
- `weight` (REAL, optional)

FTS content is maintained via triggers on `nodes`, `node_labels`, and
`relationships`.
