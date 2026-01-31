# Neo4j Dump Import

Grafito can import data from Neo4j database dumps without requiring a running Neo4j instance.

## Prerequisites

```bash
# zstandard is included by default
pip install grafito
```

## Importing a Dump

### Basic Import

```python
from grafito import GrafitoDatabase

# Create database
db = GrafitoDatabase(':memory:')

# Import Neo4j dump
db.import_neo4j_dump('path/to/neo4j.dump')

# Query imported data
result = db.execute('MATCH (n) RETURN count(n) as count')
print(f'Imported {result[0]["count"]} nodes')
```

### Import to File

```python
# Import to persistent database
db = GrafitoDatabase('imported.db')
db.import_neo4j_dump('neo4j.dump')
db.close()

# Later use
 db = GrafitoDatabase('imported.db')
```

## Supported Neo4j Versions

| Neo4j Version | Dump Format | Support |
|--------------|-------------|---------|
| 4.x | Zstandard compressed | ✅ Full |
| 5.x | Zstandard compressed | ✅ Full |

## Low-Level API

For advanced use cases, import components separately.

### Extract Dump

```python
from grafito.importers import extract_dump

# Extract to directory
extract_dump('neo4j.dump', 'extracted/')
```

### Find Store Directory

```python
from grafito.importers import find_store_dir

# Locate store files
store_path = find_store_dir('extracted/')
print(f'Store at: {store_path}')
```

### Custom Import

```python
from grafito.importers import import_dump, Neo4jStoreParser

# Parse store directly
parser = Neo4jStoreParser('path/to/store')

# Iterate over records
for record in parser.iter_nodes():
    print(f'Node: {record}')

for record in parser.iter_relationships():
    print(f'Relationship: {record}')

# Import with progress callback
def progress(current, total, type):
    pct = (current / total) * 100
    print(f'{type}: {current}/{total} ({pct:.1f}%)')

import_dump(
    'neo4j.dump',
    db,
    progress_callback=progress
)
```

## Migration Workflows

### Full Migration

```python
def migrate_neo4j_to_grafito(neo4j_dump, grafito_db_path):
    """Complete Neo4j to Grafito migration."""
    from grafito import GrafitoDatabase

    # Create fresh Grafito database
    db = GrafitoDatabase(grafito_db_path)

    try:
        # Import
        print('Importing...')
        db.import_neo4j_dump(neo4j_dump)

        # Create indexes for common queries
        print('Creating indexes...')
        db.create_node_index('Person', 'name')
        db.create_node_index('User', 'email')

        # Verify
        result = db.execute('MATCH (n) RETURN labels(n) as label, count(*) as count')
        for row in result:
            print(f"{row['label']}: {row['count']}")

        return db
    except Exception as e:
        print(f'Error: {e}')
        raise

# Usage
db = migrate_neo4j_to_grafito('neo4j.dump', 'migrated.db')
```

### Selective Import

```python
def import_only_users(neo4j_dump, db):
    """Import only User nodes."""
    from grafito.importers import Neo4jStoreParser

    parser = Neo4jStoreParser(neo4j_dump)

    with db:
        for record in parser.iter_nodes():
            if 'User' in record.get('labels', []):
                # Import only User nodes
                db.create_node(
                    labels=record['labels'],
                    properties=record['properties']
                )
```

## What Gets Imported

| Neo4j Feature | Grafito Mapping |
|--------------|-----------------|
| Nodes | Nodes with labels and properties |
| Relationships | Relationships with type and properties |
| Properties | JSON-compatible types (string, int, float, bool, list) |
| Constraints | Manual recreation via Cypher |
| Indexes | Manual recreation after import |

## Limitations

- Neo4j constraints must be manually recreated
- Full-text indexes require separate configuration
- Spatial/temporal types converted to strings
- Neo4j's `id()` function differs from Grafito's IDs
