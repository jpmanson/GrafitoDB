# RDF/Turtle Integration

Grafito can export to and import from RDF (Resource Description Framework) format.

## Prerequisites

```bash
pip install grafito[rdf]
```

This installs `rdflib` for RDF handling.

## Exporting to RDF

### Basic Export

```python
from grafito import GrafitoDatabase
from grafito.integrations import export_rdf, export_turtle

# Create graph
db = GrafitoDatabase(':memory:')
alice = db.create_node(labels=['Person'], properties={'name': 'Alice'})
bob = db.create_node(labels=['Person'], properties={'name': 'Bob'})
db.create_relationship(alice.id, bob.id, 'KNOWS', {'since': 2021})

# Export to RDFLib Graph
rdf_graph = export_rdf(db, base_uri='grafito:')

print(f'Triples: {len(rdf_graph)}')
```

### Export to Turtle

```python
# Export to Turtle format
turtle_str = export_turtle(
    db,
    base_uri='grafito:',
    prefixes={
        'schema': 'http://schema.org/',
        'foaf': 'http://xmlns.com/foaf/0.1/'
    }
)

# Save to file
with open('export.ttl', 'w') as f:
    f.write(turtle_str)

print(turtle_str)
```

Output format:
```turtle
@prefix gr: <grafito:> .
@prefix schema: <http://schema.org/> .

gr:node_1 a schema:Person ;
    schema:name "Alice" .

gr:node_2 a schema:Person ;
    schema:name "Bob" .

gr:rel_1 a gr:KNOWS ;
    schema:since 2021 ;
    schema:source gr:node_1 ;
    schema:target gr:node_2 .
```

### Custom Namespace Mapping

```python
# Map labels to schema types
turtle = export_turtle(
    db,
    base_uri='http://example.org/',
    prefixes={'ex': 'http://example.org/'},
    type_map={
        'Person': 'http://schema.org/Person',
        'Company': 'http://schema.org/Organization'
    }
)
```

## RDF to Grafito

### Basic Import

```python
from rdflib import Graph
from grafito import GrafitoDatabase

# Load RDF
rdf_graph = Graph()
rdf_graph.parse('data.ttl', format='turtle')

# Convert to Grafito
db = GrafitoDatabase(':memory:')
# Import functionality would be implemented here
```

### Handling Common Vocabularies

RDF export handles common vocabularies:

| Grafito Concept | RDF Mapping |
|----------------|-------------|
| Node | `rdfs:Resource` |
| Label | `rdf:type` |
| Property | Predicate |
| Relationship | Reified statement |

## Typed RDF Export

```python
from grafito.integrations import export_typed_rdf

# Export with type inference
turtle = export_typed_rdf(
    db,
    base_uri='http://myapp.org/',
    type_inference=True
)
```

## Ontology Export

Export schema as OWL/RDFS:

```python
from grafito.integrations import export_ontology

# Export schema
turtle = export_ontology(
    db,
    base_uri='http://myapp.org/',
    include_data=False  # Schema only
)
```

## Examples

### Exporting Social Network

```python
db = GrafitoDatabase(':memory:')

# Create social network
alice = db.create_node(labels=['Person'], properties={'name': 'Alice', 'age': 30})
bob = db.create_node(labels=['Person'], properties={'name': 'Bob', 'age': 25})
company = db.create_node(labels=['Company'], properties={'name': 'TechCorp'})

db.create_relationship(alice.id, bob.id, 'FRIEND', {'since': 2020})
db.create_relationship(alice.id, company.id, 'WORKS_AT', {'role': 'Engineer'})

# Export with FOAF vocabulary
turtle = export_turtle(
    db,
    base_uri='http://example.org/',
    prefixes={
        'foaf': 'http://xmlns.com/foaf/0.1/',
        'schema': 'http://schema.org/'
    },
    type_map={
        'Person': 'foaf:Person',
        'Company': 'schema:Organization'
    }
)

print(turtle)
```

### Working with Schema.org

```python
# Export using Schema.org vocabulary
schema = {
    'Person': 'http://schema.org/Person',
    'name': 'http://schema.org/name',
    'email': 'http://schema.org/email',
    'KNOWS': 'http://schema.org/knows'
}

turtle = export_turtle(db, base_uri='https://example.com/', type_map=schema)
```

## Limitations

- RDF export reifies relationships (creates separate nodes for edges)
- Property types are mapped to RDF literals
- Multiple labels become multiple `rdf:type` statements
