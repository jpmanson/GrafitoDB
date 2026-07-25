# RDF/Turtle Integration

GrafitoDB can **export to** and **import from** RDF (Resource Description Framework),
using [`rdflib`](https://rdflib.readthedocs.io) under the hood. This makes a Grafito
property graph interoperable with the Semantic Web / Linked Data ecosystem.

## Prerequisites

```bash
pip install grafito[rdf]
```

This installs `rdflib` for RDF handling.

## Public API

The integration lives in `grafito.integrations`:

| Function | Purpose |
| --- | --- |
| `export_rdf(db, ...)` | Export the graph to an `rdflib.Graph` |
| `export_turtle(db, ...)` | Export the graph to a Turtle string |
| `export_string(db, format=..., ...)` | Export to any rdflib format as a string |
| `export_to_file(db, path, ...)` | Export to a file, format inferred from extension |
| `import_rdf(db, graph, ...)` | Import an `rdflib.Graph` into the database |
| `import_turtle(db, source, ...)` | Parse Turtle (string or file) and import it |
| `import_from_file(db, path, ...)` | Import a file, format inferred from extension |
| `query_sparql(db, query, ...)` | Run SPARQL over the graph (delegated to rdflib) |
| `graph_diff(a, b, ...)` | Compare two graphs by RDF isomorphism |

---

## Exporting to RDF

### Basic Export

```python
from grafito import GrafitoDatabase
from grafito.integrations import export_rdf, export_turtle

db = GrafitoDatabase(':memory:')
alice = db.create_node(labels=['Person'], properties={'name': 'Alice'})
bob = db.create_node(labels=['Person'], properties={'name': 'Bob'})
db.create_relationship(alice.id, bob.id, 'KNOWS', {'since': 2021})

# Export to an rdflib Graph
rdf_graph = export_rdf(db, base_uri='http://example.org/')
print(f'Triples: {len(rdf_graph)}')
```

`export_rdf` signature:

```python
export_rdf(
    db,
    base_uri="grafito:",     # namespace for nodes/labels/property predicates
    node_prefix="node/",     # URI segment for nodes without an explicit uri
    rel_prefix="rel/",       # URI segment for reified relationships
    prefixes=None,           # dict {prefix: namespace} merged with the defaults
) -> rdflib.Graph
```

The following prefixes are always bound: `rdf`, `rdfs`, `xsd`, `owl`, `schema`.

### Export to Turtle

```python
turtle_str = export_turtle(
    db,
    base_uri='http://example.org/',
    prefixes={
        'schema': 'http://schema.org/',
        'foaf': 'http://xmlns.com/foaf/0.1/',
    },
)

with open('export.ttl', 'w') as f:
    f.write(turtle_str)
```

Output:

```turtle
@prefix : <http://example.org/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

<http://example.org/rel/1> a :KNOWS ;
    :since 2021 ;
    :source <http://example.org/node/1> ;
    :target <http://example.org/node/2> .

<http://example.org/node/1> a :Person ;
    :KNOWS <http://example.org/node/2> ;
    :name "Alice" .

<http://example.org/node/2> a :Person ;
    :name "Bob" .
```

### How the mapping works

| GrafitoDB concept | RDF mapping |
| --- | --- |
| Node | Subject IRI (from the node's `uri`, or `base_uri + node_prefix + id`) |
| Label | `rdf:type` |
| Property (literal) | Predicate `base_uri:<key>` → literal object |
| Relationship | Direct triple `(source) base_uri:<TYPE> (target)` |
| Relationship **with properties** | Additionally **reified**: a `rel/<id>` resource typed as the relationship, carrying `:source`, `:target` and one predicate per property |

Relationships are reified so that edge properties (which plain RDF triples cannot
carry) are preserved. This is the inverse of what `import_rdf` recognises, so a
round-trip is loss-free (see [Round-tripping](#round-tripping)).

### Custom predicates and typed literals (`__rdf__`)

For fine-grained control over predicates, datatypes and language tags, add an
`__rdf__` block to a node's (or relationship's) properties. It supports a JSON-LD
style `@context`, plus `@id` (IRI object), `@value`/`@type`/`@lang` (typed/tagged
literals), and lists:

```python
db.create_node(
    labels=['Person'],
    properties={
        'name': 'Alice',
        '__rdf__': {
            '@context': {'foaf': 'http://xmlns.com/foaf/0.1/'},
            'foaf:homepage': {'@id': 'https://alice.example'},
            'foaf:age': {'@value': 30, '@type': 'xsd:integer'},
            'foaf:name': {'@value': 'Alice', '@lang': 'en'},
        },
    },
)
```

The `__rdf__` key itself is never emitted as a literal; only its expanded triples are.

---

## Importing RDF into GrafitoDB

### From a Turtle string or file

```python
from grafito import GrafitoDatabase
from grafito.integrations import import_turtle

db = GrafitoDatabase(':memory:')
summary = import_turtle(db, 'data.ttl', base_uri='http://example.org/')
print(summary)  # {'nodes': 42, 'relationships': 87}
```

`import_turtle` accepts either a **file path** or a **Turtle string** as `source`,
and forwards `format` to `rdflib` (default `"turtle"`; also `"xml"`, `"json-ld"`,
`"nt"`, `"n3"`, `"trig"`, …).

### From an existing `rdflib.Graph`

```python
from rdflib import Graph
from grafito.integrations import import_rdf

g = Graph()
g.parse('data.jsonld', format='json-ld')

db = GrafitoDatabase(':memory:')
import_rdf(db, g, base_uri='http://example.org/')
```

### Import mapping

`import_rdf` projects RDF onto the property-graph model:

- A subject's `rdf:type` values become node **labels**.
- Triples whose object is a **literal** become node **properties**.
- Triples whose object is another **resource** become **relationships**.
- Grafito's **reified edges** (resources carrying `<base>source` and `<base>target`)
  are recognised and imported as relationships that keep their literal predicates as
  **edge properties**; the redundant direct triple is de-duplicated.
- When `store_uri=True` (default), the original subject IRI is stored on the node's
  `uri` field, enabling loss-free re-export.

IRIs are shortened to compact label/property names by stripping `base_uri`, or
falling back to the fragment (`#…`) or last path segment.

It works on arbitrary RDF, not just Grafito's own export:

```python
foaf = """
@prefix foaf: <http://xmlns.com/foaf/0.1/> .
@prefix ex: <http://example.org/> .
ex:alice a foaf:Person ; foaf:name "Alice" ; foaf:knows ex:bob .
ex:bob a foaf:Person ; foaf:name "Bob" .
"""
db = GrafitoDatabase(':memory:')
import_turtle(db, foaf, base_uri='http://example.org/')
# -> node Person{name: Alice} -[knows]-> node Person{name: Bob}
```

### Round-tripping

`export` and `import` are inverses, so a graph survives a full cycle:

```python
turtle = export_turtle(db, base_uri='http://example.org/')

restored = GrafitoDatabase(':memory:')
import_turtle(restored, turtle, base_uri='http://example.org/')
# restored has the same nodes, labels, properties, relationships and edge properties
```

---

## Other RDF formats (JSON-LD, N-Triples, RDF/XML…)

`export_string` and `export_to_file` reach any format `rdflib` supports. On files the
format is inferred from the extension (override with `format=`):

| Extension | Format | | Extension | Format |
| --- | --- | --- | --- | --- |
| `.ttl` | turtle | | `.rdf` / `.xml` | xml |
| `.jsonld` / `.json` | json-ld | | `.n3` | n3 |
| `.nt` | nt (N-Triples) | | `.trig` | trig |
| `.nq` | nquads | | | |

```python
from grafito.integrations import export_string, export_to_file, import_from_file

# As a string, in any format
jsonld = export_string(db, base_uri='http://example.org/', format='json-ld')

# To a file (format inferred from the extension)
export_to_file(db, 'graph.jsonld', base_uri='http://example.org/')

# Import back, format inferred
restored = GrafitoDatabase(':memory:')
import_from_file(restored, 'graph.jsonld', base_uri='http://example.org/')
```

## SPARQL queries

Grafito has no native SPARQL engine, but `query_sparql` materialises the graph as an
`rdflib.Graph` and runs the query there, giving full **SPARQL 1.1**
(SELECT / ASK / CONSTRUCT / DESCRIBE) over Grafito data:

```python
from grafito.integrations import query_sparql

rows = query_sparql(db, """
    PREFIX ex: <http://example.org/>
    SELECT ?name ?age
    WHERE { ?p a ex:Person ; ex:name ?name ; ex:age ?age }
    ORDER BY DESC(?age)
""", base_uri='http://example.org/')
# -> [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]

query_sparql(db, 'PREFIX ex: <http://example.org/> ASK { ?x ex:name "Alice" }',
             base_uri='http://example.org/')
# -> [{'boolean': True}]
```

Return shapes: `SELECT` → list of `{variable: value}` dicts; `ASK` → `[{'boolean': ...}]`;
`CONSTRUCT`/`DESCRIBE` → list of `{'s', 'p', 'o'}` triples.

!!! note
    The whole graph is loaded into memory for each call, so this is meant for interop
    and small/medium graphs. For large graphs and traversals, use Cypher
    (`db.execute(...)`) or the programmatic API, which query SQLite directly and use
    indexes.

## Comparing two graphs

`graph_diff` compares two databases by **RDF isomorphism** (via `rdflib.compare`),
correctly ignoring node-id ordering and blank-node labelling:

```python
from grafito.integrations import graph_diff

graph_diff(db_a, db_b, base_uri='http://example.org/')
# -> {'isomorphic': True, 'in_both': 11, 'only_in_first': 0, 'only_in_second': 0}
```

---

## Limitations

- **Relationships are reified** to carry edge properties (an extra resource per edge
  with properties). This is the standard RDF n-ary/reification pattern.
- **Parallel edges collapse**: RDF is a *set* of triples, so two relationships of the
  same type between the same pair of nodes cannot be distinguished after export.
- **Blank nodes** are imported as property-less nodes without a stable `uri`.
- Datatypes are inferred by `rdflib` on import (`toPython()`); `xsd:decimal` becomes a
  Python `float`, and RDF types Grafito cannot store natively are coerced to strings.
