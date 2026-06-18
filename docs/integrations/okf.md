# Open Knowledge Format (OKF)

[Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf)
(OKF) is an open, human- and agent-friendly format for representing *knowledge*:
a directory tree of markdown files with YAML frontmatter. There is no schema
registry and no required tooling — if you can `git clone` a repo, you can ship
an OKF bundle.

GrafitoDB can **import** an OKF bundle into a queryable property graph and
**export** a graph back to an OKF bundle. This makes OKF a durable, diffable,
human-readable storage layer while GrafitoDB provides Cypher, full-text, and
semantic search on top — a natural fit for **agent memory**: persist knowledge
as markdown in git, index and query it at runtime.

## Prerequisites

```bash
# PyYAML is included by default
pip install grafito
```

## How concepts map to the graph

| OKF | GrafitoDB |
|-----|-----------|
| Concept (a `.md` file) | Node |
| `type` (frontmatter, required) | Node label |
| `title`, `description`, `resource`, `tags`, `timestamp`, extra keys | Node properties |
| Concept ID (e.g. `tables/orders`) | Node `uri` (`okf:tables/orders`) |
| Markdown body | `body` property (feeds full-text search) |
| Markdown link `[x](/tables/y.md)` | Relationship (`LINKS_TO` by default) |
| Link under `# Citations` | `CITES` relationship (to a concept or a `Reference` node) |
| `index.md` / `log.md` | Skipped (reserved, derivable) |

Concepts without a `type` and links to not-yet-written concepts fall back to the
generic `Concept` label (permissive consumption — broken links are tolerated,
not errors).

## Importing a bundle

```python
from grafito import GrafitoDatabase

db = GrafitoDatabase(":memory:")
summary = db.import_okf_bundle("path/to/bundle")
print(summary)  # {'nodes': 8, 'relationships': 9, 'stubs': 3, 'skipped': 6}
```

Once imported, the knowledge is fully queryable:

```python
# Cypher: what does the Orders table link to?
db.execute("""
    MATCH (a {title: 'Orders'})-[:LINKS_TO]->(b)
    RETURN b.title AS target
""")

# Full-text search over titles, descriptions, and bodies
db.text_search("customer", k=5)
```

### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `link_type` | `"LINKS_TO"` | Relationship type created for intra-bundle markdown links. |
| `configure_fts` | `True` | Configure full-text search over `title`/`description`/`body` (best-effort; skipped if SQLite lacks FTS5). |
| `uri_prefix` | `"okf:"` | Prefix prepended to each concept ID to form the node `uri`. |

The returned summary dict reports `nodes`, `relationships`, `stubs` (nodes
created for links whose target is not in the bundle), and `skipped`
(`index.md`/`log.md` files).

## Exporting a bundle

The inverse operation serializes the graph back to OKF markdown:

```python
db.export_okf_bundle("out/bundle", write_viz=True)
# {'concepts': 8, 'skipped': 0, 'viz': True}
```

This writes:

- one markdown file per node (label → `type`, properties → frontmatter, `body`
  → markdown body);
- per-directory `index.md` files for progressive disclosure — the root index
  lists child directories under `# Subdirectories`, and each directory groups
  its concepts by `type`;
- an optional self-contained `viz.html` graph viewer (`write_viz=True`).

Stub nodes (created for broken links during import) are **not** exported. Nodes
created programmatically without a stored `body` get a synthesized `# Links`
section listing their outgoing relationships.

### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `uri_prefix` | `"okf:"` | Prefix used to recover concept IDs from node URIs. Should match the import value. |
| `write_index` | `True` | Generate per-directory `index.md` files. |
| `write_viz` | `False` | Also emit a self-contained `viz.html` at the bundle root. |

## Round-trip and agent memory

Import → query/enrich → export is lossless for the graph structure and
preserves unknown frontmatter keys:

```python
db = GrafitoDatabase(":memory:")
db.import_okf_bundle("bundle")

# ... query, traverse, or add knowledge via Cypher / the programmatic API ...
db.execute("CREATE (n:Playbook {title: 'New runbook', body: 'Steps...'})")

# Persist the enriched knowledge back to markdown (commit it to git)
db.export_okf_bundle("bundle")
```

!!! note "Multi-label nodes"
    OKF concepts have a single `type`. When a node has several labels, the
    exporter uses the first label as `type`. See `todo/okf/IMPROVEMENTS.md` for
    the open design question on representing multi-label nodes.

## Example

A runnable end-to-end example (import → Cypher → full-text search → export with
`viz.html`) lives in `examples/okf_import.py`, using the sample bundle in
`examples/okf_bundle/`:

```bash
python examples/okf_import.py
```
