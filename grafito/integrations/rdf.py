"""RDF/Turtle import/export helpers."""

from __future__ import annotations

import orjson
from typing import Any

from ..database import GrafitoDatabase


def export_rdf(
    db: GrafitoDatabase,
    base_uri: str = "grafito:",
    node_prefix: str = "node/",
    rel_prefix: str = "rel/",
    prefixes: dict[str, str] | None = None,
) -> "Graph":
    """Export the Grafito graph to an rdflib Graph."""
    try:
        from rdflib import Graph, Literal, Namespace, RDF, URIRef
    except ImportError as exc:
        raise ImportError(
            "rdflib is not installed. Install with `pip install grafito[rdf]` "
            "or `uv pip install grafito[rdf]`."
        ) from exc
    graph = Graph()
    ns = Namespace(base_uri)
    rdf_key = "__rdf__"

    default_prefixes = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
        "owl": "http://www.w3.org/2002/07/owl#",
        "schema": "http://schema.org/",
    }
    merged_prefixes = default_prefixes.copy()
    if isinstance(prefixes, dict):
        merged_prefixes.update(prefixes)
    for prefix, uri in merged_prefixes.items():
        if prefix is None:
            continue
        graph.bind(prefix, uri)
    graph.bind("", base_uri)

    def resolve_term(term: str, prefixes: dict[str, str]) -> URIRef:
        if "://" in term or term.startswith("urn:"):
            return URIRef(term)
        if ":" in term:
            prefix, local = term.split(":", 1)
            if prefix in prefixes:
                return URIRef(prefixes[prefix] + local)
        return URIRef(f"{base_uri}{term}")

    def add_rdf_values(
        subject: URIRef,
        predicate: URIRef,
        value: Any,
        prefixes: dict[str, str],
    ) -> None:
        if isinstance(value, list):
            for item in value:
                add_rdf_values(subject, predicate, item, prefixes)
            return
        if isinstance(value, dict):
            if "@id" in value:
                obj = resolve_term(str(value["@id"]), prefixes)
                graph.add((subject, predicate, obj))
                return
            if "@value" in value:
                datatype = value.get("@type")
                lang = value.get("@lang") or value.get("@language")
                datatype_ref = resolve_term(str(datatype), prefixes) if datatype else None
                graph.add((subject, predicate, Literal(value["@value"], datatype=datatype_ref, lang=lang)))
                return
        graph.add((subject, predicate, Literal(value)))

    node_uris: dict[int, str] = {}
    cursor = db.conn.execute("SELECT id, properties, uri FROM nodes ORDER BY id")
    for row in cursor.fetchall():
        node_id = int(row["id"])
        node_uri_value = row["uri"] or f"{base_uri}{node_prefix}{node_id}"
        node_uris[node_id] = node_uri_value
        node_uri = URIRef(node_uri_value)
        labels = db._get_node_labels(node_id)
        for label in labels:
            graph.add((node_uri, RDF.type, ns[label]))
        properties = row["properties"]
        if properties:
            prop_map = orjson.loads(properties)
            for key, value in prop_map.items():
                if key == rdf_key:
                    continue
                graph.add((node_uri, ns[key], Literal(value)))
            rdf_block = prop_map.get(rdf_key)
            if isinstance(rdf_block, dict):
                context = rdf_block.get("@context")
                prefixes = merged_prefixes.copy()
                if isinstance(context, dict):
                    prefixes.update(context)
                for key, value in rdf_block.items():
                    if key.startswith("@"):
                        continue
                    predicate = resolve_term(key, prefixes)
                    add_rdf_values(node_uri, predicate, value, prefixes)

    cursor = db.conn.execute(
        "SELECT id, source_node_id, target_node_id, type, properties, uri FROM relationships ORDER BY id"
    )
    for row in cursor.fetchall():
        rel_id = int(row["id"])
        source_uri = URIRef(
            node_uris.get(int(row["source_node_id"]))
            or f"{base_uri}{node_prefix}{int(row['source_node_id'])}"
        )
        target_uri = URIRef(
            node_uris.get(int(row["target_node_id"]))
            or f"{base_uri}{node_prefix}{int(row['target_node_id'])}"
        )
        pred = ns[row["type"]]
        graph.add((source_uri, pred, target_uri))
        props = row["properties"]
        if props:
            prop_map = orjson.loads(props)
            rel_uri_value = row["uri"] or f"{base_uri}{rel_prefix}{rel_id}"
            rel_uri = URIRef(rel_uri_value)
            graph.add((rel_uri, RDF.type, ns[row["type"]]))
            graph.add((rel_uri, ns["source"], source_uri))
            graph.add((rel_uri, ns["target"], target_uri))
            for key, value in prop_map.items():
                if key == rdf_key:
                    continue
                graph.add((rel_uri, ns[key], Literal(value)))
            rdf_block = prop_map.get(rdf_key)
            if isinstance(rdf_block, dict):
                context = rdf_block.get("@context")
                prefixes = merged_prefixes.copy()
                if isinstance(context, dict):
                    prefixes.update(context)
                for key, value in rdf_block.items():
                    if key.startswith("@"):
                        continue
                    predicate = resolve_term(key, prefixes)
                    add_rdf_values(rel_uri, predicate, value, prefixes)

    return graph


def export_turtle(
    db: GrafitoDatabase,
    base_uri: str = "grafito:",
    prefixes: dict[str, str] | None = None,
) -> str:
    """Export the Grafito graph to Turtle."""
    graph = export_rdf(db, base_uri=base_uri, prefixes=prefixes)
    return graph.serialize(format="turtle")


# rdflib format name keyed by common file extension.
_EXT_FORMATS = {
    ".ttl": "turtle",
    ".turtle": "turtle",
    ".jsonld": "json-ld",
    ".json": "json-ld",
    ".nt": "nt",
    ".ntriples": "nt",
    ".nq": "nquads",
    ".rdf": "xml",
    ".xml": "xml",
    ".owl": "xml",
    ".n3": "n3",
    ".trig": "trig",
}


def _format_from_path(path: str, override: str | None) -> str:
    """Pick an rdflib format from an explicit override or the file extension."""
    if override:
        return override
    import os

    ext = os.path.splitext(path)[1].lower()
    fmt = _EXT_FORMATS.get(ext)
    if fmt is None:
        raise ValueError(
            f"Cannot infer RDF format from extension {ext!r}. "
            f"Pass format= explicitly (one of: turtle, json-ld, nt, nquads, xml, n3, trig)."
        )
    return fmt


def export_string(
    db: GrafitoDatabase,
    base_uri: str = "grafito:",
    format: str = "turtle",
    prefixes: dict[str, str] | None = None,
) -> str:
    """Export the Grafito graph to an RDF string in any rdflib format.

    Args:
        db: Source database.
        base_uri: Namespace for nodes/labels/property predicates.
        format: rdflib serialization format (``turtle``, ``json-ld``, ``nt``,
            ``nquads``, ``xml``, ``n3``, ``trig`` ...).
        prefixes: Extra ``{prefix: namespace}`` bindings.
    """
    graph = export_rdf(db, base_uri=base_uri, prefixes=prefixes)
    return graph.serialize(format=format)


def export_to_file(
    db: GrafitoDatabase,
    path: str,
    base_uri: str = "grafito:",
    format: str | None = None,
    prefixes: dict[str, str] | None = None,
) -> str:
    """Export the graph to a file, inferring the format from the extension.

    ``.ttl`` -> turtle, ``.jsonld``/``.json`` -> JSON-LD, ``.nt`` -> N-Triples,
    ``.nq`` -> N-Quads, ``.rdf``/``.xml`` -> RDF/XML, ``.n3`` -> N3, ``.trig`` -> TriG.
    Pass ``format`` to override.

    Returns:
        The format that was used.
    """
    fmt = _format_from_path(path, format)
    graph = export_rdf(db, base_uri=base_uri, prefixes=prefixes)
    graph.serialize(destination=path, format=fmt, encoding="utf-8")
    return fmt


def import_from_file(
    db: GrafitoDatabase,
    path: str,
    base_uri: str = "grafito:",
    format: str | None = None,
    store_uri: bool = True,
) -> dict[str, int]:
    """Import an RDF file, inferring the format from the extension.

    Thin wrapper over :func:`import_turtle` that resolves the rdflib format from
    the file extension (see :func:`export_to_file` for the mapping).
    """
    fmt = _format_from_path(path, format)
    return import_turtle(db, path, base_uri=base_uri, store_uri=store_uri, format=fmt)


def _term_to_py(term: Any) -> Any:
    """Convert an rdflib query term (Literal/URIRef/BNode) to a Python value."""
    try:
        from rdflib import Literal
    except ImportError:  # pragma: no cover
        return term
    if term is None:
        return None
    if isinstance(term, Literal):
        return _py_value(term)
    return str(term)


def query_sparql(
    db: GrafitoDatabase,
    query: str,
    base_uri: str = "grafito:",
    prefixes: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Run a SPARQL query against the graph, delegating to rdflib.

    Grafito has no native SPARQL engine; this materialises the graph as an
    ``rdflib.Graph`` (via :func:`export_rdf`) and runs the query there, giving
    full SPARQL 1.1 (SELECT/ASK/CONSTRUCT/DESCRIBE) over Grafito data.

    Args:
        db: Source database.
        query: A SPARQL query string.
        base_uri: Namespace used when exporting (predicates live under it).
        prefixes: Extra ``{prefix: namespace}`` bindings available to the query.

    Returns:
        - SELECT: a list of dicts ``{variable: python_value}``.
        - ASK: ``[{"boolean": True/False}]``.
        - CONSTRUCT/DESCRIBE: a list of dicts ``{"s","p","o"}`` for each triple.

    Note:
        The whole graph is loaded into memory for each call, so this is intended
        for interop and small/medium graphs, not high-throughput production use.
    """
    graph = export_rdf(db, base_uri=base_uri, prefixes=prefixes)
    result = graph.query(query)

    if result.type == "ASK":
        return [{"boolean": bool(result.askAnswer)}]
    if result.type in ("CONSTRUCT", "DESCRIBE"):
        return [
            {"s": _term_to_py(s), "p": _term_to_py(p), "o": _term_to_py(o)}
            for s, p, o in result
        ]
    # SELECT
    variables = [str(v) for v in (result.vars or [])]
    rows: list[dict[str, Any]] = []
    for row in result:
        rows.append({var: _term_to_py(getattr(row, var)) for var in variables})
    return rows


def graph_diff(
    first: GrafitoDatabase,
    second: GrafitoDatabase,
    base_uri: str = "grafito:",
) -> dict[str, Any]:
    """Compare two graphs by RDF isomorphism (delegates to ``rdflib.compare``).

    Both databases are exported to canonical (isomorphic) RDF graphs and diffed,
    which correctly ignores node-id ordering and blank-node labelling.

    Returns:
        A dict with ``isomorphic`` (bool) plus triple counts
        ``in_both`` / ``only_in_first`` / ``only_in_second``.
    """
    try:
        from rdflib.compare import graph_diff as _rdf_graph_diff, to_isomorphic
    except ImportError as exc:  # pragma: no cover - guarded like export_rdf
        raise ImportError(
            "rdflib is not installed. Install with `pip install grafito[rdf]` "
            "or `uv pip install grafito[rdf]`."
        ) from exc

    iso_first = to_isomorphic(export_rdf(first, base_uri=base_uri))
    iso_second = to_isomorphic(export_rdf(second, base_uri=base_uri))
    in_both, only_first, only_second = _rdf_graph_diff(iso_first, iso_second)
    return {
        "isomorphic": iso_first == iso_second,
        "in_both": len(in_both),
        "only_in_first": len(only_first),
        "only_in_second": len(only_second),
    }


def _local_name(term: Any, base_uri: str) -> str:
    """Shorten an RDF term to a compact label/property name.

    Strips the base namespace when present, otherwise falls back to the fragment
    (after ``#``) or the last path segment. Full IRIs are returned unchanged when
    they cannot be shortened, so information is never silently lost.
    """
    text = str(term)
    if base_uri and text.startswith(base_uri):
        return text[len(base_uri):]
    if "#" in text:
        return text.rsplit("#", 1)[1]
    if "/" in text:
        tail = text.rstrip("/").rsplit("/", 1)[1]
        if tail:
            return tail
    return text


def _py_value(literal: Any) -> Any:
    """Convert an rdflib Literal to a Grafito-storable Python value."""
    from decimal import Decimal

    value = literal.toPython()
    if isinstance(value, Decimal):
        return float(value)
    # Anything Grafito does not store natively (e.g. rdflib types) becomes a str.
    if isinstance(value, (str, int, float, bool)):
        return value
    from datetime import date, datetime, time

    if isinstance(value, (datetime, date, time)):
        return value
    return str(value)


def import_rdf(
    db: GrafitoDatabase,
    graph: "Graph",
    base_uri: str = "grafito:",
    store_uri: bool = True,
    default_rel_type: str = "RELATED",
) -> dict[str, int]:
    """Import an rdflib ``Graph`` into a GrafitoDatabase as a property graph.

    This is the inverse of :func:`export_rdf` and also handles arbitrary RDF:

    - A subject's ``rdf:type`` values become node **labels**.
    - Triples whose object is a literal become node **properties**.
    - Triples whose object is another resource become **relationships**.
    - Grafito's reified edges (resources carrying ``<base>source`` and
      ``<base>target``) are recognised and imported as relationships that keep
      their literal predicates as **edge properties**; the redundant direct
      triple emitted by :func:`export_rdf` is de-duplicated.

    Args:
        db: Target database (nodes/relationships are created in-place).
        graph: Source ``rdflib.Graph``.
        base_uri: Namespace used to shorten IRIs to labels/property names and to
            detect the reification predicates ``<base>source`` / ``<base>target``.
        store_uri: When True, the original subject IRI is stored on the node's
            ``uri`` field (enabling loss-free re-export).
        default_rel_type: Fallback type for reified edges lacking an ``rdf:type``.

    Returns:
        A summary dict ``{"nodes": int, "relationships": int}``.

    Note:
        RDF is a set of triples, so parallel edges of the same type between the
        same pair of nodes collapse into one, and blank nodes are imported as
        property-less nodes without a stable URI.
    """
    try:
        from rdflib import BNode, Literal, Namespace, RDF, URIRef
    except ImportError as exc:  # pragma: no cover - guarded like export_rdf
        raise ImportError(
            "rdflib is not installed. Install with `pip install grafito[rdf]` "
            "or `uv pip install grafito[rdf]`."
        ) from exc

    ns = Namespace(base_uri)
    source_pred = ns["source"]
    target_pred = ns["target"]
    structural = {RDF.type, source_pred, target_pred}

    # --- Pass 1: detect grafito's reified edge resources -------------------- #
    edge_resources: dict[Any, dict[str, Any]] = {}
    for subject in set(graph.subjects()):
        src = graph.value(subject, source_pred)
        tgt = graph.value(subject, target_pred)
        if src is None or tgt is None:
            continue
        rel_type_term = graph.value(subject, RDF.type)
        rel_type = _local_name(rel_type_term, base_uri) if rel_type_term else default_rel_type
        props: dict[str, Any] = {}
        for pred, obj in graph.predicate_objects(subject):
            if pred in structural:
                continue
            if isinstance(obj, Literal):
                props[_local_name(pred, base_uri)] = _py_value(obj)
        edge_resources[subject] = {
            "source": src,
            "target": tgt,
            "type": rel_type,
            "props": props,
        }

    edge_subjects = set(edge_resources)
    # Direct triples (source, <base>type, target) that reified edges already cover.
    covered: set[tuple] = {
        (info["source"], ns[info["type"]], info["target"])
        for info in edge_resources.values()
    }

    # --- Pass 2: collect node terms ----------------------------------------- #
    def is_node_term(term: Any) -> bool:
        return isinstance(term, (URIRef, BNode)) and term not in edge_subjects

    node_terms: set[Any] = set()
    for s, p, o in graph:
        if s in edge_subjects:
            continue
        if is_node_term(s):
            node_terms.add(s)
        if p != RDF.type and is_node_term(o):
            node_terms.add(o)

    # --- Pass 3: create nodes ----------------------------------------------- #
    node_map: dict[Any, int] = {}
    for term in node_terms:
        labels = [
            _local_name(o, base_uri)
            for o in graph.objects(term, RDF.type)
        ]
        properties: dict[str, Any] = {}
        for pred, obj in graph.predicate_objects(term):
            if pred in structural:
                continue
            if isinstance(obj, Literal):
                properties[_local_name(pred, base_uri)] = _py_value(obj)
        uri = str(term) if (store_uri and isinstance(term, URIRef)) else None
        node = db.create_node(labels=labels, properties=properties, uri=uri)
        node_map[term] = node.id

    # --- Pass 4: create relationships --------------------------------------- #
    rel_count = 0
    # 4a) reified edges (carry properties)
    for info in edge_resources.values():
        s_id = node_map.get(info["source"])
        t_id = node_map.get(info["target"])
        if s_id is None or t_id is None:
            continue
        db.create_relationship(s_id, t_id, info["type"], info["props"])
        rel_count += 1
    # 4b) direct resource-to-resource triples not already covered
    for s, p, o in graph:
        if p == RDF.type or p in (source_pred, target_pred):
            continue
        if s in edge_subjects or not isinstance(o, (URIRef, BNode)):
            continue
        if (s, p, o) in covered:
            continue
        s_id = node_map.get(s)
        t_id = node_map.get(o)
        if s_id is None or t_id is None:
            continue
        db.create_relationship(s_id, t_id, _local_name(p, base_uri), {})
        rel_count += 1

    return {"nodes": len(node_map), "relationships": rel_count}


def import_turtle(
    db: GrafitoDatabase,
    source: str,
    base_uri: str = "grafito:",
    store_uri: bool = True,
    format: str = "turtle",
) -> dict[str, int]:
    """Parse Turtle (from a string or file path) and import it into ``db``.

    Args:
        db: Target database.
        source: Either a Turtle string or a path to a ``.ttl`` file.
        base_uri: Namespace used for shortening (see :func:`import_rdf`).
        store_uri: Store the original IRI on each node's ``uri`` field.
        format: rdflib parse format (``turtle`` by default; also ``xml``,
            ``json-ld``, ``nt``, ``n3``, ``trig`` ...).

    Returns:
        Summary dict from :func:`import_rdf`.
    """
    try:
        from rdflib import Graph
    except ImportError as exc:  # pragma: no cover - guarded like export_rdf
        raise ImportError(
            "rdflib is not installed. Install with `pip install grafito[rdf]` "
            "or `uv pip install grafito[rdf]`."
        ) from exc

    graph = Graph()
    import os

    if "\n" in source or not os.path.exists(source):
        graph.parse(data=source, format=format)
    else:
        graph.parse(source, format=format)
    return import_rdf(db, graph, base_uri=base_uri, store_uri=store_uri)
