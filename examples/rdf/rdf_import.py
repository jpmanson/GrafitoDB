"""Import RDF/Turtle into GrafitoDB, and demonstrate a loss-free round-trip.

Requires the RDF extra: ``pip install grafito[rdf]``.
"""

from grafito import GrafitoDatabase
from grafito.integrations import export_turtle, import_turtle


def main() -> None:
    base = "http://example.org/"

    # --- 1. Import arbitrary external RDF (FOAF) ----------------------------
    foaf = """
    @prefix foaf: <http://xmlns.com/foaf/0.1/> .
    @prefix ex: <http://example.org/> .
    ex:alice a foaf:Person ; foaf:name "Alice" ; foaf:age 30 ; foaf:knows ex:bob .
    ex:bob a foaf:Person ; foaf:name "Bob" .
    """
    db = GrafitoDatabase(":memory:")
    summary = import_turtle(db, foaf, base_uri=base)
    print("Imported external FOAF:", summary)
    for node in db.match_nodes(labels=["Person"]):
        print("  node", sorted(node.labels), node.properties, node.uri)
    for rel in db.match_relationships():
        print("  rel", rel.type, rel.properties)

    # --- 2. Loss-free round-trip: build -> export -> import -----------------
    src = GrafitoDatabase(":memory:")
    a = src.create_node(labels=["Person"], properties={"name": "Ada", "age": 36})
    b = src.create_node(labels=["Company"], properties={"name": "Analytical Engine"})
    src.create_relationship(a.id, b.id, "WORKS_AT", {"role": "Engineer"})

    turtle = export_turtle(src, base_uri=base)
    restored = GrafitoDatabase(":memory:")
    summary = import_turtle(restored, turtle, base_uri=base)
    print("\nRound-trip restored:", summary)
    works = restored.match_relationships(rel_type="WORKS_AT")
    print("  edge property survived:", works[0].properties)


if __name__ == "__main__":
    main()
