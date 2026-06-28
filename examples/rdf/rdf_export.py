from grafito import GrafitoDatabase
from grafito.integrations import export_rdf, export_turtle


def main() -> None:
    db = GrafitoDatabase(":memory:")
    alice = db.create_node(labels=["Person"], properties={"name": "Alice"})
    bob = db.create_node(labels=["Person"], properties={"name": "Bob"})
    db.create_relationship(alice.id, bob.id, "KNOWS", properties={"since": 2021})

    graph = export_rdf(db, base_uri="grafito:")
    print("RDF triples:", len(graph))

    turtle = export_turtle(db, base_uri="grafito:")
    output_path = "grafito_export.ttl"
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(turtle)
    print("Turtle preview:")
    print(turtle)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
