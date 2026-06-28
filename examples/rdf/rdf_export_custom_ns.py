import argparse

from grafito import GrafitoDatabase
from grafito.integrations import export_turtle


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RDF/Turtle with a custom base URI.")
    parser.add_argument("--base-uri", default="https://example.com/grafito/")
    parser.add_argument("--out", default="grafito_custom_ns.ttl")
    args = parser.parse_args()

    db = GrafitoDatabase(":memory:")
    alice = db.create_node(labels=["Person"], properties={"name": "Alice"})
    bob = db.create_node(labels=["Person"], properties={"name": "Bob"})
    db.create_relationship(alice.id, bob.id, "KNOWS", properties={"since": 2021})

    turtle = export_turtle(db, base_uri=args.base_uri)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(turtle)

    print(f"Wrote {args.out} with base URI {args.base_uri}")


if __name__ == "__main__":
    main()
