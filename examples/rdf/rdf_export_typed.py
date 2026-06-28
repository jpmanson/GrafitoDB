from grafito import GrafitoDatabase
from grafito.integrations import export_turtle


def main() -> None:
    db = GrafitoDatabase(":memory:")

    alice = db.create_node(
        labels=["Person"],
        properties={
            "name": "Alice",
            "__rdf__": {
                "@context": {"schema": "http://schema.org/"},
                "schema:name": {"@value": "Alicia", "@lang": "es"},
                "schema:birthDate": {"@value": "1990-01-01", "@type": "xsd:date"},
                "schema:sameAs": {"@id": "http://example.com/alice"},
            },
        },
        uri="http://example.com/person/alice",
    )

    bob = db.create_node(
        labels=["Person"],
        properties={"name": "Bob"},
        uri="http://example.com/person/bob",
    )

    db.create_relationship(
        alice.id,
        bob.id,
        "KNOWS",
        properties={
            "since": 2021,
            "__rdf__": {"schema:since": {"@value": "2021", "@type": "xsd:gYear"}},
        },
        uri="http://example.com/rel/knows-1",
    )

    turtle = export_turtle(
        db,
        base_uri="grafito:",
        prefixes={"schema": "http://schema.org/"},
    )
    print(turtle)


if __name__ == "__main__":
    main()
