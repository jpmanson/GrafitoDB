from grafito import GrafitoDatabase
from grafito.integrations import export_turtle


ONTOLOGY_TTL = """@prefix : <http://kg.example.org/resource#> .
@prefix ont: <http://kg.example.org/ontology#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

# ===========================================
# The Knowledge Layer — A simple OWL Ontology
# ===========================================

# --- Class Hierarchy ---

ont:Event rdf:type owl:Class ;
    rdfs:label "Event" .

ont:Organization rdf:type owl:Class ;
    rdfs:label "Organization" .

ont:Place rdf:type owl:Class ;
    rdfs:label "Place" .

ont:City rdf:type owl:Class ;
    rdfs:subClassOf ont:Place ;
    rdfs:label "City" .

ont:Venue rdf:type owl:Class ;
    rdfs:subClassOf ont:Place ;
    rdfs:label "Venue" .

# --- Object Properties (relationships) ---

:hostedBy rdf:type owl:ObjectProperty ;
    rdfs:label "hosted by" ;
    rdfs:domain ont:Event ;
    rdfs:range ont:Organization .

:locatedIn rdf:type owl:ObjectProperty ;
    rdfs:label "located in" ;
    rdfs:domain ont:Place ;
    rdfs:range ont:Place .

# --- Datatype Properties (literal values) ---

:title rdf:type owl:DatatypeProperty ;
    rdfs:label "title" ;
    rdfs:domain ont:Event ;
    rdfs:range xsd:string .
"""


def main() -> None:
    db = GrafitoDatabase(":memory:")

    org = db.create_node(
        properties={
            "name": "City Arts Council",
            "__rdf__": {
                "@context": {"ont": "http://kg.example.org/ontology#"},
                "rdf:type": {"@id": "ont:Organization"},
            },
        },
        uri="http://kg.example.org/resource#org/city-arts-council",
    )

    city = db.create_node(
        properties={
            "name": "Valemont",
            "__rdf__": {
                "@context": {"ont": "http://kg.example.org/ontology#"},
                "rdf:type": {"@id": "ont:City"},
            },
        },
        uri="http://kg.example.org/resource#city/valemont",
    )

    venue = db.create_node(
        properties={
            "name": "Riverside Hall",
            "__rdf__": {
                "@context": {"ont": "http://kg.example.org/ontology#"},
                "rdf:type": {"@id": "ont:Venue"},
            },
        },
        uri="http://kg.example.org/resource#venue/riverside-hall",
    )

    event = db.create_node(
        properties={
            "name": "Night Market",
            "__rdf__": {
                "@context": {
                    "ont": "http://kg.example.org/ontology#",
                    "schema": "http://schema.org/",
                },
                "rdf:type": {"@id": "ont:Event"},
                "schema:startDate": {"@value": "2025-10-12", "@type": "xsd:date"},
                "schema:endDate": {"@value": "2025-10-12", "@type": "xsd:date"},
            },
        },
        uri="http://kg.example.org/resource#event/night-market",
    )

    db.create_relationship(
        event.id,
        org.id,
        "hostedBy",
        properties={"role": "organizer"},
        uri="http://kg.example.org/resource#rel/hosted-by-1",
    )
    db.create_relationship(
        event.id,
        venue.id,
        "locatedIn",
        uri="http://kg.example.org/resource#rel/located-in-1",
    )
    db.create_relationship(
        venue.id,
        city.id,
        "locatedIn",
        uri="http://kg.example.org/resource#rel/located-in-2",
    )

    turtle = export_turtle(
        db,
        base_uri="http://kg.example.org/resource#",
        prefixes={
            "ont": "http://kg.example.org/ontology#",
            "schema": "http://schema.org/",
        },
    )

    out_path = "kg_example.ttl"
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(ONTOLOGY_TTL)
        handle.write("\n")
        handle.write(turtle)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
