from grafito import GrafitoDatabase
from grafito.integrations import export_graph


def main() -> None:
    db = GrafitoDatabase(":memory:")
    alice = db.create_node(labels=["Person"], properties={"name": "Alice"})
    bob = db.create_node(labels=["Person"], properties={"name": "Bob"})
    clara = db.create_node(labels=["Person"], properties={"name": "Clara"})
    company = db.create_node(labels=["Company"], properties={"name": "Acme"})
    project = db.create_node(labels=["Project"], properties={"name": "Atlas"})
    db.create_relationship(alice.id, bob.id, "KNOWS")
    db.create_relationship(bob.id, clara.id, "KNOWS")
    db.create_relationship(alice.id, company.id, "WORKS_AT")
    db.create_relationship(bob.id, company.id, "WORKS_AT")
    db.create_relationship(clara.id, project.id, "WORKS_ON")
    db.create_relationship(company.id, project.id, "SPONSORS")

    graph = db.to_networkx()
    output_path = export_graph(
        graph,
        "grafito_d3.html",
        backend="d3",
        node_label="label_and_name",
        color_by_label=True,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
