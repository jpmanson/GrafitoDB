from grafito import GrafitoDatabase
from grafito.integrations import save_pyvis_html


def main() -> None:
    db = GrafitoDatabase(":memory:")
    alice = db.create_node(labels=["Person"], properties={"name": "Alice"})
    bob = db.create_node(labels=["Person"], properties={"name": "Bob"})
    company = db.create_node(labels=["Company"], properties={"name": "Acme"})
    clara = db.create_node(labels=["Person"], properties={"name": "Clara"})
    diego = db.create_node(labels=["Person"], properties={"name": "Diego"})
    ella = db.create_node(labels=["Person"], properties={"name": "Ella"})
    hq = db.create_node(labels=["Office"], properties={"name": "HQ"})
    remote = db.create_node(labels=["Office"], properties={"name": "Remote"})
    project = db.create_node(labels=["Project"], properties={"name": "Atlas"})
    db.create_relationship(alice.id, bob.id, "KNOWS")
    db.create_relationship(alice.id, company.id, "WORKS_AT")
    db.create_relationship(bob.id, company.id, "WORKS_AT")
    db.create_relationship(clara.id, company.id, "WORKS_AT")
    db.create_relationship(diego.id, company.id, "WORKS_AT")
    db.create_relationship(ella.id, company.id, "WORKS_AT")
    db.create_relationship(alice.id, clara.id, "KNOWS")
    db.create_relationship(clara.id, diego.id, "KNOWS")
    db.create_relationship(diego.id, ella.id, "KNOWS")
    db.create_relationship(company.id, hq.id, "LOCATED_IN")
    db.create_relationship(company.id, remote.id, "HAS_OFFICE")
    db.create_relationship(alice.id, project.id, "WORKS_ON")
    db.create_relationship(clara.id, project.id, "WORKS_ON")
    db.create_relationship(diego.id, project.id, "WORKS_ON")

    graph = db.to_networkx()
    output_path = save_pyvis_html(
        graph,
        path="grafito_graph.html",
        node_label="name",
        color_by_label=True,
        physics="compact",
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
