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
    export_graph(graph, "grafito_graph.dot", backend="graphviz", node_label="label_and_name")

    # Requires Graphviz: brew install graphviz
    export_graph(
        graph,
        "grafito_graph.dot",
        backend="graphviz",
        render="svg",
        engine="neato",
        node_label="label_and_name",
    )
    export_graph(
        graph,
        "grafito_graph.dot",
        backend="graphviz",
        render="svg",
        engine="sfdp",
        node_label="label_and_name",
    )
    print("Wrote grafito_graph.dot and SVG renders.")


if __name__ == "__main__":
    main()
