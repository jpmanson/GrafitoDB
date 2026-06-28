"""Export a Grafito graph to NetworkX and import it back."""

from grafito import GrafitoDatabase


def main() -> None:
    db = GrafitoDatabase(":memory:")
    alice = db.create_node(labels=["Person"], properties={"name": "Alice"})
    bob = db.create_node(labels=["Person"], properties={"name": "Bob"})
    db.create_relationship(alice.id, bob.id, "KNOWS", {"since": 2020})

    graph = db.to_networkx()
    print(f"NetworkX nodes: {graph.number_of_nodes()}")
    print(f"NetworkX edges: {graph.number_of_edges()}")

    db2 = GrafitoDatabase(":memory:")
    node_map = db2.from_networkx(graph)
    print(f"Imported nodes: {len(node_map)}")


if __name__ == "__main__":
    main()
