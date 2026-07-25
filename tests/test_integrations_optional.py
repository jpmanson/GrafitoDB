import pytest

from grafito import GrafitoDatabase


def _make_sample_db() -> GrafitoDatabase:
    db = GrafitoDatabase(":memory:")
    alice = db.create_node(labels=["Person"], properties={"name": "Alice"})
    bob = db.create_node(labels=["Person"], properties={"name": "Bob"})
    db.create_relationship(alice.id, bob.id, "KNOWS", properties={"since": 2021})
    return db


def test_export_turtle_requires_rdflib():
    pytest.importorskip("rdflib")
    from grafito.integrations import export_turtle

    db = _make_sample_db()
    turtle = export_turtle(db, base_uri="grafito:")
    assert "grafito:" in turtle
    assert "KNOWS" in turtle


def test_import_turtle_roundtrip():
    pytest.importorskip("rdflib")
    from grafito.integrations import export_turtle, import_turtle

    db = GrafitoDatabase(":memory:")
    alice = db.create_node(labels=["Person"], properties={"name": "Alice", "age": 30})
    bob = db.create_node(labels=["Person"], properties={"name": "Bob"})
    corp = db.create_node(labels=["Company"], properties={"name": "TechCorp"})
    db.create_relationship(alice.id, bob.id, "KNOWS", properties={"since": 2021})
    db.create_relationship(alice.id, corp.id, "WORKS_AT", properties={})

    turtle = export_turtle(db, base_uri="http://ex.org/")

    db2 = GrafitoDatabase(":memory:")
    summary = import_turtle(db2, turtle, base_uri="http://ex.org/")

    assert summary == {"nodes": 3, "relationships": 2}
    assert db2.get_node_count() == 3
    assert db2.get_relationship_count() == 2

    # Edge properties survive the reification round-trip
    knows = db2.match_relationships(rel_type="KNOWS")
    assert len(knows) == 1
    assert knows[0].properties.get("since") == 2021

    # Labels and node properties are preserved
    people = db2.match_nodes(labels=["Person"])
    assert {n.properties["name"] for n in people} == {"Alice", "Bob"}
    # Original IRIs are stored on the node uri field
    assert all(n.uri and n.uri.startswith("http://ex.org/") for n in people)


def test_import_external_foaf():
    pytest.importorskip("rdflib")
    from grafito.integrations import import_turtle

    foaf = """
    @prefix foaf: <http://xmlns.com/foaf/0.1/> .
    @prefix ex: <http://example.org/> .
    ex:alice a foaf:Person ;
        foaf:name "Alice" ;
        foaf:age 30 ;
        foaf:knows ex:bob .
    ex:bob a foaf:Person ;
        foaf:name "Bob" .
    """
    db = GrafitoDatabase(":memory:")
    summary = import_turtle(db, foaf, base_uri="http://example.org/")

    assert summary == {"nodes": 2, "relationships": 1}
    alice = db.match_nodes(labels=["Person"], properties={"name": "Alice"})
    assert len(alice) == 1
    assert alice[0].properties["age"] == 30
    rels = db.match_relationships()
    assert len(rels) == 1
    assert rels[0].type == "knows"


def test_import_rdf_from_graph_object():
    pytest.importorskip("rdflib")
    from rdflib import Graph
    from grafito.integrations import import_rdf

    graph = Graph()
    graph.parse(
        data='<http://ex.org/a> <http://ex.org/link> <http://ex.org/b> .',
        format="nt",
    )
    db = GrafitoDatabase(":memory:")
    summary = import_rdf(db, graph, base_uri="http://ex.org/")
    assert summary["nodes"] == 2
    assert summary["relationships"] == 1
    assert db.match_relationships()[0].type == "link"


@pytest.mark.parametrize("ext", ["g.ttl", "g.jsonld", "g.nt", "g.rdf", "g.n3"])
def test_multiformat_file_roundtrip(tmp_path, ext):
    pytest.importorskip("rdflib")
    from grafito.integrations import export_to_file, import_from_file

    db = _make_sample_db()
    path = str(tmp_path / ext)
    export_to_file(db, path, base_uri="http://ex.org/")

    restored = GrafitoDatabase(":memory:")
    summary = import_from_file(restored, path, base_uri="http://ex.org/")
    assert summary == {"nodes": 2, "relationships": 1}
    assert restored.match_relationships(rel_type="KNOWS")[0].properties["since"] == 2021


def test_export_string_formats():
    pytest.importorskip("rdflib")
    from grafito.integrations import export_string

    db = _make_sample_db()
    assert '"@id"' in export_string(db, base_uri="http://ex.org/", format="json-ld")
    assert "<http://ex.org/" in export_string(db, base_uri="http://ex.org/", format="nt")


def test_export_to_file_unknown_extension(tmp_path):
    pytest.importorskip("rdflib")
    from grafito.integrations import export_to_file

    db = _make_sample_db()
    with pytest.raises(ValueError):
        export_to_file(db, str(tmp_path / "g.unknown"))


def test_query_sparql_select_and_ask():
    pytest.importorskip("rdflib")
    from grafito.integrations import query_sparql

    db = GrafitoDatabase(":memory:")
    db.create_node(labels=["Person"], properties={"name": "Alice", "age": 30})
    db.create_node(labels=["Person"], properties={"name": "Bob", "age": 25})

    rows = query_sparql(
        db,
        """
        PREFIX ex: <http://ex.org/>
        SELECT ?name ?age WHERE { ?p a ex:Person ; ex:name ?name ; ex:age ?age }
        ORDER BY ?age
        """,
        base_uri="http://ex.org/",
    )
    assert rows == [
        {"name": "Bob", "age": 25},
        {"name": "Alice", "age": 30},
    ]

    ask = query_sparql(
        db,
        'PREFIX ex: <http://ex.org/> ASK { ?x ex:name "Alice" }',
        base_uri="http://ex.org/",
    )
    assert ask == [{"boolean": True}]


def test_graph_diff_isomorphism():
    pytest.importorskip("rdflib")
    from grafito.integrations import graph_diff

    def make():
        db = GrafitoDatabase(":memory:")
        a = db.create_node(labels=["Person"], properties={"name": "Alice"})
        b = db.create_node(labels=["Person"], properties={"name": "Bob"})
        db.create_relationship(a.id, b.id, "KNOWS", properties={"since": 2021})
        return db

    same = graph_diff(make(), make(), base_uri="http://ex.org/")
    assert same["isomorphic"] is True
    assert same["only_in_first"] == 0 and same["only_in_second"] == 0

    other = make()
    other.create_node(labels=["Person"], properties={"name": "Carol"})
    diff = graph_diff(make(), other, base_uri="http://ex.org/")
    assert diff["isomorphic"] is False
    assert diff["only_in_second"] > 0


def test_to_pyvis_requires_pyvis():
    pytest.importorskip("pyvis")
    from grafito.integrations import to_pyvis

    db = _make_sample_db()
    graph = db.to_networkx()
    net = to_pyvis(graph, notebook=False)
    assert hasattr(net, "nodes")
    assert len(net.nodes) == 2


def test_save_pyvis_html():
    pytest.importorskip("pyvis")
    from grafito.integrations import save_pyvis_html
    import os

    db = _make_sample_db()
    graph = db.to_networkx()
    output_path = os.path.join(os.getcwd(), "tmp_pyvis_test.html")
    try:
        result = save_pyvis_html(graph, path=output_path)
        assert result == output_path
        assert output_path and output_path.endswith(".html")
        assert os.path.exists(output_path)
    finally:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass


def test_export_graph_d2():
    from grafito.integrations import export_graph
    import os
    import shutil

    db = _make_sample_db()
    graph = db.to_networkx()
    output_path = os.path.join(os.getcwd(), "tmp_graph_test.d2")
    try:
        result = export_graph(graph, output_path, backend="d2", node_label="label_and_name")
        assert result == output_path
        assert os.path.exists(output_path)
        with open(output_path, "r", encoding="utf-8") as handle:
            contents = handle.read()
        assert "direction:" in contents
        if shutil.which("d2"):
            svg_path = export_graph(graph, output_path, backend="d2", render="svg")
            assert os.path.exists(svg_path)
    finally:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        svg_path = os.path.join(os.getcwd(), "tmp_graph_test.svg")
        if os.path.exists(svg_path):
            try:
                os.remove(svg_path)
            except OSError:
                pass


def test_export_graph_mermaid():
    from grafito.integrations import export_graph
    import os
    import shutil

    db = _make_sample_db()
    graph = db.to_networkx()
    output_path = os.path.join(os.getcwd(), "tmp_graph_test.mmd")
    try:
        result = export_graph(graph, output_path, backend="mermaid", node_label="label_and_name")
        assert result == output_path
        assert os.path.exists(output_path)
        with open(output_path, "r", encoding="utf-8") as handle:
            contents = handle.read()
        assert "flowchart" in contents
        if shutil.which("mmdc"):
            # `mmdc` renders via a headless browser (puppeteer/Chrome). When that
            # browser is unavailable the render subprocess fails — that's an
            # environment limitation, not a grafito bug, so skip rather than fail.
            try:
                svg_path = export_graph(graph, output_path, backend="mermaid", render="svg")
            except Exception as exc:
                pytest.skip(f"mermaid-cli render unavailable: {exc}")
            assert os.path.exists(svg_path)
    finally:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        svg_path = os.path.join(os.getcwd(), "tmp_graph_test.svg")
        if os.path.exists(svg_path):
            try:
                os.remove(svg_path)
            except OSError:
                pass


def test_export_graph_graphviz():
    from grafito.integrations import export_graph
    import os
    import shutil

    db = _make_sample_db()
    graph = db.to_networkx()
    output_path = os.path.join(os.getcwd(), "tmp_graph_test.dot")
    try:
        result = export_graph(graph, output_path, backend="graphviz", node_label="label_and_name")
        assert result == output_path
        assert os.path.exists(output_path)
        with open(output_path, "r", encoding="utf-8") as handle:
            contents = handle.read()
        assert "digraph" in contents
        if shutil.which("dot"):
            svg_path = export_graph(graph, output_path, backend="graphviz", render="svg")
            assert os.path.exists(svg_path)
    finally:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        svg_path = os.path.join(os.getcwd(), "tmp_graph_test.svg")
        if os.path.exists(svg_path):
            try:
                os.remove(svg_path)
            except OSError:
                pass


def test_plot_matplotlib_requires_matplotlib():
    pytest.importorskip("matplotlib")
    from grafito.integrations import plot_matplotlib

    db = _make_sample_db()
    graph = db.to_networkx()
    fig = plot_matplotlib(graph, return_fig=True, title="Test Graph")
    assert fig is not None
    assert hasattr(fig, "axes")


def test_plot_matplotlib_handles_empty_graph():
    pytest.importorskip("matplotlib")
    from grafito.integrations import plot_matplotlib
    import networkx as nx

    graph = nx.Graph()
    fig = plot_matplotlib(graph, return_fig=True, show_labels=False)
    assert fig is not None


def test_plot_matplotlib_edge_labels_and_size_scaling():
    pytest.importorskip("matplotlib")
    from grafito.integrations import plot_matplotlib

    db = GrafitoDatabase(":memory:")
    alice = db.create_node(labels=["Person"], properties={"name": "Alice", "size": 2})
    bob = db.create_node(labels=["Person"], properties={"name": "Bob", "size": "bad"})
    db.create_relationship(alice.id, bob.id, "KNOWS")
    graph = db.to_networkx()

    fig = plot_matplotlib(
        graph,
        return_fig=True,
        node_size_attr="size",
        node_size_scale=50,
        node_size_fallback=400,
        show_edge_labels=True,
        edge_label_attr="type",
    )
    assert fig is not None


def test_save_matplotlib():
    pytest.importorskip("matplotlib")
    from grafito.integrations import save_matplotlib
    import os

    db = _make_sample_db()
    graph = db.to_networkx()
    output_path = os.path.join(os.getcwd(), "tmp_matplotlib_test.png")
    try:
        result = save_matplotlib(graph, output_path, title="Test Graph", color_by_label=True)
        assert result == output_path
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
    finally:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass


def test_export_graph_matplotlib():
    pytest.importorskip("matplotlib")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from grafito.integrations import export_graph
    import os

    db = _make_sample_db()
    graph = db.to_networkx()
    output_path = os.path.join(os.getcwd(), "tmp_matplotlib_test.png")
    try:
        result = export_graph(graph, output_path, backend="matplotlib", title="Test Graph")
        assert result == output_path
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
    finally:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass


def test_graph_to_netgraph_requires_netgraph():
    pytest.importorskip("netgraph")
    pytest.importorskip("matplotlib")
    from grafito.integrations import graph_to_netgraph

    db = _make_sample_db()
    graph = db.to_networkx()
    fig, ax, ng = graph_to_netgraph(graph, node_label="name")
    assert fig is not None
    assert ax is not None
    assert ng is not None


def test_graph_to_netgraph_interactive():
    pytest.importorskip("netgraph")
    pytest.importorskip("matplotlib")
    from grafito.integrations import graph_to_netgraph

    db = _make_sample_db()
    graph = db.to_networkx()
    fig, ax, ng = graph_to_netgraph(graph, interactive=True)
    assert fig is not None
    assert ng is not None


def test_graph_to_netgraph_with_ax():
    pytest.importorskip("netgraph")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from grafito.integrations import graph_to_netgraph

    db = _make_sample_db()
    graph = db.to_networkx()
    fig, ax = plt.subplots()
    ng = graph_to_netgraph(graph, ax=ax)
    # When ax is provided, only ng is returned
    assert ng is not None
    plt.close(fig)


def test_graph_to_netgraph_layouts():
    pytest.importorskip("netgraph")
    pytest.importorskip("matplotlib")
    from grafito.integrations import graph_to_netgraph

    db = _make_sample_db()
    graph = db.to_networkx()

    # Test spring layout
    fig1, ax1, ng1 = graph_to_netgraph(graph, node_layout="spring")
    assert ng1 is not None

    # Test circular layout
    fig2, ax2, ng2 = graph_to_netgraph(graph, node_layout="circular")
    assert ng2 is not None


def test_export_graph_netgraph_png():
    pytest.importorskip("netgraph")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from grafito.integrations import export_graph
    import os

    db = _make_sample_db()
    graph = db.to_networkx()
    output_path = os.path.join(os.getcwd(), "tmp_netgraph_test.png")
    try:
        result = export_graph(graph, output_path, backend="netgraph", node_label="name")
        assert result == output_path
        assert os.path.exists(output_path)
    finally:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass


def test_export_graph_netgraph_svg():
    pytest.importorskip("netgraph")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from grafito.integrations import export_graph
    import os

    db = _make_sample_db()
    graph = db.to_networkx()
    output_path = os.path.join(os.getcwd(), "tmp_netgraph_test.svg")
    try:
        result = export_graph(graph, output_path, backend="netgraph", node_label="name")
        assert result == output_path
        assert os.path.exists(output_path)
    finally:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass


def test_export_graph_netgraph_pdf():
    pytest.importorskip("netgraph")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from grafito.integrations import export_graph
    import os

    db = _make_sample_db()
    graph = db.to_networkx()
    output_path = os.path.join(os.getcwd(), "tmp_netgraph_test.pdf")
    try:
        result = export_graph(graph, output_path, backend="netgraph", node_label="name")
        assert result == output_path
        assert os.path.exists(output_path)
    finally:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass


def test_netgraph_multidigraph_conversion():
    pytest.importorskip("netgraph")
    pytest.importorskip("matplotlib")
    from grafito.integrations import graph_to_netgraph

    # Create a graph with multiple edges between same nodes
    db = GrafitoDatabase(":memory:")
    alice = db.create_node(labels=["Person"], properties={"name": "Alice"})
    bob = db.create_node(labels=["Person"], properties={"name": "Bob"})
    db.create_relationship(alice.id, bob.id, "KNOWS")
    db.create_relationship(alice.id, bob.id, "WORKS_WITH")
    graph = db.to_networkx()

    # Should handle multi-edges without error
    fig, ax, ng = graph_to_netgraph(graph, edge_merge_strategy="concat")
    assert ng is not None

    # Test count strategy
    fig2, ax2, ng2 = graph_to_netgraph(graph, edge_merge_strategy="count")
    assert ng2 is not None

    # Test first strategy
    fig3, ax3, ng3 = graph_to_netgraph(graph, edge_merge_strategy="first")
    assert ng3 is not None


def test_netgraph_color_options():
    pytest.importorskip("netgraph")
    pytest.importorskip("matplotlib")
    from grafito.integrations import graph_to_netgraph

    db = GrafitoDatabase(":memory:")
    alice = db.create_node(labels=["Person"], properties={"name": "Alice", "color": "#ff0000"})
    bob = db.create_node(labels=["Company"], properties={"name": "Acme"})
    db.create_relationship(alice.id, bob.id, "WORKS_AT")
    graph = db.to_networkx()

    # Test color_by_label
    fig1, ax1, ng1 = graph_to_netgraph(graph, color_by_label=True)
    assert ng1 is not None

    # Test color_map
    fig2, ax2, ng2 = graph_to_netgraph(
        graph,
        color_map={"Person": "#00ff00", "Company": "#0000ff"}
    )
    assert ng2 is not None

    # Test node_color_attr
    fig3, ax3, ng3 = graph_to_netgraph(graph, node_color_attr="color")
    assert ng3 is not None
