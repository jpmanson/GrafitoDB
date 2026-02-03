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
            svg_path = export_graph(graph, output_path, backend="mermaid", render="svg")
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
