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


def test_plot_matplotlib_custom_styling():
    pytest.importorskip("matplotlib")
    from grafito.integrations import plot_matplotlib

    db = GrafitoDatabase(":memory:")
    alice = db.create_node(labels=["Person"], properties={"name": "Alice", "group": "A"})
    bob = db.create_node(labels=["Person"], properties={"name": "Bob", "group": "B"})
    company = db.create_node(labels=["Company"], properties={"name": "TechCorp", "group": "C"})
    db.create_relationship(alice.id, bob.id, "KNOWS")
    db.create_relationship(alice.id, company.id, "WORKS_AT")

    graph = db.to_networkx()

    # Test with various styling options
    fig = plot_matplotlib(
        graph,
        return_fig=True,
        node_label="name",
        color_by_label=True,
        palette=["#e74c3c", "#3498db", "#2ecc71"],
        node_size=1000,
        font_size=12,
        layout="circular",
        title="Custom Styled Graph"
    )
    assert fig is not None

    # Test coloring by attribute
    fig2 = plot_matplotlib(
        graph,
        return_fig=True,
        node_label="name",
        color_by_label=False,
        color_attr="group",
        color_map={"A": "#ff6b6b", "B": "#4ecdc4", "C": "#ffe66d"},
        layout="spring",
        layout_kwargs={"seed": 42},
        title="Colored by Group"
    )
    assert fig2 is not None
