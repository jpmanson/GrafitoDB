"""Optional integrations."""

__all__ = [
    "export_rdf",
    "export_turtle",
    "to_pyvis",
    "save_pyvis_html",
    "plot_matplotlib",
    "save_matplotlib",
    "render_graph",
    "export_graph",
    "available_viz_backends",
    "register_viz_backend",
    "graph_to_netgraph",
]


def __getattr__(name: str):
    if name in ("export_rdf", "export_turtle"):
        from .rdf import export_rdf, export_turtle

        return export_rdf if name == "export_rdf" else export_turtle
    if name in ("to_pyvis", "save_pyvis_html", "plot_matplotlib", "save_matplotlib", "render_graph", "export_graph", "available_viz_backends", "register_viz_backend", "graph_to_netgraph"):
        from .viz import (
            to_pyvis,
            save_pyvis_html,
            plot_matplotlib,
            save_matplotlib,
            render_graph,
            export_graph,
            available_viz_backends,
            register_viz_backend,
            graph_to_netgraph,
        )

        return {
            "to_pyvis": to_pyvis,
            "save_pyvis_html": save_pyvis_html,
            "plot_matplotlib": plot_matplotlib,
            "save_matplotlib": save_matplotlib,
            "render_graph": render_graph,
            "export_graph": export_graph,
            "available_viz_backends": available_viz_backends,
            "register_viz_backend": register_viz_backend,
            "graph_to_netgraph": graph_to_netgraph,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
