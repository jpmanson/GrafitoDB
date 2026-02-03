"""Visualization helpers (PyVis)."""

from __future__ import annotations

from typing import Any, Callable, Protocol
import math
import json
import shutil
import subprocess

# Default color palette for visualizations
_DEFAULT_COLORS = [
    "#8ecae6",
    "#219ebc",
    "#023047",
    "#ffb703",
    "#fb8500",
    "#90be6d",
    "#f94144",
    "#577590",
    "#43aa8b",
    "#f3722c",
]


def _resolve_node_label(
    node_id: Any,
    attrs: dict[str, Any],
    node_label: str = "id",
    label_attr: str | None = None,
    label_fn: Callable[[Any, dict[str, Any]], str] | None = None,
) -> str:
    """Resolve the display label for a node."""
    if label_fn:
        return str(label_fn(node_id, attrs))
    properties = attrs.get("properties", {})
    if label_attr:
        if label_attr in attrs:
            return str(attrs[label_attr])
        if label_attr in properties:
            return str(properties[label_attr])
    if node_label == "name":
        return str(properties.get("name", node_id))
    if node_label == "labels":
        labels = attrs.get("labels", [])
        return ":".join(labels) if labels else str(node_id)
    if node_label == "label_and_name":
        labels = attrs.get("labels", [])
        label_prefix = ":".join(labels) + " " if labels else ""
        return f"{label_prefix}{properties.get('name', node_id)}"
    return str(node_id)

_DEFAULT_COLORS = [
    "#8ecae6",
    "#219ebc",
    "#023047",
    "#ffb703",
    "#fb8500",
    "#90be6d",
    "#f94144",
    "#577590",
    "#43aa8b",
    "#f3722c",
]


def _resolve_label(
    node_id: Any,
    attrs: dict[str, Any],
    node_label: str = "id",
    label_attr: str | None = None,
    label_fn: Callable[[Any, dict[str, Any]], str] | None = None,
) -> str:
    """Resolve the label to display for a node."""
    if label_fn:
        return str(label_fn(node_id, attrs))
    properties = attrs.get("properties", {})
    if label_attr:
        if label_attr in attrs:
            return str(attrs[label_attr])
        if label_attr in properties:
            return str(properties[label_attr])
    if node_label == "name":
        return str(properties.get("name", node_id))
    if node_label == "labels":
        labels = attrs.get("labels", [])
        return ":".join(labels) if labels else str(node_id)
    if node_label == "label_and_name":
        labels = attrs.get("labels", [])
        label_prefix = ":".join(labels) + " " if labels else ""
        return f"{label_prefix}{properties.get('name', node_id)}"
    return str(node_id)


def _multidigraph_to_digraph(
    graph,
    edge_separator: str = " | ",
    merge_strategy: str = "concat",
):
    """Convert MultiDiGraph to DiGraph for netgraph compatibility.

    Args:
        graph: A NetworkX MultiDiGraph.
        edge_separator: Separator for concatenating edge types.
        merge_strategy: How to merge multiple edges:
            - "concat": Concatenate edge types with separator
            - "first": Keep only the first edge's attributes
            - "count": Keep first edge attrs and add _edge_count

    Returns:
        A NetworkX DiGraph with merged edges.
    """
    import networkx as nx

    simple = nx.DiGraph()
    for node_id, attrs in graph.nodes(data=True):
        simple.add_node(node_id, **attrs)

    edge_data: dict[tuple, list[dict]] = {}
    for source, target, key, attrs in graph.edges(keys=True, data=True):
        edge_key = (source, target)
        if edge_key not in edge_data:
            edge_data[edge_key] = []
        edge_data[edge_key].append(attrs)

    for (source, target), attrs_list in edge_data.items():
        if merge_strategy == "first":
            merged = attrs_list[0].copy()
        elif merge_strategy == "count":
            merged = attrs_list[0].copy()
            merged["_edge_count"] = len(attrs_list)
        else:  # concat
            types = [a.get("type", "RELATED_TO") for a in attrs_list]
            merged = attrs_list[0].copy()
            merged["type"] = edge_separator.join(types)
        simple.add_edge(source, target, **merged)

    return simple


def to_pyvis(
    graph,
    notebook: bool = True,
    directed: bool = True,
    node_label: str = "id",
    label_attr: str | None = None,
    label_fn: Callable[[Any, dict[str, Any]], str] | None = None,
    node_color_attr: str | None = None,
    color_map: dict[str, str] | None = None,
    color_by_label: bool = False,
    default_color: str = "#8ecae6",
    palette: list[str] | None = None,
    **kwargs: Any,
):
    """Convert a NetworkX graph into a PyVis Network."""
    try:
        from pyvis.network import Network
    except ImportError as exc:
        raise ImportError(
            "pyvis is not installed. Install with `pip install grafito[viz]` "
            "or `uv pip install grafito[viz]`."
        ) from exc
    net = Network(notebook=notebook, directed=directed, **kwargs)
    palette = palette or _DEFAULT_COLORS
    label_colors: dict[str, str] = {}

    def pick_label_color(labels: list[str]) -> str:
        if not labels:
            return default_color
        for label in labels:
            if color_map and label in color_map:
                return color_map[label]
        if not color_by_label:
            return default_color
        label = labels[0]
        if label not in label_colors:
            label_colors[label] = palette[len(label_colors) % len(palette)]
        return label_colors[label]

    for node_id, attrs in graph.nodes(data=True):
        labels = attrs.get("labels", [])
        title = attrs.get("properties", {}).get("name", str(node_id))
        color = default_color
        if node_color_attr:
            props = attrs.get("properties", {})
            color = props.get(node_color_attr) or attrs.get(node_color_attr) or default_color
        else:
            color = pick_label_color(labels)
        net.add_node(
            node_id,
            label=_resolve_label(node_id, attrs, node_label, label_attr, label_fn),
            title=f"{labels} {title}",
            color=color,
        )
    for source, target, key, attrs in graph.edges(keys=True, data=True):
        rel_type = attrs.get("type", "RELATED_TO")
        net.add_edge(source, target, label=rel_type)
    return net


_PHYSICS_PRESETS = {
    "compact": {
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -8000,
                "centralGravity": 0.4,
                "springLength": 80,
                "springConstant": 0.04,
            }
        }
    },
    "spread": {
        "physics": {
            "repulsion": {
                "nodeDistance": 200,
                "springLength": 200,
                "springConstant": 0.01,
            }
        }
    },
}


def plot_matplotlib(
    graph,
    figsize: tuple[int, int] = (10, 8),
    dpi: int = 100,
    layout: str | Callable = "spring",
    layout_kwargs: dict[str, Any] | None = None,
    # Node styling
    node_label: str = "id",
    label_attr: str | None = None,
    label_fn: Callable[[Any, dict[str, Any]], str] | None = None,
    node_color: str | list[str] | None = None,
    color_by_label: bool = True,
    color_attr: str | None = None,
    color_map: dict[str, str] | None = None,
    node_size: int | list[int] = 600,
    node_size_attr: str | None = None,
    node_size_scale: float = 100.0,
    node_size_fallback: float = 600.0,
    node_alpha: float = 0.9,
    node_shape: str = "o",
    node_edge_color: str = "#333333",
    node_linewidth: float = 1.5,
    # Edge styling
    edge_color: str = "#666666",
    edge_alpha: float = 0.6,
    edge_width: float = 1.5,
    edge_style: str = "solid",
    edge_arrow_size: int = 15,
    # Edge labels
    show_edge_labels: bool = True,
    edge_label_attr: str = "type",
    edge_label_fn: Callable[[Any, Any, int | None, dict[str, Any]], str] | None = None,
    edge_font_size: int = 9,
    edge_font_color: str = "#556",
    edge_label_rotate: bool = True,
    # Label styling
    font_size: int = 10,
    font_color: str = "#1b1f23",
    font_weight: str = "normal",
    show_labels: bool = True,
    label_offset: tuple[float, float] | str = (0, 0.05),
    # Legend
    show_legend: bool = True,
    legend_loc: str = "upper left",
    legend_bbox_to_anchor: tuple[float, float] | None = None,
    # Title
    title: str | None = None,
    title_fontsize: int = 14,
    title_fontweight: str = "bold",
    # Background
    bgcolor: str = "#ffffff",
    # Axes
    show_axes: bool = False,
    # Margins
    margins: tuple[float, float] = (0.1, 0.1),
    # Return mode
    return_fig: bool = False,
    palette: list[str] | None = None,
    **kwargs: Any,
):
    """Plot a NetworkX graph using Matplotlib.

    This function provides extensive customization options for creating
    static graph visualizations. It uses NetworkX for layout calculation
    and Matplotlib for rendering.

    Args:
        graph: NetworkX graph to plot
        figsize: Figure size as (width, height) in inches
        dpi: Dots per inch for the figure
        layout: Layout algorithm name ('spring', 'circular', 'random',
            'shell', 'spectral', 'kamada_kawai') or callable
        layout_kwargs: Additional arguments passed to the layout function
        node_label: Property to use for node labels ('id', 'name', 'labels',
            'label_and_name')
        label_attr: Specific attribute to use for labels (overrides node_label)
        label_fn: Custom function (node_id, attrs) -> str for labels
        node_color: Single color, list of colors, or None for auto-coloring
        color_by_label: Whether to color nodes by their labels
        color_attr: Property attribute to use for coloring groups
        color_map: Dictionary mapping label values to colors
        node_size: Size of nodes (single value or list)
        node_size_attr: Property to use for sizing nodes
        node_size_scale: Scale factor for node_size_attr-derived sizes
        node_size_fallback: Fallback size when node_size_attr values are invalid
        node_alpha: Node transparency (0-1)
        node_shape: Node shape marker (Matplotlib style, e.g., 'o', 's', '^')
        node_edge_color: Color of node borders
        node_linewidth: Width of node border lines
        edge_color: Color of edges
        edge_alpha: Edge transparency (0-1)
        edge_width: Width of edges
        edge_style: Edge line style ('solid', 'dashed', 'dotted', etc.)
        edge_arrow_size: Size of arrow heads for directed graphs
        show_edge_labels: Whether to display edge labels
        edge_label_attr: Attribute to use for edge labels
        edge_label_fn: Custom function (source, target, key, attrs) -> str for edge labels
        edge_font_size: Font size for edge labels
        edge_font_color: Color for edge labels
        edge_label_rotate: Rotate edge labels to align with edge direction
        font_size: Font size for node labels
        font_color: Color of node labels
        font_weight: Font weight for labels ('normal', 'bold', etc.)
        show_labels: Whether to display node labels
        label_offset: Offset for labels as (x, y) fraction or "auto"
        show_legend: Whether to show a legend for node colors
        legend_loc: Legend location (Matplotlib style)
        legend_bbox_to_anchor: Legend anchor position
        title: Figure title
        title_fontsize: Title font size
        title_fontweight: Title font weight
        bgcolor: Background color
        show_axes: Whether to show axis lines
        margins: Margins around the graph as (x, y)
        return_fig: If True, return the figure object instead of displaying
        palette: Custom color palette for auto-coloring
        **kwargs: Additional arguments passed to nx.draw_networkx

    Returns:
        matplotlib.figure.Figure if return_fig=True, None otherwise

    Raises:
        ImportError: If matplotlib or networkx is not installed

    Example:
        >>> from grafito import GrafitoDatabase
        >>> from grafito.integrations import plot_matplotlib
        >>> db = GrafitoDatabase(':memory:')
        >>> alice = db.create_node(labels=['Person'], properties={'name': 'Alice'})
        >>> bob = db.create_node(labels=['Person'], properties={'name': 'Bob'})
        >>> db.create_relationship(alice.id, bob.id, 'KNOWS')
        >>> graph = db.to_networkx()
        >>> plot_matplotlib(
        ...     graph,
        ...     figsize=(12, 10),
        ...     color_by_label=True,
        ...     node_size=800,
        ...     node_size_scale=80,
        ...     node_size_fallback=500,
        ...     show_edge_labels=True,
        ...     title="My Social Network"
        ... )
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "matplotlib and networkx are required. "
            "Install with `pip install matplotlib networkx`"
        ) from exc

    # Resolve layout
    layout_kwargs = layout_kwargs or {}
    if isinstance(layout, str):
        layout_map = {
            "spring": nx.spring_layout,
            "circular": nx.circular_layout,
            "random": nx.random_layout,
            "shell": nx.shell_layout,
            "spectral": nx.spectral_layout,
            "kamada_kawai": nx.kamada_kawai_layout,
            "planar": nx.planar_layout,
            "fruchterman_reingold": nx.fruchterman_reingold_layout,
        }
        if layout not in layout_map:
            raise ValueError(f"Unknown layout '{layout}'. Available: {list(layout_map.keys())}")
        pos = layout_map[layout](graph, **layout_kwargs)
    else:
        pos = layout(graph, **layout_kwargs)

    # Resolve node colors
    palette = palette or _DEFAULT_COLORS
    label_colors: dict[str, str] = {}
    legend_colors: dict[str, str] = {}

    def pick_label_color(labels: list[str]) -> str:
        if not labels:
            return palette[0]
        label = labels[0]
        if label not in label_colors:
            label_colors[label] = palette[len(label_colors) % len(palette)]
        return label_colors[label]

    if node_color is None:
        if color_attr:
            # Color by a specific attribute
            unique_values: set[str] = set()
            for _, attrs in graph.nodes(data=True):
                props = attrs.get("properties", {})
                if color_attr in props:
                    val = props.get(color_attr)
                else:
                    val = attrs.get(color_attr)
                if val is not None:
                    unique_values.add(str(val))
            value_colors = {v: palette[i % len(palette)] for i, v in enumerate(sorted(unique_values))}
            legend_colors = dict(value_colors)
            node_color = []
            for _, attrs in graph.nodes(data=True):
                props = attrs.get("properties", {})
                if color_attr in props:
                    val = props.get(color_attr)
                else:
                    val = attrs.get(color_attr)
                mapped = value_colors.get(str(val), palette[0])
                if color_map:
                    mapped = color_map.get(str(val), mapped)
                    if val is not None:
                        legend_colors[str(val)] = mapped
                node_color.append(mapped)
        elif color_by_label:
            node_color = []
            for _, attrs in graph.nodes(data=True):
                labels = attrs.get("labels", [])
                if color_map and labels:
                    color = color_map.get(labels[0], pick_label_color(labels))
                    legend_colors[labels[0]] = color
                else:
                    color = pick_label_color(labels)
                    if labels:
                        legend_colors[labels[0]] = color
                node_color.append(color)
        else:
            node_color = palette[0]

    # Resolve node sizes
    node_size_by_node: dict[Any, float] | None = None
    if node_size_attr:
        sizes = []
        for node_id, attrs in graph.nodes(data=True):
            props = attrs.get("properties", {})
            if node_size_attr in props:
                val = props.get(node_size_attr)
            else:
                val = attrs.get(node_size_attr, 1)
            try:
                sizes.append(float(val) * node_size_scale)
            except (ValueError, TypeError):
                sizes.append(node_size_fallback)
        node_size = sizes
        node_size_by_node = {node_id: float(size) for node_id, size in zip(graph.nodes(), sizes)}
    elif isinstance(node_size, (list, tuple)):
        node_size_by_node = {
            node_id: float(size)
            for node_id, size in zip(graph.nodes(), node_size)
        }

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=bgcolor)
    ax.set_facecolor(bgcolor)

    # Draw edges
    is_directed = graph.is_directed() if hasattr(graph, "is_directed") else False
    edge_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k.startswith("edge_")
        and not k.startswith("edge_label_")
        and k not in {"edge_color", "edge_alpha", "edge_width", "edge_style"}
    }
    if is_directed:
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edge_color=edge_color,
            alpha=edge_alpha,
            width=edge_width,
            style=edge_style,
            arrows=True,
            arrowsize=edge_arrow_size,
            **edge_kwargs,
        )
    else:
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edge_color=edge_color,
            alpha=edge_alpha,
            width=edge_width,
            style=edge_style,
            **edge_kwargs,
        )

    # Draw nodes
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_color,
        node_size=node_size,
        alpha=node_alpha,
        node_shape=node_shape,
        edgecolors=node_edge_color,
        linewidths=node_linewidth,
        **{k: v for k, v in kwargs.items() if k.startswith("node_") and k not in [
            "node_color", "node_size", "node_shape", "node_edge_color"
        ]},
    )

    # Draw edge labels
    if show_edge_labels:
        edge_labels = {}
        is_multi = graph.is_multigraph() if hasattr(graph, "is_multigraph") else False
        if is_multi:
            for source, target, key, attrs in graph.edges(keys=True, data=True):
                if edge_label_fn:
                    label = edge_label_fn(source, target, key, attrs)
                else:
                    label = attrs.get(edge_label_attr, "")
                edge_labels[(source, target, key)] = str(label)
        else:
            for source, target, attrs in graph.edges(data=True):
                if edge_label_fn:
                    label = edge_label_fn(source, target, None, attrs)
                else:
                    label = attrs.get(edge_label_attr, "")
                edge_labels[(source, target)] = str(label)

        edge_label_kwargs = {
            k[len("edge_label_"):]: v
            for k, v in kwargs.items()
            if k.startswith("edge_label_")
        }
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels=edge_labels,
            ax=ax,
            font_size=edge_font_size,
            font_color=edge_font_color,
            rotate=edge_label_rotate,
            **edge_label_kwargs,
        )

    # Pre-compute limits for label transforms
    xlim = None
    ylim = None
    if pos:
        xlim = (min(x for x, y in pos.values()) - margins[0],
                max(x for x, y in pos.values()) + margins[0])
        ylim = (min(y for x, y in pos.values()) - margins[1],
                max(y for x, y in pos.values()) + margins[1])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # Draw labels
    if show_labels:
        labels = {}
        for node_id, attrs in graph.nodes(data=True):
            labels[node_id] = _resolve_node_label(node_id, attrs, node_label, label_attr, label_fn)

        # Apply label offset
        if label_offset == "auto":
            offset_pos = {}
            for node, (x, y) in pos.items():
                size = node_size_by_node.get(node, node_size_fallback) if node_size_by_node else node_size
                try:
                    size_val = float(size)
                except (ValueError, TypeError):
                    size_val = node_size_fallback
                radius_pts = math.sqrt(max(size_val, 0.0)) / 2.0
                text_half_pts = max(1.0, float(font_size) * 0.6)
                pad_pts = 3.8 + min(9.0, radius_pts * 0.4)
                total_pts = radius_pts + text_half_pts + pad_pts
                x_disp, y_disp = ax.transData.transform((x, y))
                x_data, y_data = ax.transData.inverted().transform((x_disp, y_disp + total_pts))
                offset_pos[node] = (x_data, y_data)
        elif label_offset != (0, 0):
            offset_pos = {
                node: (x + label_offset[0], y + label_offset[1])
                for node, (x, y) in pos.items()
            }
        else:
            offset_pos = pos

        nx.draw_networkx_labels(
            graph,
            offset_pos,
            labels,
            ax=ax,
            font_size=font_size,
            font_color=font_color,
            font_weight=font_weight,
        )

    # Add legend
    if show_legend and (legend_colors or label_colors):
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=color, edgecolor=node_edge_color, label=label)
            for label, color in sorted((legend_colors or label_colors).items())
        ]
        if legend_bbox_to_anchor:
            ax.legend(
                handles=legend_elements,
                loc=legend_loc,
                bbox_to_anchor=legend_bbox_to_anchor,
                framealpha=0.9,
            )
        else:
            ax.legend(handles=legend_elements, loc=legend_loc, framealpha=0.9)

    # Set title
    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight)

    # Configure axes
    if not show_axes:
        ax.axis("off")
    if xlim and ylim:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    plt.tight_layout()

    if return_fig:
        return fig
    plt.show()
    return None


def save_matplotlib(
    graph,
    path: str,
    format: str | None = None,
    bbox_inches: str = "tight",
    pad_inches: float = 0.1,
    **kwargs: Any,
) -> str:
    """Save a NetworkX graph visualization to a file using Matplotlib.

    Args:
        graph: NetworkX graph to plot
        path: Output file path
        format: File format ('png', 'svg', 'pdf', etc.). Auto-detected from path if None
        bbox_inches: Bounding box setting ('tight' to remove extra whitespace)
        pad_inches: Padding around the figure when bbox_inches='tight'
        **kwargs: Additional arguments passed to plot_matplotlib

    Returns:
        The output file path

    Example:
        >>> save_matplotlib(graph, 'network.png', figsize=(10, 8), color_by_label=True)
        >>> save_matplotlib(graph, 'network.svg', format='svg', title="My Graph")
    """
    fig = plot_matplotlib(graph, return_fig=True, **kwargs)
    if fig is None:
        raise RuntimeError("Failed to create figure")

    try:
        fig.savefig(path, format=format, bbox_inches=bbox_inches, pad_inches=pad_inches)
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)

    return path


class MatplotlibBackend:
    """Matplotlib backend adapter for static visualizations."""

    name = "matplotlib"

    def render(self, graph, **kwargs: Any):
        """Render graph and return the matplotlib figure."""
        return plot_matplotlib(graph, return_fig=True, **kwargs)

    def export(self, graph, path: str, **kwargs: Any) -> str:
        """Render graph and save to a file path."""
        return save_matplotlib(graph, path=path, **kwargs)


class VizBackend(Protocol):
    """Visualization backend interface."""

    name: str

    def render(self, graph, **kwargs: Any) -> Any:
        """Render graph and return backend-specific object."""

    def export(self, graph, path: str, **kwargs: Any) -> str:
        """Render graph and save to a file path."""


class PyVisBackend:
    """PyVis backend adapter."""

    name = "pyvis"

    def render(self, graph, **kwargs: Any):
        return to_pyvis(graph, **kwargs)

    def export(self, graph, path: str, **kwargs: Any) -> str:
        return save_pyvis_html(graph, path=path, **kwargs)

class D2Backend:
    """D2 backend adapter (text export, optional render via d2 CLI)."""

    name = "d2"

    def render(self, graph, **kwargs: Any) -> str:
        return graph_to_d2(graph, **kwargs)

    def export(self, graph, path: str, **kwargs: Any) -> str:
        d2_kwargs = dict(kwargs)
        d2_kwargs.pop("render", None)
        d2_kwargs.pop("theme", None)
        d2_text = graph_to_d2(graph, **d2_kwargs)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(d2_text)
        render = kwargs.get("render")
        if render:
            render_d2(path, render=render, theme=kwargs.get("theme"))
        return path

class MermaidBackend:
    """Mermaid backend adapter (text export, optional render via mmdc CLI)."""

    name = "mermaid"

    def render(self, graph, **kwargs: Any) -> str:
        return graph_to_mermaid(graph, **kwargs)

    def export(self, graph, path: str, **kwargs: Any) -> str:
        mermaid_kwargs = dict(kwargs)
        mermaid_kwargs.pop("render", None)
        mermaid_text = graph_to_mermaid(graph, **mermaid_kwargs)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(mermaid_text)
        render = kwargs.get("render")
        if render:
            render_mermaid(path, render=render, theme=kwargs.get("theme"))
        return path

class GraphvizBackend:
    """Graphviz DOT backend adapter (text export, optional render via dot CLI)."""

    name = "graphviz"

    def render(self, graph, **kwargs: Any) -> str:
        return graph_to_dot(graph, **kwargs)

    def export(self, graph, path: str, **kwargs: Any) -> str:
        dot_kwargs = dict(kwargs)
        dot_kwargs.pop("render", None)
        engine = dot_kwargs.pop("engine", "dot")
        dot_text = graph_to_dot(graph, **dot_kwargs)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(dot_text)
        render = kwargs.get("render")
        if render:
            render_dot(path, render=render, engine=engine)
        return path

class D3Backend:
    """D3 backend adapter (self-contained HTML export)."""

    name = "d3"

    def render(self, graph, **kwargs: Any) -> str:
        return graph_to_d3_html(graph, **kwargs)

    def export(self, graph, path: str, **kwargs: Any) -> str:
        html = graph_to_d3_html(graph, **kwargs)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(html)
        return path

class CytoscapeBackend:
    """Cytoscape.js backend adapter (self-contained HTML export)."""

    name = "cytoscape"

    def render(self, graph, **kwargs: Any) -> str:
        return graph_to_cytoscape_html(graph, **kwargs)

    def export(self, graph, path: str, **kwargs: Any) -> str:
        html = graph_to_cytoscape_html(graph, **kwargs)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(html)
        return path


class NetgraphBackend:
    """Netgraph backend adapter (matplotlib-based publication quality graphs)."""

    name = "netgraph"

    def render(self, graph, **kwargs: Any):
        return graph_to_netgraph(graph, **kwargs)

    def export(self, graph, path: str, **kwargs: Any) -> str:
        # Extract matplotlib savefig options
        export_kwargs = dict(kwargs)
        dpi = export_kwargs.pop("dpi", 150)
        bbox_inches = export_kwargs.pop("bbox_inches", "tight")
        # Force non-interactive for export
        export_kwargs["interactive"] = False

        result = graph_to_netgraph(graph, **export_kwargs)
        if isinstance(result, tuple):
            fig, ax, ng = result
        else:
            # ax was provided, get figure from the netgraph instance
            fig = result.ax.figure

        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
        return path


_VIZ_BACKENDS: dict[str, VizBackend] = {
    "pyvis": PyVisBackend(),
    "matplotlib": MatplotlibBackend(),
    "d2": D2Backend(),
    "mermaid": MermaidBackend(),
    "graphviz": GraphvizBackend(),
    "d3": D3Backend(),
    "cytoscape": CytoscapeBackend(),
    "netgraph": NetgraphBackend(),
}


def register_viz_backend(name: str, backend: VizBackend, override: bool = False) -> None:
    """Register a visualization backend."""
    if not name:
        raise ValueError("Backend name cannot be empty")
    if not override and name in _VIZ_BACKENDS:
        raise ValueError(f"Backend '{name}' already registered")
    _VIZ_BACKENDS[name] = backend


def available_viz_backends() -> list[str]:
    """List registered visualization backends."""
    return sorted(_VIZ_BACKENDS.keys())


def render_graph(graph, backend: str = "pyvis", **kwargs: Any) -> Any:
    """Render a graph using a registered backend and return its object."""
    if backend not in _VIZ_BACKENDS:
        available = ", ".join(available_viz_backends())
        raise ValueError(f"Unknown backend '{backend}'. Available: {available}")
    return _VIZ_BACKENDS[backend].render(graph, **kwargs)


def export_graph(
    graph,
    path: str,
    backend: str = "pyvis",
    **kwargs: Any,
) -> str:
    """Render a graph using a registered backend and save to a file."""
    if backend not in _VIZ_BACKENDS:
        available = ", ".join(available_viz_backends())
        raise ValueError(f"Unknown backend '{backend}'. Available: {available}")
    return _VIZ_BACKENDS[backend].export(graph, path, **kwargs)


def _safe_d2_id(value: Any) -> str:
    text = str(value).replace("\"", "'")
    return f"\"{text}\""


def graph_to_d2(
    graph,
    node_label: str = "id",
    label_attr: str | None = None,
    label_fn: Callable[[Any, dict[str, Any]], str] | None = None,
    include_properties: bool = False,
) -> str:
    """Serialize a NetworkX graph to D2 text."""
    lines: list[str] = ["direction: right"]

    for node_id, attrs in graph.nodes(data=True):
        node_key = _safe_d2_id(node_id)
        label = _resolve_label(node_id, attrs, node_label, label_attr, label_fn)
        lines.append(f"{node_key}: { _safe_d2_id(label) }")
        if include_properties:
            props = attrs.get("properties", {})
            for key, value in props.items():
                lines.append(f"{node_key}.{key}: { _safe_d2_id(value) }")

    for source, target, key, attrs in graph.edges(keys=True, data=True):
        rel_type = attrs.get("type", "RELATED_TO")
        lines.append(f"{_safe_d2_id(source)} -> {_safe_d2_id(target)}: { _safe_d2_id(rel_type) }")

    return "\n".join(lines) + "\n"


def render_d2(path: str, render: str = "svg", theme: str | None = None) -> str:
    """Render a .d2 file using the d2 CLI."""
    binary = shutil.which("d2")
    if not binary:
        raise RuntimeError(
            "d2 CLI not found. Install with `brew install d2` and ensure it is on PATH."
        )
    output = path.rsplit(".", 1)[0] + f".{render}"
    cmd = [binary, path, output]
    if theme:
        cmd.extend(["--theme", theme])
    subprocess.run(cmd, check=True)
    return output


def graph_to_mermaid(
    graph,
    node_label: str = "id",
    label_attr: str | None = None,
    label_fn: Callable[[Any, dict[str, Any]], str] | None = None,
    include_properties: bool = False,
    direction: str = "LR",
) -> str:
    """Serialize a NetworkX graph to Mermaid flowchart text."""
    lines: list[str] = [f"flowchart {direction}"]

    def mermaid_id(value: Any) -> str:
        text = str(value)
        safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text)
        return f"n_{safe}" if safe and safe[0].isdigit() else safe or "n"

    node_ids: dict[Any, str] = {}
    for node_id, attrs in graph.nodes(data=True):
        m_id = mermaid_id(node_id)
        node_ids[node_id] = m_id
        label = _resolve_label(node_id, attrs, node_label, label_attr, label_fn)
        lines.append(f'{m_id}["{label}"]')
        if include_properties:
            props = attrs.get("properties", {})
            for key, value in props.items():
                lines.append(f'{m_id} -- "{key}" --> {mermaid_id(f"{m_id}_{key}")}["{value}"]')

    for source, target, key, attrs in graph.edges(keys=True, data=True):
        rel_type = attrs.get("type", "RELATED_TO")
        lines.append(f'{node_ids.get(source)} -- "{rel_type}" --> {node_ids.get(target)}')

    return "\n".join(lines) + "\n"


def render_mermaid(path: str, render: str = "svg", theme: str | None = None) -> str:
    """Render a .mmd file using the mermaid-cli (mmdc)."""
    binary = shutil.which("mmdc")
    if not binary:
        raise RuntimeError(
            "mermaid-cli (mmdc) not found. Install with `npm i -g @mermaid-js/mermaid-cli`."
        )
    output = path.rsplit(".", 1)[0] + f".{render}"
    cmd = [binary, "-i", path, "-o", output]
    if theme:
        cmd.extend(["-t", theme])
    subprocess.run(cmd, check=True)
    return output


def graph_to_dot(
    graph,
    node_label: str = "id",
    label_attr: str | None = None,
    label_fn: Callable[[Any, dict[str, Any]], str] | None = None,
    include_properties: bool = False,
    directed: bool = True,
    graph_attrs: dict[str, Any] | None = None,
    node_attrs: dict[str, Any] | None = None,
    edge_attrs: dict[str, Any] | None = None,
) -> str:
    """Serialize a NetworkX graph to Graphviz DOT text."""
    graph_type = "digraph" if directed else "graph"
    edge_op = "->" if directed else "--"
    lines: list[str] = [f"{graph_type} grafito {{"]

    graph_attrs = graph_attrs or {"overlap": "false", "splines": "true"}
    node_attrs = node_attrs or {
        "shape": "ellipse",
        "fontsize": "10",
        "fontname": "Helvetica",
        "fontcolor": "#1b1f23",
    }
    edge_attrs = edge_attrs or {
        "fontsize": "9",
        "fontname": "Helvetica",
        "fontcolor": "#556",
    }

    def _attrs_to_dot(attrs: dict[str, Any]) -> str:
        parts = []
        for key, value in attrs.items():
            escaped = str(value).replace('"', '\\"')
            parts.append(f'{key}="{escaped}"')
        return ", ".join(parts)

    if graph_attrs:
        lines.append(f"  graph [{_attrs_to_dot(graph_attrs)}];")
    if node_attrs:
        lines.append(f"  node [{_attrs_to_dot(node_attrs)}];")
    if edge_attrs:
        lines.append(f"  edge [{_attrs_to_dot(edge_attrs)}];")

    for node_id, attrs in graph.nodes(data=True):
        label = _resolve_label(node_id, attrs, node_label, label_attr, label_fn).replace("\"", "\\\"")
        lines.append(f'  "{node_id}" [label="{label}"];')
        if include_properties:
            props = attrs.get("properties", {})
            for key, value in props.items():
                prop_node = f"{node_id}_{key}"
                lines.append(f'  "{prop_node}" [label="{value}"];')
                lines.append(f'  "{node_id}" {edge_op} "{prop_node}" [label="{key}"];')

    for source, target, key, attrs in graph.edges(keys=True, data=True):
        rel_type = attrs.get("type", "RELATED_TO").replace("\"", "\\\"")
        lines.append(f'  "{source}" {edge_op} "{target}" [label="{rel_type}"];')

    lines.append("}")
    return "\n".join(lines) + "\n"


def render_dot(path: str, render: str = "svg", engine: str = "dot") -> str:
    """Render a .dot file using the Graphviz CLI."""
    binary = shutil.which(engine)
    if not binary:
        raise RuntimeError(
            f"Graphviz '{engine}' not found. Install with `brew install graphviz`."
        )
    output = path.rsplit(".", 1)[0] + f".{render}"
    cmd = [binary, f"-T{render}", path, "-o", output]
    subprocess.run(cmd, check=True)
    return output


def graph_to_d3_html(
    graph,
    width: int = 960,
    height: int = 600,
    node_label: str = "id",
    label_attr: str | None = None,
    label_fn: Callable[[Any, dict[str, Any]], str] | None = None,
    color_by_label: bool = True,
    palette: list[str] | None = None,
) -> str:
    """Serialize a NetworkX graph into a self-contained D3 HTML document."""
    palette = palette or _DEFAULT_COLORS
    label_colors: dict[str, str] = {}

    def pick_label_color(labels: list[str]) -> str:
        if not color_by_label or not labels:
            return _DEFAULT_COLORS[0]
        label = labels[0]
        if label not in label_colors:
            label_colors[label] = palette[len(label_colors) % len(palette)]
        return label_colors[label]

    nodes = []
    for node_id, attrs in graph.nodes(data=True):
        labels = attrs.get("labels", [])
        nodes.append(
            {
                "id": str(node_id),
                "label": _resolve_label(node_id, attrs, node_label, label_attr, label_fn),
                "group": labels[0] if labels else "Node",
                "color": pick_label_color(labels),
                "title": attrs.get("properties", {}).get("name", str(node_id)),
            }
        )

    links = []
    for source, target, key, attrs in graph.edges(keys=True, data=True):
        links.append(
            {
                "source": str(source),
                "target": str(target),
                "label": attrs.get("type", "RELATED_TO"),
            }
        )

    data = {"nodes": nodes, "links": links}
    data_json = json.dumps(data)

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Grafito D3 Graph</title>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; }}
    svg {{ width: 100vw; height: 100vh; background: #f8f9fb; }}
    .link {{ stroke: #9aa4b2; stroke-width: 1.2px; }}
    .node text {{ font-size: 12px; fill: #1b1f23; pointer-events: none; }}
    .edge-label {{ font-size: 10px; fill: #556; }}
    .tooltip {{
      position: absolute;
      background: rgba(15, 23, 42, 0.95);
      color: #f8fafc;
      padding: 6px 8px;
      border-radius: 6px;
      font-size: 12px;
      pointer-events: none;
      opacity: 0;
      transition: opacity 0.15s ease;
    }}
  </style>
  <script src="https://d3js.org/d3.v7.min.js"></script>
</head>
<body>
<svg width="{width}" height="{height}"></svg>
<script>
const data = {data_json};

const svg = d3.select("svg");
const tooltip = d3.select("body").append("div").attr("class", "tooltip");
const width = +svg.attr("width");
const height = +svg.attr("height");

const simulation = d3.forceSimulation(data.nodes)
  .force("link", d3.forceLink(data.links).id(d => d.id).distance(120))
  .force("charge", d3.forceManyBody().strength(-250))
  .force("center", d3.forceCenter(width / 2, height / 2));

const container = svg.append("g");

const link = container.append("g")
  .selectAll("line")
  .data(data.links)
  .join("line")
  .attr("class", "link");

const node = container.append("g")
  .selectAll("circle")
  .data(data.nodes)
  .join("circle")
  .attr("r", 12)
  .attr("fill", d => d.color)
  .on("mouseover", (event, d) => {{
    tooltip.style("opacity", 1).html(d.title || d.label);
  }})
  .on("mousemove", (event) => {{
    tooltip
      .style("left", (event.pageX + 12) + "px")
      .style("top", (event.pageY + 12) + "px");
  }})
  .on("mouseout", () => {{
    tooltip.style("opacity", 0);
  }})
  .call(d3.drag()
    .on("start", dragstarted)
    .on("drag", dragged)
    .on("end", dragended));

const labels = container.append("g")
  .selectAll("text")
  .data(data.nodes)
  .join("text")
  .text(d => d.label)
  .attr("dx", 16)
  .attr("dy", 4)
  .attr("class", "node");

const edgeLabels = container.append("g")
  .selectAll("text")
  .data(data.links)
  .join("text")
  .text(d => d.label)
  .attr("class", "edge-label");

svg.call(d3.zoom().on("zoom", (event) => {{
  container.attr("transform", event.transform);
}}));

simulation.on("tick", () => {{
  link
    .attr("x1", d => d.source.x)
    .attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x)
    .attr("y2", d => d.target.y);

  node
    .attr("cx", d => d.x)
    .attr("cy", d => d.y);

  labels
    .attr("x", d => d.x)
    .attr("y", d => d.y);

  edgeLabels
    .attr("x", d => (d.source.x + d.target.x) / 2)
    .attr("y", d => (d.source.y + d.target.y) / 2);
}});

function dragstarted(event, d) {{
  if (!event.active) simulation.alphaTarget(0.3).restart();
  d.fx = d.x;
  d.fy = d.y;
}}

function dragged(event, d) {{
  d.fx = event.x;
  d.fy = event.y;
}}

function dragended(event, d) {{
  if (!event.active) simulation.alphaTarget(0);
  d.fx = null;
  d.fy = null;
}}
</script>
</body>
</html>
"""


def graph_to_cytoscape_html(
    graph,
    width: int = 960,
    height: int = 600,
    node_label: str = "id",
    label_attr: str | None = None,
    label_fn: Callable[[Any, dict[str, Any]], str] | None = None,
    color_by_label: bool = True,
    palette: list[str] | None = None,
    layout: str = "cose",
    layout_options: dict[str, Any] | None = None,
    node_size: int = 22,
    font_size: int = 9,
    edge_font_size: int = 8,
    show_edge_labels: bool = True,
    max_label_width: int = 80,
) -> str:
    """Serialize a NetworkX graph into a Cytoscape.js HTML document."""
    palette = palette or _DEFAULT_COLORS
    label_colors: dict[str, str] = {}

    def pick_label_color(labels: list[str]) -> str:
        if not color_by_label or not labels:
            return _DEFAULT_COLORS[0]
        label = labels[0]
        if label not in label_colors:
            label_colors[label] = palette[len(label_colors) % len(palette)]
        return label_colors[label]

    elements = []
    for node_id, attrs in graph.nodes(data=True):
        labels = attrs.get("labels", [])
        elements.append(
            {
                "data": {
                    "id": str(node_id),
                    "label": _resolve_label(node_id, attrs, node_label, label_attr, label_fn),
                    "group": labels[0] if labels else "Node",
                    "color": pick_label_color(labels),
                    "title": attrs.get("properties", {}).get("name", str(node_id)),
                }
            }
        )

    for source, target, key, attrs in graph.edges(keys=True, data=True):
        elements.append(
            {
                "data": {
                    "id": f"{source}-{target}-{key}",
                    "source": str(source),
                    "target": str(target),
                    "label": attrs.get("type", "RELATED_TO"),
                }
            }
        )

    data_json = json.dumps(elements)
    layout_defaults = {"name": layout, "padding": 30}
    if layout_options:
        layout_defaults.update(layout_options)
    layout_json = json.dumps(layout_defaults)

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Grafito Cytoscape Graph</title>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; }}
    #cy {{ width: 100vw; height: 100vh; background: #f8f9fb; }}
  </style>
  <script src="https://unpkg.com/cytoscape@3.26.0/dist/cytoscape.min.js"></script>
</head>
<body>
<div id="cy"></div>
<script>
const elements = {data_json};

const cy = cytoscape({{
  container: document.getElementById('cy'),
  elements: elements,
  style: [
    {{
      selector: 'node',
      style: {{
        'label': 'data(label)',
        'background-color': 'data(color)',
        'text-valign': 'center',
        'text-halign': 'center',
        'color': '#1b1f23',
        'font-size': {font_size},
        'width': {node_size},
        'height': {node_size},
        'text-wrap': 'wrap',
        'text-max-width': {max_label_width}
      }}
    }},
    {{
      selector: 'edge',
      style: {{
        'curve-style': 'bezier',
        'target-arrow-shape': 'triangle',
        'line-color': '#9aa4b2',
        'target-arrow-color': '#9aa4b2',
        'label': {("'data(label)'" if show_edge_labels else "''")},
        'font-size': {edge_font_size},
        'color': '#556'
      }}
    }}
  ],
  layout: {layout_json}
}});
</script>
</body>
</html>
"""


def graph_to_netgraph(
    graph,
    # Labels (consistent with other backends)
    node_label: str = "id",
    label_attr: str | None = None,
    label_fn: Callable[[Any, dict[str, Any]], str] | None = None,
    # Colors
    color_by_label: bool = True,
    palette: list[str] | None = None,
    default_color: str = "#8ecae6",
    node_color_attr: str | None = None,
    color_map: dict[str, str] | None = None,
    # Netgraph specifics
    interactive: bool = False,
    node_layout: str = "spring",
    edge_layout: str = "straight",
    node_size: float = 3.0,
    edge_width: float = 1.0,
    arrows: bool = True,
    # Conversion
    edge_merge_strategy: str = "concat",
    # Matplotlib
    ax: Any = None,
    figsize: tuple[float, float] = (12, 8),
    **kwargs: Any,
):
    """Convert a NetworkX graph into a netgraph visualization.

    Args:
        graph: A NetworkX graph (MultiDiGraph will be converted to DiGraph).
        node_label: How to label nodes: "id", "name", "labels", or "label_and_name".
        label_attr: Specific attribute to use as label.
        label_fn: Custom function to compute labels.
        color_by_label: Assign colors based on node labels.
        palette: List of colors for color_by_label.
        default_color: Default node color.
        node_color_attr: Node property to use for color.
        color_map: Map from label names to colors.
        interactive: Enable interactive mode (drag nodes).
        node_layout: Layout algorithm: "spring", "circular", "random", etc.
        edge_layout: Edge routing: "straight", "curved", "arc", "bundled".
        node_size: Size of nodes.
        edge_width: Width of edges.
        arrows: Show directional arrows.
        edge_merge_strategy: How to merge multi-edges: "concat", "first", "count".
        ax: Matplotlib axes to draw on.
        figsize: Figure size if creating new figure.
        **kwargs: Additional arguments passed to netgraph.

    Returns:
        If ax is provided: netgraph instance.
        If ax is None: tuple of (fig, ax, netgraph_instance).
    """
    try:
        import netgraph
    except ImportError as exc:
        raise ImportError(
            "netgraph is not installed. Install with `pip install grafito[netgraph]` "
            "or `uv pip install grafito[netgraph]`."
        ) from exc

    import matplotlib.pyplot as plt
    import networkx as nx

    # Convert MultiDiGraph to DiGraph if needed
    if isinstance(graph, nx.MultiDiGraph):
        graph = _multidigraph_to_digraph(graph, merge_strategy=edge_merge_strategy)
    elif isinstance(graph, nx.MultiGraph):
        simple = nx.Graph()
        for node_id, attrs in graph.nodes(data=True):
            simple.add_node(node_id, **attrs)
        edge_data: dict[tuple, list[dict]] = {}
        for u, v, key, attrs in graph.edges(keys=True, data=True):
            edge_key = tuple(sorted([u, v]))
            if edge_key not in edge_data:
                edge_data[edge_key] = []
            edge_data[edge_key].append(attrs)
        for (u, v), attrs_list in edge_data.items():
            if edge_merge_strategy == "first":
                merged = attrs_list[0].copy()
            elif edge_merge_strategy == "count":
                merged = attrs_list[0].copy()
                merged["_edge_count"] = len(attrs_list)
            else:
                types = [a.get("type", "RELATED_TO") for a in attrs_list]
                merged = attrs_list[0].copy()
                merged["type"] = " | ".join(types)
            simple.add_edge(u, v, **merged)
        graph = simple

    palette = palette or _DEFAULT_COLORS
    label_colors: dict[str, str] = {}

    def pick_label_color(labels: list[str]) -> str:
        if not labels:
            return default_color
        for lbl in labels:
            if color_map and lbl in color_map:
                return color_map[lbl]
        if not color_by_label:
            return default_color
        lbl = labels[0]
        if lbl not in label_colors:
            label_colors[lbl] = palette[len(label_colors) % len(palette)]
        return label_colors[lbl]

    # Build node labels and colors
    node_labels = {}
    node_colors = {}
    for node_id, attrs in graph.nodes(data=True):
        node_labels[node_id] = _resolve_label(node_id, attrs, node_label, label_attr, label_fn)
        labels = attrs.get("labels", [])
        if node_color_attr:
            props = attrs.get("properties", {})
            node_colors[node_id] = props.get(node_color_attr) or attrs.get(node_color_attr) or default_color
        else:
            node_colors[node_id] = pick_label_color(labels)

    # Build edge labels
    edge_labels = {}
    if isinstance(graph, nx.DiGraph):
        for source, target, attrs in graph.edges(data=True):
            edge_labels[(source, target)] = attrs.get("type", "RELATED_TO")
    else:
        for u, v, attrs in graph.edges(data=True):
            edge_labels[(u, v)] = attrs.get("type", "RELATED_TO")

    # Create figure if not provided
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    # Choose netgraph class
    if interactive:
        NetgraphClass = netgraph.EditableGraph
    else:
        NetgraphClass = netgraph.Graph

    # Create netgraph instance
    ng = NetgraphClass(
        graph,
        node_labels=node_labels,
        node_color=node_colors,
        edge_labels=edge_labels,
        node_layout=node_layout,
        edge_layout=edge_layout,
        node_size=node_size,
        edge_width=edge_width,
        arrows=arrows,
        ax=ax,
        **kwargs,
    )

    if created_fig:
        return fig, ax, ng
    return ng


def save_pyvis_html(
    graph,
    path: str = "grafito_graph.html",
    notebook: bool = False,
    directed: bool = True,
    physics: str | dict[str, Any] | None = None,
    **kwargs: Any,
) -> str:
    """Render a NetworkX graph to a PyVis HTML file."""
    if "cdn_resources" not in kwargs:
        kwargs["cdn_resources"] = "in_line"
    net = to_pyvis(graph, notebook=notebook, directed=directed, **kwargs)
    if isinstance(physics, dict):
        net.set_options(json.dumps(physics))
    elif isinstance(physics, str):
        if physics in _PHYSICS_PRESETS:
            net.set_options(json.dumps(_PHYSICS_PRESETS[physics]))
        elif physics == "barnes_hut":
            net.barnes_hut()
        elif physics == "repulsion":
            net.repulsion()
        elif physics == "force_atlas_2based":
            net.force_atlas_2based()
    html = net.generate_html(notebook=notebook)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(html)
    return path
