"""Visualization helpers (PyVis)."""

from __future__ import annotations

from typing import Any, Callable, Protocol
import json
import shutil
import subprocess

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

    def resolve_label(node_id: Any, attrs: dict[str, Any]) -> str:
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
            label=resolve_label(node_id, attrs),
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

_VIZ_BACKENDS: dict[str, VizBackend] = {
    "pyvis": PyVisBackend(),
    "d2": D2Backend(),
    "mermaid": MermaidBackend(),
    "graphviz": GraphvizBackend(),
    "d3": D3Backend(),
    "cytoscape": CytoscapeBackend(),
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

    def resolve_label(node_id: Any, attrs: dict[str, Any]) -> str:
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

    for node_id, attrs in graph.nodes(data=True):
        node_key = _safe_d2_id(node_id)
        label = resolve_label(node_id, attrs)
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

    def resolve_label(node_id: Any, attrs: dict[str, Any]) -> str:
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

    def mermaid_id(value: Any) -> str:
        text = str(value)
        safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text)
        return f"n_{safe}" if safe and safe[0].isdigit() else safe or "n"

    node_ids: dict[Any, str] = {}
    for node_id, attrs in graph.nodes(data=True):
        m_id = mermaid_id(node_id)
        node_ids[node_id] = m_id
        label = resolve_label(node_id, attrs)
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

    def resolve_label(node_id: Any, attrs: dict[str, Any]) -> str:
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

    for node_id, attrs in graph.nodes(data=True):
        label = resolve_label(node_id, attrs).replace("\"", "\\\"")
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

    def resolve_label(node_id: Any, attrs: dict[str, Any]) -> str:
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
                "label": resolve_label(node_id, attrs),
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

    def resolve_label(node_id: Any, attrs: dict[str, Any]) -> str:
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
                    "label": resolve_label(node_id, attrs),
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
