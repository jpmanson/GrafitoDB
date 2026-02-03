# Visualization

GrafitoDB supports multiple visualization backends for exploring your graphs.

## PyVis (Interactive HTML)

PyVis generates interactive web-based visualizations.

### Installation

```bash
pip install grafito[viz]
# Or directly
pip install pyvis
```

### Basic Export

```python
from grafito import GrafitoDatabase
from grafito.integrations import save_pyvis_html

# Create sample data
db = GrafitoDatabase(':memory:')
alice = db.create_node(labels=['Person'], properties={'name': 'Alice', 'group': 'A'})
bob = db.create_node(labels=['Person'], properties={'name': 'Bob', 'group': 'B'})
charlie = db.create_node(labels=['Person'], properties={'name': 'Charlie', 'group': 'A'})
db.create_relationship(alice.id, bob.id, 'KNOWS')
db.create_relationship(bob.id, charlie.id, 'KNOWS')

# Export to NetworkX first
graph = db.to_networkx()

# Create PyVis visualization
save_pyvis_html(
    graph,
    path='graph.html',
    node_label='name',           # Property to use as label
    color_by_label=True,         # Color nodes by their labels
    physics='compact'            # Physics preset: 'compact' or 'spread'
)
```

### Physics Presets

| Preset | Best For |
|--------|----------|
| `compact` | Dense graphs, clusters |
| `spread`  | Sparse graphs, clear separation |

### Custom Styling

```python
from grafito.integrations import save_pyvis_html

# Group-based coloring
save_pyvis_html(
    graph,
    path='graph.html',
    node_label='name',
    color_by='group',           # Color by 'group' property
    physics='spread'
)
```

## Advanced PyVis Customization

If you need deeper control over node/edge styling, use PyVis directly after
exporting to NetworkX.

### Per-Type Node Styling

```python
from pyvis.network import Network

graph = db.to_networkx()
net = Network(height="650px", width="100%", bgcolor="#ffffff", font_color="black")

node_colors = {"ACTION": "#ffcc80", "OBSERVATION": "#e3f2fd"}

for node_id, attrs in graph.nodes(data=True):
    props = attrs.get("properties", attrs)
    ntype = props.get("type", "OBSERVATION")
    label = props.get("content", "")[:20] + "..."
    color = node_colors.get(ntype, "#e3f2fd")
    shape = "box" if ntype == "ACTION" else "ellipse"
    net.add_node(node_id, label=label, color=color, shape=shape)
```

### Per-Relationship Styling

```python
edge_colors = {"TEMPORAL_NEXT": "#2962ff", "SEMANTIC": "#00c853", "ENTITY": "#d50000"}

for source, target, attrs in graph.edges(data=True):
    rel_type = attrs.get("type")
    color = edge_colors.get(rel_type, "#aaaaaa")
    net.add_edge(source, target, color=color, label=rel_type, width=2)
```

### Physics Tuning

```python
net.barnes_hut(
    gravity=-4000,
    central_gravity=0.3,
    spring_length=250,
    spring_strength=0.05,
    damping=0.09,
)
```

### Add a Legend (Optional)

```python
from IPython.display import HTML, display

legend_html = """
<div style="font-family: sans-serif; margin-bottom: 10px; border: 1px solid #ddd; padding: 10px; border-radius: 5px; background: #f9f9f9;">
  <strong>Legend:</strong><br>
  <div style="display: flex; gap: 15px; flex-wrap: wrap; margin-top: 5px;">
    <div><span style="display:inline-block; width:12px; height:12px; background-color:#ffcc80; border:1px solid #333; margin-right:5px;"></span> Action (Box)</div>
    <div><span style="display:inline-block; width:12px; height:12px; background-color:#e3f2fd; border:1px solid #333; margin-right:5px; border-radius:50%;"></span> Observation (Ellipse)</div>
    <div><span style="color:#2962ff; font-weight:bold; margin-right:5px;">───</span> Temporal</div>
    <div><span style="color:#00c853; font-weight:bold; margin-right:5px;">───</span> Semantic</div>
    <div><span style="color:#d50000; font-weight:bold; margin-right:5px;">───</span> Entity</div>
  </div>
</div>
"""

display(HTML(legend_html))
```

## D2 (Declarative Diagramming)

D2 generates text-based diagrams that can be rendered to SVG/PNG.

### Installation

```bash
# Install D2 CLI separately
brew install d2        # macOS
# or
curl -fsSL https://d2.dev/install.sh | sh -s --
```

### Basic Export

```python
from grafito.integrations import export_graph

# Export to D2 format
graph = db.to_networkx()
export_graph(
    graph,
    'graph.d2',
    backend='d2',
    node_label='name'
)

# Content looks like:
# Alice: Alice
# Bob: Bob
# Alice -> Bob: KNOWS
```

### Render with D2

```python
# Export and render to SVG
export_graph(
    graph,
    'graph.d2',
    backend='d2',
    node_label='name',
    render='svg'        # Requires D2 CLI
)
# Generates graph.svg
```

Note: The D2 renderer is a separate CLI and is not bundled with Grafito.

## Mermaid

Mermaid is supported in Markdown and many documentation platforms.

### Basic Export

```python
from grafito.integrations import export_graph

# Export to Mermaid format
export_graph(
    graph,
    'graph.mmd',
    backend='mermaid',
    node_label='name'
)

# Content:
# graph TD
#     Alice[Alice]
#     Bob[Bob]
#     Alice -->|KNOWS| Bob
```

### Render

```python
# Render with mermaid-cli (requires npm install -g @mermaid-js/mermaid-cli)
export_graph(
    graph,
    'graph.mmd',
    backend='mermaid',
    node_label='name',
    render='svg'
)
```

Note: Mermaid rendering requires `mmdc` (`npm i -g @mermaid-js/mermaid-cli`).

## Graphviz (DOT)

Graphviz is the classic graph visualization tool.

### Installation

```bash
brew install graphviz    # macOS
apt-get install graphviz # Ubuntu
```

### Basic Export

```python
from grafito.integrations import export_graph

# Export to DOT format
export_graph(
    graph,
    'graph.dot',
    backend='graphviz',
    node_label='name'
)
```

### Render

```python
# Export and render
export_graph(
    graph,
    'graph.dot',
    backend='graphviz',
    node_label='name',
    render='svg'        # dot command must be in PATH
)
```

Note: Graphviz rendering requires the `dot` CLI (`brew install graphviz`).

## D3 (Self-Contained HTML)

D3 export produces a standalone HTML file (no build step).

```python
from grafito.integrations import export_graph

graph = db.to_networkx()
export_graph(
    graph,
    'graph.html',
    backend='d3',
    node_label='label_and_name'
)
```

## Cytoscape.js (Self-Contained HTML)

Cytoscape export produces a standalone HTML file (no build step).

```python
from grafito.integrations import export_graph

graph = db.to_networkx()
export_graph(
    graph,
    'graph.html',
    backend='cytoscape',
    node_label='label_and_name',
    layout='cose'
)
```

## Netgraph (Publication Quality)

Netgraph produces publication-quality static visualizations via matplotlib, with optional
interactive mode for dragging nodes.

### Installation

```bash
pip install grafito[netgraph]
# Or directly
pip install netgraph matplotlib
```

### Basic Export

```python
from grafito.integrations import export_graph

graph = db.to_networkx()

# Export to PNG
export_graph(
    graph,
    'graph.png',
    backend='netgraph',
    node_label='name',
    color_by_label=True
)
```

### Vector Formats (SVG/PDF)

Netgraph excels at producing vector graphics for publications:

```python
# SVG for web/docs
export_graph(graph, 'graph.svg', backend='netgraph', node_label='name')

# PDF for papers
export_graph(graph, 'graph.pdf', backend='netgraph', node_label='name', dpi=300)
```

### Custom Colors

```python
# Color map by label type
export_graph(
    graph,
    'graph.png',
    backend='netgraph',
    node_label='name',
    color_map={
        'Person': '#4ecdc4',
        'Company': '#ff6b6b',
        'City': '#ffe66d'
    }
)

# Custom palette for color_by_label
export_graph(
    graph,
    'graph.png',
    backend='netgraph',
    color_by_label=True,
    palette=['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
)
```

### Font Customization

```python
export_graph(
    graph,
    'graph.png',
    backend='netgraph',
    node_label='name',
    node_size=6,
    node_label_fontdict={'size': 14, 'fontweight': 'bold'},
    edge_label_fontdict={'size': 10}
)
```

### Custom Label Function with Word Wrap

```python
def label_with_wrap(node_id, attrs):
    """Two-line label: type on top, name below."""
    labels = attrs.get("labels", [])
    props = attrs.get("properties", {})
    name = props.get("name", str(node_id))
    label_type = labels[0] if labels else ""
    return f"{label_type}\n{name}" if label_type else name

export_graph(
    graph,
    'graph.png',
    backend='netgraph',
    label_fn=label_with_wrap,
    node_size=8
)
```

### Matplotlib Composition

Netgraph integrates with matplotlib for complex figures:

```python
import matplotlib.pyplot as plt
from grafito.integrations import graph_to_netgraph

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Different layouts side by side
graph_to_netgraph(graph, ax=axes[0], node_layout='spring', node_label='name')
axes[0].set_title('Spring Layout')

graph_to_netgraph(graph, ax=axes[1], node_layout='shell', node_label='name')
axes[1].set_title('Shell Layout')

plt.tight_layout()
plt.savefig('comparison.png', dpi=150)
```

### Interactive Mode

Enable interactive mode to drag nodes (requires a display):

```python
from grafito.integrations import graph_to_netgraph

fig, ax, ng = graph_to_netgraph(
    graph,
    interactive=True,
    node_label='name'
)
plt.show()
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `node_label` | Label mode: `"id"`, `"name"`, `"labels"`, `"label_and_name"` | `"id"` |
| `label_fn` | Custom function `(node_id, attrs) -> str` | `None` |
| `color_by_label` | Auto-color by node labels | `True` |
| `color_map` | Dict mapping label names to colors | `None` |
| `palette` | List of colors for `color_by_label` | default palette |
| `node_color_attr` | Node property to use for color | `None` |
| `node_layout` | Layout: `"spring"`, `"shell"`, `"random"` | `"spring"` |
| `node_size` | Size of nodes | `3.0` |
| `edge_width` | Width of edges | `1.0` |
| `arrows` | Show directional arrows | `True` |
| `figsize` | Figure size `(width, height)` | `(12, 8)` |
| `dpi` | Resolution for raster export | `150` |
| `interactive` | Enable node dragging | `False` |

## Comparison

| Backend | Output | Interactive | Best For |
|---------|--------|-------------|----------|
| **PyVis** | HTML | ✅ Yes | Exploration, dashboards |
| **D2** | Text/SVG | ❌ No | Documentation, version control |
| **Mermaid** | Markdown/SVG | ⚠️ Partial | READMEs, docs integration |
| **Graphviz** | PNG/SVG/PDF | ❌ No | Static diagrams |
| **D3** | HTML | ✅ Yes | Custom web views |
| **Cytoscape** | HTML | ✅ Yes | Large graphs, rich UI |
| **Netgraph** | PNG/SVG/PDF | ⚠️ Optional | Publications, matplotlib integration |

## Backend Availability

```python
from grafito.integrations import available_viz_backends

print(available_viz_backends())
# ['cytoscape', 'd2', 'd3', 'graphviz', 'mermaid', 'netgraph', 'pyvis']
```

## Large Graph Handling

For large graphs (>1000 nodes):

```python
# Sample before visualizing
all_nodes = list(graph.nodes())
sample_size = min(100, len(all_nodes))
sample_nodes = random.sample(all_nodes, sample_size)
subgraph = graph.subgraph(sample_nodes)

# Export sample
save_pyvis_html(subgraph, 'sample.html')
```

## Custom Visualization

Build custom visualizations using the data directly:

```python
import matplotlib.pyplot as plt
import networkx as nx

# Export
graph = db.to_networkx()

# Custom matplotlib plot
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(graph)
nx.draw(graph, pos, with_labels=True, node_color='lightblue')
plt.savefig('custom.png')
```
