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

## Matplotlib

Matplotlib provides static, publication-quality graph visualizations with extensive customization options.

### Installation

```bash
pip install matplotlib
```

### Basic Usage

```python
from grafito import GrafitoDatabase
from grafito.integrations import plot_matplotlib

# Create sample data
db = GrafitoDatabase(':memory:')
alice = db.create_node(labels=['Person'], properties={'name': 'Alice', 'group': 'A'})
bob = db.create_node(labels=['Person'], properties={'name': 'Bob', 'group': 'B'})
charlie = db.create_node(labels=['Person'], properties={'name': 'Charlie', 'group': 'A'})
db.create_relationship(alice.id, bob.id, 'KNOWS')
db.create_relationship(bob.id, charlie.id, 'KNOWS')

# Export to NetworkX
graph = db.to_networkx()

# Basic plot
plot_matplotlib(graph, title="Social Network")
```

### Save to File

```python
from grafito.integrations import save_matplotlib

# Save as PNG
save_matplotlib(graph, 'network.png', title="Social Network")

# Save as SVG for vector graphics
save_matplotlib(graph, 'network.svg', format='svg', figsize=(12, 10))
```

### Using the Generic Export API

```python
from grafito.integrations import export_graph

# Basic export
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

### Custom Edge Labels with Properties

Use `edge_label_fn` to show relationship properties on edges:

```python
def edge_label_with_props(source, target, attrs):
    """Show relationship type and properties."""
    rel_type = attrs.get("type", "RELATED_TO")
    props = attrs.get("properties", {})
    if props:
        props_str = "\n".join(f"{k}: {v}" for k, v in props.items())
        return f"{rel_type}\n{props_str}"
    return rel_type

export_graph(
    graph,
    'graph.png',
    backend='netgraph',
    label_fn=label_with_wrap,
    edge_label_fn=edge_label_with_props,
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
```

---

## Matplotlib Backend

Matplotlib provides static, publication-quality graph visualizations with extensive customization options.

### Installation

```bash
pip install matplotlib
```

### Basic Usage

```python
from grafito import GrafitoDatabase
from grafito.integrations import plot_matplotlib

# Create sample data
db = GrafitoDatabase(':memory:')
alice = db.create_node(labels=['Person'], properties={'name': 'Alice', 'group': 'A'})
bob = db.create_node(labels=['Person'], properties={'name': 'Bob', 'group': 'B'})
charlie = db.create_node(labels=['Person'], properties={'name': 'Charlie', 'group': 'A'})
db.create_relationship(alice.id, bob.id, 'KNOWS')
db.create_relationship(bob.id, charlie.id, 'KNOWS')

# Export to NetworkX
graph = db.to_networkx()

# Basic plot
plot_matplotlib(graph, title="Social Network")
```

### Save to File

```python
from grafito.integrations import save_matplotlib

# Save as PNG
save_matplotlib(graph, 'network.png', title="Social Network")

# Save as SVG for vector graphics
save_matplotlib(graph, 'network.svg', format='svg', figsize=(12, 10))
```

### Using the Generic Export API

```python
# Export using matplotlib backend
export_graph(graph, 'network.png', backend='matplotlib', title="My Graph")
```

### Custom Styling

#### Color by Labels

```python
plot_matplotlib(
    graph,
    color_by_label=True,           # Color nodes by their label
    palette=['#ff6b6b', '#4ecdc4'], # Custom color palette
    node_size=800,                 # Larger nodes
    title="Colored by Label"
)
```

#### Color by Property

```python
plot_matplotlib(
    graph,
    color_by=False,                # Disable auto-coloring
    color_attr='group',            # Color by 'group' property
    color_map={'A': '#ff6b6b', 'B': '#4ecdc4'},
    title="Colored by Group"
)
```

#### Node Sizes

```python
plot_matplotlib(
    graph,
    node_size=1000,                # Fixed size for all nodes
    # Or use a property for variable sizes:
    # node_size_attr='importance',
    node_shape='s',                # Square nodes
    node_alpha=0.8,
    title="Custom Node Styles"
)
```

#### Edge Styling

```python
plot_matplotlib(
    graph,
    edge_color='#888888',
    edge_width=2.0,
    edge_alpha=0.5,
    edge_style='dashed',
    title="Styled Edges"
)
```

### Layout Options

```python
# Available layouts: 'spring', 'circular', 'random', 'shell',
#                    'spectral', 'kamada_kawai', 'planar', 'fruchterman_reingold'

plot_matplotlib(graph, layout='circular', title="Circular Layout")

plot_matplotlib(
    graph,
    layout='spring',
    layout_kwargs={'k': 2, 'iterations': 50},  # Spring layout parameters
    title="Spring Layout (Customized)"
)

plot_matplotlib(graph, layout='kamada_kawai', title="Kamada-Kawai Layout")
```

### Labels and Fonts

```python
plot_matplotlib(
    graph,
    node_label='name',             # Use 'name' property as label
    font_size=12,
    font_color='#333333',
    font_weight='bold',
    label_offset="auto",           # Automatic positioning above nodes
    # Or use manual offset: label_offset=(0, 0.08)
    title="Custom Labels"
)
```

### Edge Labels with Properties

```python
def edge_label_with_props(source, target, key, attrs):
    """Show relationship type and properties."""
    rel_type = attrs.get("type", "RELATED_TO")
    props = attrs.get("properties", {})
    if props:
        return f"{rel_type}\n({', '.join(f'{k}: {v}' for k, v in props.items())})"
    return rel_type

plot_matplotlib(
    graph,
    show_edge_labels=True,
    edge_label_fn=edge_label_with_props,
    edge_font_size=8,
    title="Graph with Edge Properties"
)
```

### Complete Customization Example

```python
from grafito.integrations import plot_matplotlib

fig = plot_matplotlib(
    graph,
    # Figure
    figsize=(14, 12),
    dpi=150,
    bgcolor='#f8f9fa',
    # Layout
    layout='spring',
    layout_kwargs={'k': 1.5, 'seed': 42},
    # Nodes
    color_by_label=True,
    palette=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'],
    node_size=1200,
    node_shape='o',
    node_alpha=0.9,
    node_edge_color='#2c3e50',
    node_linewidth=2.0,
    # Edges
    edge_color='#7f8c8d',
    edge_width=2.0,
    edge_alpha=0.6,
    edge_arrow_size=20,
    # Labels
    node_label='name',
    font_size=11,
    font_color='#2c3e50',
    font_weight='bold',
    label_offset=(0, 0.06),
    # Legend
    show_legend=True,
    legend_loc='upper right',
    # Title
    title="Complete Customization Example",
    title_fontsize=18,
    title_fontweight='bold',
    # Return figure for further customization
    return_fig=True
)

# Further matplotlib customization
fig.axes[0].annotate(
    'Central Node',
    xy=(0.5, 0.5), xytext=(0.7, 0.8),
    arrowprops=dict(arrowstyle='->', color='red'),
    fontsize=10, color='red'
)

fig.savefig('custom_network.png', dpi=200, bbox_inches='tight')
```

### Advanced: Using the Backend System

```python
from grafito.integrations import render_graph, export_graph

# Render returns the matplotlib Figure
fig = render_graph(graph, backend='matplotlib', title="Rendered")

# Modify the figure before saving
fig.axes[0].set_xlabel("Custom X Label")
fig.savefig('modified.png')
```


## Comparison

| Backend | Output | Interactive | Best For |
|---------|--------|-------------|----------|
| **PyVis** | HTML | ✅ Yes | Exploration, dashboards |
| **Matplotlib** | PNG/SVG/PDF | ❌ No | Publications, static analysis |
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
# ['cytoscape', 'd2', 'd3', 'graphviz', 'matplotlib', 'mermaid', 'netgraph', 'pyvis']
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
