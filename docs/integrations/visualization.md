# Visualization

Grafito supports multiple visualization backends for exploring your graphs.

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

## Comparison

| Backend | Output | Interactive | Best For |
|---------|--------|-------------|----------|
| **PyVis** | HTML | ✅ Yes | Exploration, dashboards |
| **D2** | Text/SVG | ❌ No | Documentation, version control |
| **Mermaid** | Markdown/SVG | ⚠️ Partial | READMEs, docs integration |
| **Graphviz** | PNG/SVG/PDF | ❌ No | Publications, static diagrams |

## Backend Availability

```python
from grafito.integrations import available_viz_backends

print(available_viz_backends())
# ['pyvis', 'd2', 'mermaid', 'graphviz']
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
