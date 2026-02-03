# NetworkX Integration

GrafitoDB provides bidirectional integration with NetworkX for graph algorithms and analysis.

## Export to NetworkX

Convert a GrafitoDB database to a NetworkX `MultiDiGraph`.

### Basic Export

```python
from grafito import GrafitoDatabase

# Create database with data
db = GrafitoDatabase(':memory:')
alice = db.create_node(labels=['Person'], properties={'name': 'Alice', 'age': 30})
bob = db.create_node(labels=['Person'], properties={'name': 'Bob', 'age': 25})
db.create_relationship(alice.id, bob.id, 'KNOWS', {'since': 2020})

# Export to NetworkX
graph = db.to_networkx()

print(f'Nodes: {graph.number_of_nodes()}')  # 2
print(f'Edges: {graph.number_of_edges()}')  # 1
```

### Node Attributes

Node attributes in NetworkX include:
- `labels`: List of node labels
- `properties`: Dictionary of node properties
- Original node ID is stored as `grafito_id`

```python
# Access node attributes
for node_id, attrs in graph.nodes(data=True):
    print(f'Node {node_id}:')
    print(f'  Labels: {attrs["labels"]}')
    print(f'  Properties: {attrs["properties"]}')
```

### Edge Attributes

Edge attributes include:
- `type`: Relationship type
- `properties`: Dictionary of relationship properties

```python
# Access edge attributes
for u, v, key, attrs in graph.edges(data=True, keys=True):
    print(f'Edge {u} -> {v}:')
    print(f'  Type: {attrs["type"]}')
    print(f'  Properties: {attrs["properties"]}')
```

## Import from NetworkX

Import a NetworkX graph into GrafitoDB.

### Basic Import

```python
import networkx as nx
from grafito import GrafitoDatabase

# Create NetworkX graph
graph = nx.MultiDiGraph()
graph.add_node('alice', labels=['Person'], properties={'name': 'Alice'})
graph.add_node('bob', labels=['Person'], properties={'name': 'Bob'})
graph.add_edge('alice', 'bob', type='KNOWS', properties={'since': 2020})

# Import into Grafito
db = GrafitoDatabase(':memory:')
node_map = db.from_networkx(graph)

# node_map maps NetworkX node IDs to Grafito node IDs
print(f'Created {len(node_map)} nodes')
print(f'Alice ID: {node_map["alice"]}')
```

### Import with Custom Mapping

```python
# Import and then work with the data
node_map = db.from_networkx(graph)

# Query imported data
alice_id = node_map['alice']
neighbors = db.get_neighbors(alice_id, direction='outgoing')
for n in neighbors:
    print(f'Alice knows: {n.properties["name"]}')
```

## Using NetworkX Algorithms

Once exported, use any NetworkX algorithm.

### Centrality Analysis

```python
import networkx as nx

# Export graph
graph = db.to_networkx()

# Degree centrality
degree_cent = nx.degree_centrality(graph)
print('Most connected:', max(degree_cent, key=degree_cent.get))

# Betweenness centrality
betweenness = nx.betweenness_centrality(graph)
print('Key connectors:', sorted(betweenness.items(), key=lambda x: -x[1])[:5])

# PageRank
pagerank = nx.pagerank(graph)
print('Highest PageRank:', max(pagerank, key=pagerank.get))
```

### Path Finding

```python
# Shortest path
try:
    path = nx.shortest_path(graph, source=alice.id, target=bob.id)
    print(f'Path: {path}')
except nx.NetworkXNoPath:
    print('No path exists')

# All simple paths
paths = list(nx.all_simple_paths(graph, alice.id, bob.id, cutoff=3))
print(f'Found {len(paths)} paths')
```

### Community Detection

```python
# Weakly connected components
components = list(nx.weakly_connected_components(graph))
print(f'{len(components)} components')

# Communities using Louvain (undirected)
undirected = graph.to_undirected()
communities = nx.community.louvain_communities(undirected)
print(f'{len(communities)} communities')
```

### Cycle Detection

```python
# Find cycles
cycles = list(nx.simple_cycles(graph))
print(f'{len(cycles)} cycles found')

# Check if DAG
is_dag = nx.is_directed_acyclic_graph(graph)
print(f'Is DAG: {is_dag}')
```

## Data Transformation

### Converting to Undirected

```python
# For algorithms requiring undirected graphs
undirected = graph.to_undirected()

# Check connectivity
is_connected = nx.is_connected(undirected)
print(f'Connected: {is_connected}')
```

### Extracting Subgraphs

```python
# Subgraph by nodes
node_subset = list(graph.nodes())[:10]
subgraph = graph.subgraph(node_subset)

# Egocentric network (1-hop around a node)
egonet = nx.ego_graph(graph.to_undirected(), alice.id, radius=1)
print(f'Egonet has {egonet.number_of_nodes()} nodes')
```

## Roundtrip Example

```python
# Create in Grafito
db = GrafitoDatabase(':memory:')
alice = db.create_node(labels=['Person'], properties={'name': 'Alice'})
bob = db.create_node(labels=['Person'], properties={'name': 'Bob'})
db.create_relationship(alice.id, bob.id, 'KNOWS')

# Export to NetworkX
graph = db.to_networkx()

# Run algorithms
scores = nx.pagerank(graph)

# Enrich graph with scores
for node_id, score in scores.items():
    graph.nodes[node_id]['pagerank'] = score

# Import back to Grafito
db2 = GrafitoDatabase(':memory:')
node_map = db2.from_networkx(graph)

# Query enriched data
for old_id, new_id in node_map.items():
    node = db2.get_node(new_id)
    print(f"{node.properties['name']}: PageRank = {node.properties.get('pagerank')}")
```

## Best Practices

### 1. Memory Considerations

```python
# For large graphs, process in batches
batch_size = 1000
all_nodes = list(db.match_nodes())

for i in range(0, len(all_nodes), batch_size):
    batch = all_nodes[i:i+batch_size]
    # Process batch
```

### 2. Algorithm Selection

```python
# Use appropriate graph type
graph = db.to_networkx()

# Some algorithms need undirected
if not nx.is_directed_acyclic_graph(graph):
    print('Graph has cycles')

# Some need specific attributes
for u, v, attrs in graph.edges(data=True):
    if 'weight' not in attrs.get('properties', {}):
        # Add default weight
        attrs['properties']['weight'] = 1.0
```

### 3. Preserving Data Integrity

```python
# Node ID mapping is preserved
node_map = db.from_networkx(graph)

# Store original IDs if needed
for old_id, new_id in node_map.items():
    db.update_node_properties(new_id, {'original_id': old_id})
```

## Limitations

- NetworkX graphs are in-memory; large graphs may require significant RAM
- Some NetworkX algorithms require specific graph types (undirected, weighted)
- Relationship direction is preserved in `MultiDiGraph`
