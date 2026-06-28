"""Matplotlib visualization example for Grafito."""

from grafito import GrafitoDatabase
from grafito.integrations import plot_matplotlib, save_matplotlib, export_graph

# Create in-memory database
db = GrafitoDatabase(':memory:')

# Create person nodes with different groups
alice = db.create_node(
    labels=['Person'], 
    properties={'name': 'Alice', 'group': 'Engineering', 'score': 95}
)
bob = db.create_node(
    labels=['Person'], 
    properties={'name': 'Bob', 'group': 'Marketing', 'score': 78}
)
charlie = db.create_node(
    labels=['Person'], 
    properties={'name': 'Charlie', 'group': 'Engineering', 'score': 88}
)
diana = db.create_node(
    labels=['Person'], 
    properties={'name': 'Diana', 'group': 'Sales', 'score': 82}
)

# Create a company node
techcorp = db.create_node(
    labels=['Company'], 
    properties={'name': 'TechCorp', 'industry': 'Technology'}
)

# Create relationships
db.create_relationship(alice.id, bob.id, 'KNOWS', properties={'since': 2020})
db.create_relationship(bob.id, charlie.id, 'KNOWS', properties={'since': 2021})
db.create_relationship(charlie.id, diana.id, 'KNOWS', properties={'since': 2022})
db.create_relationship(alice.id, charlie.id, 'KNOWS', properties={'since': 2019})
db.create_relationship(bob.id, diana.id, 'KNOWS', properties={'since': 2020})
db.create_relationship(diana.id, alice.id, 'KNOWS', properties={'since': 2023})
db.create_relationship(alice.id, techcorp.id, 'WORKS_AT')
db.create_relationship(bob.id, techcorp.id, 'WORKS_AT')
db.create_relationship(charlie.id, techcorp.id, 'WORKS_AT')

# Export to NetworkX
graph = db.to_networkx()

print(f"Graph created: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
print()

# ============================================================
# EXAMPLE 1: Basic visualization
# ============================================================
print("Generating basic visualization...")
save_matplotlib(
    graph,
    'examples/visualization/output_matplotlib_basic.png',
    title="Basic Social Network",
    color_by_label=True,
    node_label='name',
    figsize=(10, 8),
    label_offset="auto",
    show_edge_labels=True
)
print("OK - Saved: examples/visualization/output_matplotlib_basic.png")

# ============================================================
# EXAMPLE 2: Color by group property
# ============================================================
print("\nGenerating visualization by group...")
save_matplotlib(
    graph,
    'examples/visualization/output_matplotlib_by_group.png',
    title="Colored by Group",
    node_label='name',
    color_by_label=False,  # Don't color by label
    color_attr='group',     # Color by 'group' property
    color_map={
        'Engineering': '#3498db',  # Blue
        'Marketing': '#e74c3c',    # Red
        'Sales': '#2ecc71'         # Green
    },
    node_size=1200,
    font_size=11,
    font_weight='bold',
    layout='spring',
    layout_kwargs={'k': 2, 'seed': 42},
    edge_width=2,
    edge_alpha=0.6,
    label_offset="auto",
    show_edge_labels=True
)
print("OK - Saved: examples/visualization/output_matplotlib_by_group.png")

# ============================================================
# EXAMPLE 3: Circular layout with custom styles
# ============================================================
print("\nGenerating circular visualization...")
save_matplotlib(
    graph,
    'examples/visualization/output_matplotlib_circular.png',
    title="Circular Layout",
    node_label='name',
    color_by_label=True,
    palette=['#9b59b6', '#f39c12'],  # Purple and orange
    node_size=1000,
    node_shape='s',  # Squares
    node_edge_color='#2c3e50',
    node_linewidth=2,
    layout='circular',
    font_size=10,
    font_color='#2c3e50',
    edge_color='#7f8c8d',
    edge_style='dashed',
    edge_width=1.5,
    show_legend=True,
    legend_loc='upper right',
    label_offset="auto",
    show_edge_labels=True
)
print("OK - Saved: examples/visualization/output_matplotlib_circular.png")

# ============================================================
# EXAMPLE 4: Node sizes based on property (score)
# ============================================================
print("\nGenerating visualization with dynamic sizes...")
fig = plot_matplotlib(
    graph,
    return_fig=True,  # Return figure for modification
    title="Nodes by Score (proportional size)",
    node_label='name',
    color_by_label=True,
    palette=['#1abc9c', '#e67e22'],
    node_size_attr='score',  # Size based on 'score' property
    node_alpha=0.85,
    node_edge_color='#34495e',
    node_linewidth=1.5,
    layout='kamada_kawai',
    font_size=10,
    font_weight='bold',
    edge_color='#95a5a6',
    edge_width=2,
    edge_alpha=0.5,
    label_offset="auto",
    show_edge_labels=True,
    show_legend=True,
    bgcolor='#ecf0f1'
)

# Add custom annotation
fig.axes[0].annotate(
    'Tech Lead',
    xy=(0.2, 0.8), xytext=(0.4, 0.9),
    arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5),
    fontsize=10, color='#e74c3c', fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#e74c4c', alpha=0.8)
)

fig.savefig('examples/visualization/output_matplotlib_sized.png', dpi=150, bbox_inches='tight')
print("OK - Saved: examples/visualization/output_matplotlib_sized.png")

# ============================================================
# EXAMPLE 5: Using the generic export_graph backend
# ============================================================
print("\nGenerating visualization via export_graph...")
export_graph(
    graph,
    'examples/visualization/output_matplotlib_via_export.png',
    backend='matplotlib',
    title="Via export_graph() API",
    node_label='name',
    color_by_label=True,
    layout='spring',
    node_size=800,
    font_size=10,
    label_offset="auto",
    show_edge_labels=True,
    dpi=120
)
print("OK - Saved: examples/visualization/output_matplotlib_via_export.png")

print("\n" + "="*60)
print("All visualizations generated successfully!")
print("="*60)
print("\nGenerated files:")
print("  1. output_matplotlib_basic.png - Basic visualization")
print("  2. output_matplotlib_by_group.png - Colored by group")
print("  3. output_matplotlib_circular.png - Circular layout")
print("  4. output_matplotlib_sized.png - Sizes by score")
print("  5. output_matplotlib_via_export.png - Via generic API")
