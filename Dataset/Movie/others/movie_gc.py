import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def active_counts(label_str):
    label_tuple = eval(label_str)
    active_items = label_tuple[0].count(1)
    active_clusters = label_tuple[1].count(1)
    return active_items, active_clusters


nodes = pd.read_json('./d7m8/nodes.json')
nodes = nodes.transpose()

nodes['active_items'], nodes['active_clusters'] = zip(*nodes['Label'].apply(active_counts))
feature = nodes[['model_objectives', 'feature_objectives']]
nodes[['time', 'accuracy', 'complexity']] = pd.DataFrame(feature['model_objectives'].tolist(), index=nodes.index)
nodes[['fisher', 'mutual_info', 'vif']] = pd.DataFrame(feature['feature_objectives'].tolist(), index=nodes.index)


objectives = ['active_items', 'active_clusters', 'time', 'accuracy', 'complexity', 'fisher', 'mutual_info', 'vif']
correlation_matrix = nodes[objectives].corr()

G = nx.Graph()
for objective in objectives:
    G.add_node(objective)
for i, objective1 in enumerate(objectives):
    for j, objective2 in enumerate(objectives):
        if i < j:
            weight = correlation_matrix.loc[objective1, objective2]
            if weight != 1:
                G.add_edge(objective1, objective2, weight=weight)


edge_colors = ['blue' if G[u][v]['weight'] > 0 else 'red' for u, v in G.edges()]
pos = nx.circular_layout(G)


plt.figure(figsize=(12, 10))
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2500, font_size=15, edge_color=edge_colors)
labels = nx.get_edge_attributes(G, 'weight')
rounded_labels = {k: round(v, 2) for k, v in labels.items()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=rounded_labels, font_size=12, label_pos=0.4)
plt.title('Correlation Graph (GC)')


output_path_final = "correlation_graph.png"
plt.savefig(output_path_final, bbox_inches='tight')
plt.show()
