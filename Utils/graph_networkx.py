import networkx as nx
import matplotlib.pyplot as plt
import json

import warnings
warnings.filterwarnings("ignore")


L = [2, 3]
# path = "../Dataset/Movie/others/"
#
# with open(path + "cluster_k.json", 'r') as json_file:
#     data = json.load(json_file)
#
# features = list(data.keys())
# L = list(data.values())
#
# print("Keys:", features)
# print("Values:", L)


# Add nodes and edges to the graph recursively
def add_nodes_edges(G: nx.DiGraph, current_node):
    for i, item in enumerate(current_node):
        # "add": activate an inactive item, select all its values by default
        if item[0] == 0:
            new_node = list(current_node)
            new_node[i] = (1, tuple(range(L[i])))
            new_node = tuple(new_node)
            if new_node not in G:
                G.add_node(new_node)
                G.add_edge(current_node, new_node, label=(0, i, list(range(L[i]))))
                add_nodes_edges(G, new_node)
        # "modify": try dropping each value of an active item, remain at least one value
        elif item[0] == 1 and len(item[1]) > 1:
            for j in range(len(item[1])):
                new_node = list(current_node)
                new_node[i] = (1, tuple(val for k, val in enumerate(item[1]) if k != j))
                new_node = tuple(new_node)
                if new_node not in G:
                    G.add_node(new_node)
                    G.add_edge(current_node, new_node, label=(1, i, list(new_node[i][1])))
                    add_nodes_edges(G, new_node)


# Construct the graph
def construct(L: list):
    # initialize the graph
    G = nx.DiGraph()

    # add the source node
    source_node = tuple((0, ()) for _ in L)
    G.add_node(source_node)

    add_nodes_edges(G, source_node)

    return G, source_node


# Visualization of a graph
def visualize(G: nx.DiGraph, figsize=(16, 12), seed=40, node_size=2000):
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, seed=seed)
    nx.draw_networkx(G, pos, with_labels=True, node_size=node_size, node_color="skyblue")
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    return plt


def main():
    G, source_node = construct(L)

    # For Example-2_3
    nx.write_gpickle(G, "../Example/example-2_3.gpickle")
    # print(len(G.nodes))

    plt = visualize(G)
    plt.title("Schema: [[a0, a1], [b0, b1, b2]], Size(#nodes): 32, Full version", fontsize=25, pad=20)
    plt.savefig('../Example/example-2_3-Gf.png')

    bfs_nodes = list(nx.bfs_tree(G, source_node))
    Gs = G.subgraph(bfs_nodes[:8])

    plt = visualize(Gs, figsize=(12, 9))
    plt.title("Schema: [[a0, a1], [b0, b1, b2]], Size(#nodes): 32, Subgraph with 8 nodes", fontsize=18, pad=15)
    plt.savefig('../Example/example-2_3-G8.png')

    # nx.write_gpickle(G, path + "movie.gpickle")
    # print(len(G.nodes))
    #
    # plt = visualize(G)
    # plt.savefig(path + 'movie-Gf.png')
    #
    # bfs_nodes = list(nx.bfs_tree(G, source_node))
    # Gs = G.subgraph(bfs_nodes[:8])
    #
    # plt = visualize(Gs, figsize=(12, 9))
    # plt.savefig(path + 'movie-G8.png')


if __name__ == '__main__':
    main()
