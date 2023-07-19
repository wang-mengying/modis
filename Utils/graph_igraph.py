import igraph as ig
import pandas as pd

L = [2, 3]
G = ig.Graph(directed=True)
state_to_id = {}


# Get the vertex id
def get_vertex_id(state):
    if state not in state_to_id:
        id = len(state_to_id)
        state_to_id[state] = id
        G.add_vertex(id, label=str(state))
    return state_to_id[state]


# Add edges to the graph recursively
def add_edges(G, current_node):
    current_id = get_vertex_id(current_node)
    for i, item in enumerate(current_node):
        # "add": activate an inactive item, select all its values by default
        if item[0] == 0:
            new_node = list(current_node)
            new_node[i] = (1, tuple(range(L[i])))
            new_node = tuple(new_node)
            new_id = get_vertex_id(new_node)
            if not G.are_connected(current_id, new_id):
                G.add_edge(current_id, new_id, label=str((0, i, list(range(L[i])))))
                add_edges(G, new_node)
        # "modify": try dropping each value of an active item, remain at least one value
        elif item[0] == 1 and len(item[1]) > 1:
            for j in range(len(item[1])):
                new_node = list(current_node)
                new_node[i] = (1, tuple(val for k, val in enumerate(item[1]) if k != j))
                new_node = tuple(new_node)
                new_id = get_vertex_id(new_node)
                if not G.are_connected(current_id, new_id):
                    G.add_edge(current_id, new_id, label=str((1, i, list(new_node[i][1]))))
                    add_edges(G, new_node)


def export_csv(G, path="../Example/"):
    # Nodes
    node_df = pd.DataFrame({
        'Id': range(len(G.vs)),
        'Label': G.vs["label"]
    })
    node_df.to_csv(path + 'nodes.csv', index=False)

    # Edges
    edge_df = pd.DataFrame({
        'Source': [edge.source for edge in G.es],
        'Target': [edge.target for edge in G.es],
        'Label': G.es["label"],
        'Type': ['Add' if edge['label'][1] == '0' else 'Modify' for edge in G.es]
    })

    edge_df.to_csv(path + 'edges.csv', index=False)


def main():
    source_node = tuple((0, ()) for _ in L)
    add_edges(G, source_node)

    export_csv(G)


if __name__ == '__main__':
    main()
