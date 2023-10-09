import csv
from igraph import Graph
from itertools import combinations
import math
import json


def generate_graph(L):
    n = len(L)
    graph = Graph(directed=True)
    state_to_id = {}

    def get_vertex_id(state):
        if state not in state_to_id:
            id = len(state_to_id)
            state_to_id[state] = id
            graph.add_vertex(id, label=str(state))
        return state_to_id[state]

    source_node = tuple((1, tuple(range(t_i))) for t_i in L)
    get_vertex_id(source_node)

    queue = [source_node]

    while queue:
        current_node = queue.pop(0)
        current_node_id = state_to_id[current_node]
        for i, item in enumerate(current_node):
            if item[0] == 1:
                # "modify": modify values of an active item, drop at most math.ceil(L[i]/10) clusters
                t_row = max(1, L[i] - math.ceil(L[i] / 10))
                t_col = 0.7 * n
                if len(item[1]) > t_row:
                    for j in range(t_row, len(item[1])):
                        for subset in combinations(item[1], j):
                            new_node = list(current_node)
                            new_node[i] = (1, subset)
                            new_node = tuple(new_node)
                            new_id = get_vertex_id(new_node)
                            print(new_id)
                            if not graph.are_connected(current_node_id, new_id):
                                edge = graph.add_edge(current_node_id, new_id)
                                graph.es[edge.index]["label"] = (1, i, list(subset))
                                graph.es[edge.index]["type"] = "Modify"
                                queue.append(new_node)
                # "drop": deactivate an active item, only if more than t_col items are active
                if sum(x[0] for x in current_node) > t_col:
                    new_node = list(current_node)
                    new_node[i] = (0, ())
                    new_node = tuple(new_node)
                    new_id = get_vertex_id(new_node)
                    if not graph.are_connected(current_node_id, new_id):
                        edge = graph.add_edge(current_node_id, new_id)
                        graph.es[edge.index]["label"] = (0, i, [])
                        graph.es[edge.index]["type"] = "Drop"
                        queue.append(new_node)
    return graph, state_to_id


def export_to_csv(G, state_to_id, path):
    with open(path + 'nodes.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Label"])
        for state, id in state_to_id.items():
            writer.writerow([id, G.vs[id]["label"]])

    with open(path + 'edges.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Source", "Target", "Label", "Type"])
        for e in G.es:
            writer.writerow([G.vs[e.source].index, G.vs[e.target].index, e["label"], e["type"]])


def main():
    # L = [2, 3]
    L = [5, 5, 5, 5, 8]
    # dataset = "../Dataset/Kaggle/others/"
    #
    # with open(dataset + "cluster_k.json", 'r') as json_file:
    #     data = json.load(json_file)
    #
    # features = list(data.keys())
    # L = list(data.values())
    #
    # print("Keys:", features)
    # print("Values:", L)

    print('Constructing graph ......')
    G, state_to_id = generate_graph(L)

    print('Exporting graph into CSV files......')
    export_to_csv(G, state_to_id, '../Example/medium/drop/')


if __name__ == '__main__':
    main()

