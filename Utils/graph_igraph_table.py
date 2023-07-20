from igraph import Graph
from itertools import combinations
import math
import csv
import time


def generate_graph(L, k):
    n = len(L)
    graph = Graph(directed=True)
    state_to_id = {}

    def get_vertex_id(state):
        if state not in state_to_id:
            id = len(state_to_id)
            state_to_id[state] = id
            graph.add_vertex(id, label=str(state))
        return state_to_id[state]

    source_node = (tuple(1 for _ in L), tuple(1 for _ in range(k)))
    get_vertex_id(source_node)

    queue = [source_node]
    while queue:
        current_node = queue.pop(0)
        current_node_id = state_to_id[current_node]
        items_status, values_status = current_node
        # "drop": deactivate an active item, remain at least 70% items as active
        if sum(items_status) > 0.7 * n:
            for i in range(n):
                if items_status[i] == 1:
                    new_items_status = list(items_status)
                    new_items_status[i] = 0
                    new_node = (tuple(new_items_status), values_status)
                    new_id = get_vertex_id(new_node)
                    # print(new_id)
                    if not graph.are_connected(current_node_id, new_id):
                        edge = graph.add_edge(current_node_id, new_id)
                        graph.es[edge.index]["type"] = "drop"
                        graph.es[edge.index]["values"] = [i]
                        queue.append(new_node)

                # "modify": try dropping each value of the state, remain at least ceil(0.7 * k) values
                for j in range(1, k - math.ceil(0.7 * k) + 1):
                    for subset in combinations(range(k), j):
                        new_values_status = list(values_status)
                        for value in subset:
                            new_values_status[value] = 0
                        if sum(new_values_status) >= math.ceil(0.7 * k):
                            new_node = (items_status, tuple(new_values_status))
                            new_id = get_vertex_id(new_node)
                            if not graph.are_connected(current_node_id,new_id):
                                edge = graph.add_edge(current_node_id, new_id)
                                graph.es[edge.index]["type"] = "modify"
                                graph.es[edge.index]["values"] = list(subset)
                                queue.append(new_node)
    return graph, state_to_id


# L = [5, 5, 5, 5, 8]
# k = 10
# L = [2, 3]
# k = 5
start = time.time()
graph, state_to_id = generate_graph(L, k)
end = time.time()
print(end - start)
dataset = "..\\Dataset\\Movie\\others\\"


# Nodes
with open(dataset + 'nodes.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id', 'Label'])
    for v in graph.vs:
        writer.writerow([v.index, v['label']])

# Edges
with open(dataset + 'edges.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Source', 'Target', 'Type', 'Values'])
    for e in graph.es:
        writer.writerow([e.source, e.target, e['type'], e['values'] if 'values' in e.attributes() else []])


