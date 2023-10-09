from igraph import Graph
import math
import csv
import time


def generate_graph(features, clusters, t_drop=0.7, t_modify=0.8, max_depth=2):
    graph = Graph(directed=True)
    state_to_id = {}
    depth = {}

    def get_vertex_id(state):
        if state not in state_to_id:
            id = len(state_to_id)
            state_to_id[state] = id
            graph.add_vertex(id, label=str(state))
        return state_to_id[state]

    source_node = (tuple(1 for _ in range(features)), tuple(1 for _ in range(clusters)))
    source_id = get_vertex_id(source_node)
    depth[source_id] = 0

    queue = [(source_node, 0)]
    while queue:
        current_node, current_depth = queue.pop(0)
        current_node_id = state_to_id[current_node]
        items_status, values_status = current_node
        if sum(items_status) > t_drop * features and current_depth < max_depth:
            # "modify": try dropping each value of the state, remain at least ceil(t_modify * k) values
            for j in range(clusters):
                new_values_status = list(values_status)
                if new_values_status[j] == 1:
                    new_values_status[j] = 0
                    if sum(new_values_status) >= math.ceil(t_modify * clusters):
                        new_node = (items_status, tuple(new_values_status))
                        new_id = get_vertex_id(new_node)
                        print(new_id)
                        if current_node_id == new_id:
                            continue
                        if not graph.are_connected(current_node_id, new_id):
                            edge = graph.add_edge(current_node_id, new_id)
                            graph.es[edge.index]["type"] = "modify"
                            graph.es[edge.index]["values"] = [j]
                            queue.append((new_node, current_depth + 1))

            # "drop": deactivate an active item, remain at least t_drop items as active
            for i in range(features):
                if items_status[i] == 1:
                    new_items_status = list(items_status)
                    new_items_status[i] = 0
                    new_node = (tuple(new_items_status), values_status)
                    new_id = get_vertex_id(new_node)
                    print(new_id)
                    if not graph.are_connected(current_node_id, new_id):
                        edge = graph.add_edge(current_node_id, new_id)
                        graph.es[edge.index]["type"] = "drop"
                        graph.es[edge.index]["values"] = [i]
                        queue.append((new_node, current_depth + 1))

    return graph, state_to_id


def main():
    features = 11
    clusters = 11
    start = time.time()
    graph, state_to_id = generate_graph(features, clusters)
    end = time.time()
    print(end - start)
    dataset = "../Dataset/Kaggle/results/ml2/"
    # dataset = "../Example/small/t_cluster/maxl/"

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


if __name__ == '__main__':
    main()


