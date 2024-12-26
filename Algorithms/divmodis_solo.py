import json
import math
import logging
import pickle
import random
import time

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import si_direct as single

logging.basicConfig(filename='../Dataset/Kaggle/log.txt', level=logging.INFO, format='%(message)s')


def is_good(node, constraints):
    """Determine if a node is good based on given constraints"""
    objectives = node['model_objectives'] + node['feature_objectives']
    good_statuses = []

    for idx, (op, value) in enumerate(constraints):
        if op == ">":
            good_statuses.append(objectives[idx] > value)
        elif op == "<":
            good_statuses.append(objectives[idx] < value)
    return good_statuses


def pos(m, f, r, c_min, b_max):
    """Calculate POS for a node"""
    c = [f[2], m[0], m[2]]
    b = [f[0], f[1], m[1]]
    pos_q = []

    # Costs
    for i in range(len(c)):
        pos_q.append(math.floor(math.log(c[i] / c_min[i], r[i])))

    # Benefits
    for i in range(len(b)):
        pos_q.append(math.floor(math.log(b[i] / b_max[i], r[i + len(c)])))

    pos = [pos_q[1], pos_q[5], pos_q[2], pos_q[3], pos_q[4], pos_q[0]]

    return tuple(pos)


def group(constraints, r, c_min, b_max, nodes):
    """Group nodes based on objectives"""
    grouped_nodes = {i: {} for i in range(len(r))}

    for node_id, node in nodes.items():
        good_statuses = is_good(node, constraints)
        if any(good_statuses):
            node_pos = pos(node['model_objectives'], node['feature_objectives'], r, c_min, b_max)

        for idx, status in enumerate(good_statuses):
            if status:
                grouped_nodes[idx][node_id] = {'pos': str(node_pos), 'label': node['Label'],
                                               'model_objectives': node['model_objectives'],
                                               'feature_objectives': node['feature_objectives']}

    return grouped_nodes


def get_cmax_bmin(G):
    """Get maximum costs and minimum benefits"""
    num_m = len(G.nodes[0]['model_objectives'])
    num_f = len(G.nodes[0]['feature_objectives'])

    model_objectives_mins = [min(G.nodes[node]['model_objectives'][i] for node in G.nodes()) for i in range(num_m)]
    feature_objectives_mins = [min(G.nodes[node]['feature_objectives'][i] for node in G.nodes()) for i in range(num_f)]

    model_objectives_maxs = [max(G.nodes[node]['model_objectives'][i] for node in G.nodes()) for i in range(num_m)]
    feature_objectives_maxs = [max(G.nodes[node]['feature_objectives'][i] for node in G.nodes()) for i in range(num_f)]

    # Movie
    # c_max = [feature_objectives_maxs[2], model_objectives_maxs[0], model_objectives_maxs[2]]
    # b_min = [feature_objectives_mins[0], feature_objectives_mins[1], model_objectives_mins[1]]

    # House
    # c_max = [model_objectives_maxs[2]]
    # b_min = [feature_objectives_mins[0], feature_objectives_mins[1], model_objectives_mins[1], model_objectives_mins[0]]

    # ModsNet
    c_max = []
    b_min = [model_objectives_maxs[0], model_objectives_maxs[1],  model_objectives_maxs[2],
              model_objectives_maxs[3],  model_objectives_maxs[4], model_objectives_maxs[5]]


    return c_max, b_min


def get_cmax_min(G):
    """Get maximum costs and minimum benefits"""
    model_objectives_mins = [min(G.nodes[node]['model_objectives'][i] for node in G.nodes()) for i in range(3)]
    model_objectives_maxs = [max(G.nodes[node]['model_objectives'][i] for node in G.nodes()) for i in range(3)]

    c_max = [model_objectives_maxs[0], model_objectives_maxs[1], model_objectives_maxs[2]]
    c_min = [model_objectives_mins[0], model_objectives_mins[1], model_objectives_mins[2]]

    return c_min, c_max


def pairwise_distance(u, v, euclidean_max, a):
    def convert_label(label_str):
        numbers = [int(num) for num in label_str.replace('(', '').replace(')', '').split(',')]
        return numbers

    u_pos = tuple(map(float, u['pos'][1:-1].split(',')))
    v_pos = tuple(map(float, v['pos'][1:-1].split(',')))

    u_label = convert_label(u['label'])
    v_label = convert_label(v['label'])

    euclidean = np.linalg.norm(np.array(u_pos) - np.array(v_pos))
    euclidean_nor = euclidean / euclidean_max

    cosine = (1 - cosine_similarity([u_label, v_label])[0][0]) / 2

    distance = a * euclidean_nor + (1 - a) * cosine

    return distance


def average_distance(nodes, c_min, b_max, c_max, b_min, r, a=0.5):
    pos_min = pos((c_min[1], b_max[2], c_min[2]), (b_max[0], b_max[1], c_min[0]), r, c_min, b_max)
    pos_max = pos((c_max[1], b_min[2], c_max[2]), (b_min[0], b_min[1], c_max[0]), r, c_min, b_max)
    euclidean_max = np.linalg.norm(np.array(pos_min) - np.array(pos_max))

    total_distance = sum(pairwise_distance(u, v, euclidean_max, a)
                         for i, u in enumerate(nodes)
                         for j, v in enumerate(nodes)
                         if i < j)

    # Calculate average
    num_pairs = len(nodes) * (len(nodes) - 1) / 2
    avg_distance = total_distance / num_pairs

    return avg_distance


def get_pivot(grouped_nodes, n, c_min, b_max, c_max, b_min, r, a=0.5):
    sp = {}  # pivot set

    for group, nodes in grouped_nodes.items():
        # 1. Initialization
        node_ids = list(nodes.keys())
        if len(node_ids) <= n:
            sp[group] = node_ids
            continue
        random.shuffle(node_ids)
        sp[group] = node_ids[:n]

        # 2. Iteration
        improved = True
        while improved:
            improved = False
            for v in sp[group]:
                for u in node_ids:
                    if u in sp[group]:
                        continue
                    # swap v and u
                    sp_temp = sp[group].copy()
                    # print(sp_temp)
                    # print(v)
                    sp_temp.remove(v)
                    sp_temp.append(u)

                    set1 = [nodes[node_id] for node_id in sp[group]]
                    set2 = [nodes[node_id] for node_id in sp_temp]

                    distance1 = average_distance(set1, c_min, b_max, c_max, b_min, r, a)
                    distance2 = average_distance(set2, c_min, b_max, c_max, b_min, r, a)

                    if distance2 > distance1:
                        sp[group] = sp_temp
                        improved = True
                        break

    return sp


def combine(nodes, pivot):
    pivot_list = list({item for sublist in pivot.values() for item in sublist})

    pivot_nodes = {node_id: nodes[node_id] for node_id in pivot_list}

    return pivot_nodes


def main():
    epsilon = 0.5
    max_length = 2
    logging.info(f"epsilon: {epsilon}")
    logging.info(f"max_length: {max_length}")
    r = [1 + epsilon, 1 + epsilon, 1 + epsilon, 1 - epsilon, 1 - epsilon, 1 - epsilon]

    dataset = "../Dataset/Kaggle/results/ml" + str(max_length) + "/"
    with open(dataset + "nodes.json", "r") as file:
        nodes = json.load(file)
    G = pickle.load(open(dataset + 'costs.gpickle', 'rb'))
    c_min, b_max = single.get_cmin_bmax(G)
    c_max, b_min = get_cmax_bmin(G)

    constraints = [["<", 1.0], [">", 0.88], ["<", 545], [">", 0.45], [">", 0.38], ["<", 1.3]]
    grouped_nodes = group(constraints, r, c_min, b_max, nodes)
    # grouped_json = json.dumps(grouped_nodes, indent=4)
    # with open(dataset + 'group' + str(epsilon) + '.json', 'w') as json_file:
    #     json_file.write(grouped_json)

    start = time.time()
    pivot = get_pivot(grouped_nodes, 5, c_min, b_max, c_max, b_min, r)
    end = time.time()
    logging.info(f"DivMODis search time: {end - start}")
    pivot_json = json.dumps(pivot, indent=4)
    with open(dataset + 'div' + str(epsilon) + '_group.json', 'w') as json_file:
        json_file.write(pivot_json)

    pivot_nodes = combine(nodes, pivot)
    logging.info(f"Number of pivot nodes: {len(pivot_nodes)}")
    pivot_nodes_json = json.dumps(pivot_nodes, indent=4)
    with open(dataset + 'div' + str(epsilon) + '_nodes.json', 'w') as json_file:
        json_file.write(pivot_nodes_json)


if __name__ == "__main__":
    main()
