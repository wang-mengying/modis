import json
import logging
import math
import pickle
import random
import time

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import si_direct as single
import divmodis_solo as solo

# Data = "../Dataset/HuggingFace/"
# Data = "../Dataset/OpenData/House/"
Data = "../Dataset/Mental/"
logging.basicConfig(filename=Data+'log.txt', level=logging.INFO, format='%(message)s')


def get_pivot(pareto, n, c_min, b_max, c_max, b_min, r, a=0.5):
    sp = []

    node_poss = list(pareto.keys())
    if len(node_poss) <= n:
        sp = node_poss
        return sp

    random.shuffle(node_poss)
    sp = node_poss[:n]

    improved = True
    while improved:
        improved = False
        for v in sp:
            for u in node_poss:
                if u in sp:
                    continue
                # swap v and u
                sp_temp = sp.copy()
                # print(sp_temp)
                # print(v)
                sp_temp.remove(v)
                sp_temp.append(u)

                set1 = [node_pos for node_pos in sp]
                set2 = [node_pos for node_pos in sp_temp]

                distance1 = average_distance(set1, pareto, c_min, b_max, c_max, b_min, r, a)
                distance2 = average_distance(set2, pareto, c_min, b_max, c_max, b_min, r, a)

                if distance2 > distance1:
                    sp = sp_temp
                    improved = True
                    break

    return sp


def get_pivot_cost_only(pareto, n, c_min, c_max, r, a=0.5):
    sp = []

    node_poss = list(pareto.keys())
    if len(node_poss) <= n:
        sp = node_poss
        return sp

    random.shuffle(node_poss)
    sp = node_poss[:n]

    improved = True
    while improved:
        improved = False
        for v in sp:
            for u in node_poss:
                if u in sp:
                    continue
                sp_temp = sp.copy()
                sp_temp.remove(v)
                sp_temp.append(u)

                set1 = [node_pos for node_pos in sp]
                set2 = [node_pos for node_pos in sp_temp]

                distance1 = average_distance_cost_only(set1, pareto, c_min, c_max, r, a)
                distance2 = average_distance_cost_only(set2, pareto, c_min, c_max, r, a)

                if distance2 > distance1:
                    sp = sp_temp
                    improved = True
                    break

    return sp


def pos(c, b, r, c_min, b_max=0):
    pos_q = []

    # Costs
    for i in range(len(c)):
        pos_q.append(math.floor(math.log(c[i] / c_min[i], r[i])))

    # Benefits
    for i in range(len(b)):
        pos_q.append(math.floor(math.log(b[i] / b_max[i], r[i + len(c)])))

    return tuple(pos_q)


def pos_cost_only(c, r, c_min):
    pos_q = []

    c =[0.00001 if x <= 0 else x for x in c]
    c_min =[0.0001 if x <= 0 else x for x in c_min]
    # Costs
    for i in range(len(c)):
        pos_q.append(math.floor(math.log(c[i] / c_min[i], r[i])))

    return tuple(pos_q)


def pairwise_distance(u, v, pareto, euclidean_max, a):
    u_pos, v_pos = eval(u), eval(v)

    u_label = pareto[u]["nodes"][-1][0] + pareto[u]["nodes"][-1][1]
    v_label = pareto[v]["nodes"][-1][0] + pareto[v]["nodes"][-1][1]

    euclidean = np.linalg.norm(np.array(u_pos) - np.array(v_pos))
    euclidean_nor = euclidean / euclidean_max

    cosine = (1 - cosine_similarity([u_label, v_label])[0][0]) / 2

    distance = a * euclidean_nor + (1 - a) * cosine

    return distance


def average_distance(nodes, pareto, c_min, b_max, c_max, b_min, r, a=0.5):
    pos_min = pos(c_min, b_max, r, c_min, b_max)
    pos_max = pos(c_max, b_min, r, c_min, b_max)
    euclidean_max = np.linalg.norm(np.array(pos_min) - np.array(pos_max))

    total_distance = sum(pairwise_distance(u, v, pareto, euclidean_max, a)
                         for i, u in enumerate(nodes)
                         for j, v in enumerate(nodes)
                         if i < j)

    # Calculate average
    num_pairs = len(nodes) * (len(nodes) - 1) / 2
    avg_distance = total_distance / num_pairs

    return avg_distance


def average_distance_cost_only(nodes, pareto, c_min,c_max, r, a=0.5):
    pos_min = pos_cost_only(c_min, r, c_min)
    pos_max = pos_cost_only(c_max, r, c_min)
    euclidean_max = np.linalg.norm(np.array(pos_min) - np.array(pos_max))

    total_distance = sum(pairwise_distance(u, v, pareto, euclidean_max, a)
                         for i, u in enumerate(nodes)
                         for j, v in enumerate(nodes)
                         if i < j)

    # Calculate average
    num_pairs = len(nodes) * (len(nodes) - 1) / 2
    avg_distance = total_distance / num_pairs

    return avg_distance


def main():
    length = 6
    epsilon = 0.02
    algorithm = "no"
    size = 6
    cost_only = False
    r = [1 - epsilon, 1 - epsilon, 1 - epsilon, 1 - epsilon, 1 - epsilon, 1 - epsilon]
    # r = [1 + epsilon, 1 - epsilon, 1 - epsilon, 1 - epsilon, 1 - epsilon]
    # dataset = Data + "0613/"
    dataset = Data + "results/ml" + str(length) + "/"
    dataset = dataset.replace('/', '\\')
    logging.info(f"epsilon: {epsilon}")
    logging.info(f"max_length: {length}")
    logging.info(f"cardinality: {size}")

    nodes_file = dataset + algorithm + str(epsilon) + '.json'
    with open(nodes_file, "r") as file:
        pareto = json.load(file)

    G = pickle.load(open(Data + '/results/ml6/costs.gpickle', 'rb'))
    # G = pickle.load(open(dataset + 'costs.gpickle', 'rb'))
    if "HuggingFace" in dataset:
        cost_only = True

    if not cost_only:
        c_min, b_max = single.get_cmin_bmax(G)
        c_max, b_min = solo.get_cmax_bmin(G)
        print("Getting pivot set...")
        start = time.time()
        pivot = get_pivot(pareto, size, c_min, b_max, c_max, b_min, r)
        end = time.time()
        logging.info(f"Number of pivot nodes: {len(pivot)}")
        logging.info(f"DivMODis search time: {end - start}\n")
    else:
        c_min, c_max = solo.get_cmax_min(G)
        print("Getting pivot set...")
        start = time.time()
        r = [1 + epsilon, 1 + epsilon, 1 + epsilon]
        pivot = get_pivot_cost_only(pareto, size, c_min, c_max, r)
        end = time.time()
        logging.info(f"Number of pivot nodes: {len(pivot)}")
        logging.info(f"DivMODis search time: {end - start}\n")

    print("Outputting pivot")
    pivot_nodes = {node_pos: pareto[node_pos] for node_pos in pivot}
    pivot_json = json.dumps(pivot_nodes, indent=4)
    with open(dataset + 'div' + str(epsilon) + '.json', 'w') as json_file:
        json_file.write(pivot_json)


if __name__ == '__main__':
    main()


