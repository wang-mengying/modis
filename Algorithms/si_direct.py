import copy
import json
import logging
import math
import sys
import time
import pickle

import joblib
import networkx as nx
import pandas as pd
from collections import defaultdict
from multiprocessing import Pool

sys.path.append("../")
import Dataset.Kaggle.others.movie_objectives as movie_objectives

Data = "../Dataset/Kaggle/"
max_length = 5

dataset = Data + "results/ml/" + str(5)
logging.basicConfig(filename=Data+'log.txt', level=logging.INFO, format='%(message)s')
nodes_df = pd.read_csv(dataset + 'nodes.csv')
edges_df = pd.read_csv(dataset + 'edges.csv')

G = nx.from_pandas_edgelist(edges_df, 'Source', 'Target', edge_attr=['Type', 'Values'], create_using=nx.DiGraph())
labels_dict = nodes_df.set_index('Id')['Label'].to_dict()
nx.set_node_attributes(G, labels_dict, 'Label')

surrogate_model = None


def worker_initializer(model_path):
    global surrogate_model
    surrogate_model = joblib.load(model_path)


def get_objectives(node_id, model, cluster_file=Data+'others/movie_clustered_table.csv'):
    node = G.nodes[node_id]
    df = movie_objectives.surrogate_inputs(node, cluster_file)
    model_objectives = model.predict(df)[0]
    feature_objectives = movie_objectives.feature_objectives(node, cluster_file)

    return list(model_objectives), feature_objectives


def node_obj_map(node_id):
    print(node_id)
    return node_id, get_objectives(node_id, surrogate_model)


def nodes_objectives(G, model_path):
    nodes = list(G.nodes())

    num_workers = 8
    logging.info(f"Number of workers to calculate node objectives: {num_workers}")
    with Pool(num_workers, initializer=worker_initializer, initargs=(model_path,)) as pool:
        results = pool.map(node_obj_map, nodes)

    model_objectives = {node_id: model_objectives for node_id, (model_objectives, _) in results}
    feature_objectives = {node_id: feature_objectives for node_id, (_, feature_objectives) in results}
    nx.set_node_attributes(G, model_objectives, 'model_objectives')
    nx.set_node_attributes(G, feature_objectives, 'feature_objectives')


def cal_costs_benefits(args):
    edge, node_data = args
    print(edge)
    u, v = edge

    sm_objs, sf_objs = node_data[u]['model_objectives'], node_data[u]['feature_objectives']
    tm_objs, tf_objs = node_data[v]['model_objectives'], node_data[v]['feature_objectives']

    time = tm_objs[0] - sm_objs[0]
    accuracy = tm_objs[1] - sm_objs[1]
    complexity = tm_objs[2] - sm_objs[2]
    fisher = tf_objs[0] - sf_objs[0]
    mutual_info = tf_objs[1] - sf_objs[1]
    vif = tf_objs[2] - sf_objs[2]

    costs = [vif, time, complexity]
    benefits = [fisher, mutual_info, accuracy]

    return u, v, costs, benefits


def costs_benefits(G):
    edges = [(u, v) for u, v, _ in G.edges(data=True)]
    node_data = {node: G.nodes[node] for node in G.nodes()}

    num_workers = 8
    logging.info(f"Number of workers to calculate edge costs/benefits: {num_workers}")
    with Pool(num_workers) as pool:
        results = pool.map(cal_costs_benefits, [(edge, node_data) for edge in edges])

    for u, v, c, b in results:
        G[u][v]['c'] = c
        G[u][v]['b'] = b


def get_cmin_bmax(G):
    model_objectives_mins = [min(G.nodes[node]['model_objectives'][i] for node in G.nodes()) for i in range(3)]
    feature_objectives_mins = [min(G.nodes[node]['feature_objectives'][i] for node in G.nodes()) for i in range(3)]

    model_objectives_maxs = [max(G.nodes[node]['model_objectives'][i] for node in G.nodes()) for i in range(3)]
    feature_objectives_maxs = [max(G.nodes[node]['feature_objectives'][i] for node in G.nodes()) for i in range(3)]

    c_min = [feature_objectives_mins[2], model_objectives_mins[0], model_objectives_mins[2]]
    b_max = [feature_objectives_maxs[0], feature_objectives_maxs[1], model_objectives_maxs[1]]

    return c_min, b_max


def pos(q: tuple, r: list, c_min, b_max):
    pos_q = []

    # Costs
    for i in range(len(q[1])):
        pos_q.append(math.floor(math.log(q[1][i] / c_min[i], r[i])))

    # Benefits
    for i in range(len(q[2])-1):
        pos_q.append(math.floor(math.log(q[2][i] / b_max[i], r[i + len(q[1])])))

    return tuple(pos_q)


def get_pareto(G, s, r, t, c_min, b_max, max_length):
    """
    :param G: Graph
    :param s: Source node
    :param r: Approximate Ratios
    :param t: Thresholds on costs
    :param c_min: Minimum costs
    :param b_max: Maximum benefits
    :return: Pi
    """
    n = len(G)
    Pi = defaultdict(lambda: defaultdict(dict))
    c = [G.nodes[s]['feature_objectives'][2], G.nodes[s]['model_objectives'][0], G.nodes[s]['model_objectives'][2]]
    b = [G.nodes[s]['feature_objectives'][0], G.nodes[s]['feature_objectives'][1], G.nodes[s]['model_objectives'][1]]
    pos_s = pos((None, tuple(c), tuple(b), None, None), r, c_min, b_max)
    Pi[0][s][str(pos_s)] = (G.nodes[s]['Label'], tuple(c), tuple(b), None, None)
    for i in range(1, max_length+1):
        print(i)
        for v in G.nodes:
            Pi[i][v] = copy.deepcopy(Pi[i-1][v])
            for u in G.predecessors(v):
                # logging.info(f"v: {v}, u: {u}, i: {i}")
                Pi[i][v] = extend_and_merge(Pi, G, u, v, i, r, t, c_min, b_max)
    return Pi[max_length]


def extend_and_merge(Pi, G, u, v, i, r, t, c_min, b_max):
    Q = Pi[i - 1][u]
    e = G[u][v]
    # logging.info(f"Q: {Q}")
    for p in Q.values():
        # logging.info(f"p: {p}")
        # logging.info(f"e: {e}")
        q = (G.nodes[v]['Label'],
             tuple(p_c + e_c for p_c, e_c in zip(p[1], e['c'])),
             tuple(p_b + e_b for p_b, e_b in zip(p[2], e['b'])),
             p, e)
        # logging.info(f"q: {q}")
        # Guarantee cost under threshold
        if any(x > y for x, y in zip(q[1], t)):
            continue
        pos_q = pos(q, r, c_min, b_max)
        pos_q = str(pos_q)
        G.nodes[v]['pos'] = pos_q
        Pi[i] = merge(Pi[i], v, pos_q, q)
        # if pos_q not in Pi[i][v].keys() or Pi[i][v][pos_q][2][-1] < q[2][-1]:
        #     Pi[i][v][pos_q] = q
    # logging.info(f"R: {R}")
    return Pi[i][v]


def merge(D, node, pos_q, path):
    skip = False
    nodes = [k for k, v in D.items() if v != {}]
    for n in nodes:
        if pos_q in D[n].keys():
            if D[n][pos_q][2][-1] >= path[2][-1]:
                skip = True
                continue
            del D[n][pos_q]
    if not skip:
        D[node][pos_q] = path

    return D


def main():
    epsilon = 0.5
    logging.info(f"epsilon: {epsilon}")
    r = [1 + epsilon, 1 + epsilon, 1 + epsilon, 1 - epsilon, 1 - epsilon, 1]
    t = [5, 1.5, 555]

    # model_path = '../Surrogate/Movie/movie_surrogate.joblib'.replace("/", "\\")
    #
    # start = time.time()
    # logging.info("Nodes objectives...")
    # nodes_objectives(G, model_path)
    # pickle.dump(G, open(dataset + 'objectives.gpickle', 'wb'))
    # logging.info("Edges costs/benefits...")
    # costs_benefits(G)
    # end = time.time()
    # logging.info(f"Cost/Benefit Calculation Time: {end - start}")
    # pickle.dump(G, open(dataset + 'costs.gpickle', 'wb'))

    G = pickle.load(open(dataset + 'costs.gpickle', 'rb'))
    c_min, b_max = get_cmin_bmax(G)
    # logging.info(f"c_min: {c_min}, b_max: {b_max}")
    logging.info("Getting pareto set...")
    start = time.time()
    pareto_set = get_pareto(G, 0, r, t, c_min, b_max, max_length)
    end = time.time()
    logging.info(f"ApxMODis Search Time: {end - start}")

    # with open(dataset + 'pareto_old.json', "r") as file:
    #     pareto_dict = json.load(file)
    pareto_dict = dict(pareto_set)
    # pareto_json = json.dumps(pareto_dict, indent=4)
    # with open(dataset + 'si_pareto_' + str(epsilon) + '.json', 'w') as json_file:
    #     json_file.write(pareto_json)

    pareto = {}
    nodes = [k for k, v in pareto_dict.items() if v != {}]
    for node in nodes:
        for pos, path in pareto_dict[node].items():
            if pos in pareto.keys() and pareto[pos][2][-1] >= path[2][-1]:
                continue
            pareto[pos] = path
    pareto_json = json.dumps(pareto, indent=4)
    with open(dataset + 'apx' + str(epsilon) + '.json', 'w') as json_file:
        json_file.write(pareto_json)

    # pareto_size = sum(len(value.keys()) for value in pareto_dict.values())
    pareto_size = len(pareto)
    logging.info(f"Pareto Set Size: {pareto_size}")

    # Save nodes' data
    # G = nx.read_gpickle(dataset + 'costs.gpickle')
    node_data = {node: G.nodes[node] for node in G.nodes()}
    node_json = json.dumps(node_data, indent=4)
    with open(dataset + 'nodes.json', 'w') as json_file:
        json_file.write(node_json)


if __name__ == '__main__':
    main()
