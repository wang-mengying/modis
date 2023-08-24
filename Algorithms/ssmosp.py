import copy
import json
import logging
import math
import sys
import time

import joblib
import numpy as np
import networkx as nx
import pandas as pd
from collections import defaultdict
from multiprocessing import Pool, cpu_count

sys.path.append("../")
import Dataset.Movie.others.movie_objectives as movie_objectives

dataset = "../Dataset/Movie/others/d7m8/"
logging.basicConfig(filename=dataset + 'log_ssmosp.txt', level=logging.INFO, format='%(message)s')
nodes_df = pd.read_csv(dataset + 'nodes.csv')
edges_df = pd.read_csv(dataset + 'edges.csv')

# Construct graph
G = nx.from_pandas_edgelist(edges_df, 'Source', 'Target', edge_attr=['Type', 'Values'], create_using=nx.DiGraph())
labels_dict = nodes_df.set_index('Id')['Label'].to_dict()
nx.set_node_attributes(G, labels_dict, 'Label')


surrogate_model = None


def worker_initializer(model_path):
    global surrogate_model
    surrogate_model = joblib.load(model_path)


def get_objectives(node_id, model, cluster_file='../Dataset/Movie/others/movie_clustered_table.csv'):
    node = G.nodes[node_id]
    df = movie_objectives.surrogate_inputs(node, cluster_file)
    model_objectives = model.predict(df)[0]
    feature_objectives = movie_objectives.feature_objectives(node, cluster_file)

    return model_objectives, feature_objectives


def node_obj_map(node_id):
    mobjs, fobjs = get_objectives(node_id, surrogate_model)
    G.nodes[node_id]['model_objectives'] = mobjs
    G.nodes[node_id]['feature_objectives'] = fobjs
    print(node_id)


def nodes_objectives(G, model_path):
    nodes = list(G.nodes())

    with Pool(cpu_count(), initializer=worker_initializer, initargs=(model_path,)) as pool:
        pool.map(node_obj_map, nodes)


def cal_costs_benefits(edge):
    u, v = edge
    sm_objs, sf_objs = G.nodes[u]['model_objectives'], G.nodes[u]['feature_objectives']
    tm_objs, tf_objs = G.nodes[v]['model_objectives'], G.nodes[v]['feature_objectives']

    time = tm_objs[0] - sm_objs[0]
    accuracy = tm_objs[1] - sm_objs[1]
    complexity = tm_objs[2] - sm_objs[2]
    fisher = tf_objs[0] - sf_objs[0]
    mutual_info = tf_objs[1] - sf_objs[1]
    vif = tf_objs[2] - sf_objs[2]

    G[u][v]["c"] = [vif, time, complexity]
    G[u][v]["b"] = [fisher, mutual_info, accuracy]

    print(edge)

    return G[u][v]["c"], G[u][v]["b"]


def costs_benefits(G):
    edges = [(u, v) for u, v, _ in G.edges(data=True)]

    # Use a Pool to parallelize the computation with initializer to set up the model
    with Pool(cpu_count()) as pool:
        results = pool.map(cal_costs_benefits, edges)

    # Update the graph with computed costs and benefits and aggregate results
    all_costs = []
    all_benefits = []
    for c, b in results:
        all_costs.append(c)
        all_benefits.append(b)

    return all_costs, all_benefits


def pos(q: tuple, r: list, all_costs, all_benefits):
    pos_q = []
    c_min = np.min(all_costs, axis=0)
    b_max = np.max(all_benefits, axis=0)

    # Costs
    for i in range(len(q[1])):
        pos_q.append(math.floor(math.log(q[1][i] / c_min[i], r[i])))

    # Benefits
    for i in range(len(q[2])):
        pos_q.append(math.floor(math.log(q[2][i] / b_max[i], r[i + len(q[1])])))

    return tuple(pos_q)


def get_pareto(G, s, r, t, costs, benefits):
    """
    :param G: Graph
    :param s: Source node
    :param r: Approximate Ratios
    :param t: Thresholds on costs
    :param costs: All costs
    :param benefits: All benefits
    :return: Pi
    """
    n = len(G)
    Pi = defaultdict(lambda: defaultdict(dict))
    Pi[0][s][0] = (G.nodes[s]['Label'], tuple([0, 0]), tuple([0, 0]), None, None)
    for i in range(1, n):
        for v in G.nodes:
            Pi[i][v] = copy.deepcopy(Pi[i - 1][v])
            for u in G.predecessors(v):
                # logging.info(f"v: {v}, u: {u}, i: {i}")
                Pi[i][v] = extend_and_merge(Pi, G, u, v, i, r, t, costs, benefits)

    return Pi[n - 1]


def extend_and_merge(Pi, G, u, v, i, r, t, costs, benefits):
    R = Pi[i][v]
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
        pos_q = pos(q, r, costs, benefits)
        pos_q = str(pos_q)
        if pos_q not in R.keys() or R[pos_q][2][-1] < q[2][-1]:
            R[pos_q] = q
    # logging.info(f"R: {R}")
    return R


def main():
    r = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1]
    t = [5, 1.5, 555]

    model_path = '../Surrogate/Movie/movie_surrogate.joblib'

    start = time.time()
    logging.info("Nodes objectives...")
    nodes_objectives(G, model_path)
    logging.info("Edges costs/benefits...")
    costs, benefits = costs_benefits(G)
    end = time.time()
    logging.info(f"Cost/Benefit Calculation Time: {end - start}")

    logging.info("Getting pareto set...")
    start = time.time()
    pareto_set = get_pareto(G, 0, r, t, costs, benefits)
    end = time.time()
    logging.info(f"Search Time: {end - start}")

    pareto_dict = dict(pareto_set)
    pareto_json = json.dumps(pareto_dict, indent=4)

    with open(dataset + 'pareto.json', 'w') as json_file:
        json_file.write(pareto_json)


if __name__ == '__main__':
    main()
