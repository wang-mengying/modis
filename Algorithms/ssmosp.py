import copy
import json
import logging
import math
import numpy as np
import networkx as nx
import pandas as pd
from collections import defaultdict

dataset = "../Example/small/t_cluster/"
logging.basicConfig(filename=dataset+'log_ssmosp.txt', level=logging.INFO, format='%(message)s')
nodes_df = pd.read_csv(dataset + 'nodes.csv')
edges_df = pd.read_csv(dataset + 'edges.csv')

# Construct graph
G = nx.from_pandas_edgelist(edges_df, 'Source', 'Target', edge_attr=['Type', 'Values'], create_using=nx.DiGraph())
labels_dict = nodes_df.set_index('Id')['Label'].to_dict()
nx.set_node_attributes(G, labels_dict, 'Label')


def cal_costs_benefits(source_id, target_id):
    # TODO replace with objectives
    cost_1 = np.random.randint(1, source_id + 10)
    cost_2 = np.random.randint(1, target_id + 10)
    benefit_1 = np.random.randint(1, source_id + 10)
    benefit_2 = np.random.randint(1, target_id + 10)
    return [cost_1, cost_2], [benefit_1, benefit_2]


def costs_benefits(G):
    all_costs = []
    all_benefits = []
    for u, v, l in G.edges(data=True):
        source_id = u
        target_id = v
        c, b = cal_costs_benefits(source_id, target_id)
        l['c'] = c
        l['b'] = b
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
                logging.info(f"v: {v}, u: {u}, i: {i}")
                Pi[i][v] = extend_and_merge(Pi, G, u, v, i, r, t, costs, benefits)

    return Pi[n-1]


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
        if pos_q not in R.keys() or R[pos_q][2][0] < q[2][0]:
            R[pos_q] = q
    # logging.info(f"R: {R}")
    return R


def main():
    r = [0.2, 0.3, 0.2, 0.3]
    t = [1000, 1000]

    costs, benefits = costs_benefits(G)

    pareto_set = get_pareto(G, 0, r, t, costs, benefits)

    pareto_dict = dict(pareto_set)
    pareto_json = json.dumps(pareto_dict, indent=4)

    with open(dataset + 'pareto.json', 'w') as json_file:
        json_file.write(pareto_json)


if __name__ == '__main__':
    main()
