import copy
import math
import numpy as np
import networkx as nx
import pandas as pd
from collections import defaultdict

dataset = "../Example/medium/t_cluster/"
nodes_df = pd.read_csv(dataset + 'nodes.csv')
edges_df = pd.read_csv(dataset + 'edges.csv')

# Construct graph
G = nx.from_pandas_edgelist(edges_df, 'Source', 'Target', edge_attr=['Type', 'Values'], create_using=nx.DiGraph())
labels_dict = nodes_df.set_index('Id')['Label'].to_dict()
nx.set_node_attributes(G, labels_dict, 'Label')

r = [0.2, 0.3, 0.2, 0.3]
t = [2, 2]


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


def pos(q: tuple, r: list):
    pos_q = []

    # Costs
    for i in range(len(q[1])):
        pos_q.append(math.floor(math.log(q[1][i] / c_min[i], r[i])))

    # Benefits
    for i in range(len(q[2])):
        pos_q.append(math.floor(math.log(q[2][i] / b_max[i], r[i + len(q[1])])))

    return tuple(pos_q)


def get_pareto(G, s, r, t):
    """
    :param G: Graph
    :param s: Source node
    :param r: Approximate Ratios
    :param t: Thresholds on costs
    :return: Pi
    """
    n = len(G)
    Pi = defaultdict(lambda: defaultdict(dict))
    Pi[s][0] = {(None, tuple([0, 0]), tuple([0, 0]), None, None): None}
    for i in range(1, n):
        for v in G.nodes:
            Pi[v][i] = copy.deepcopy(Pi[v][i - 1])
            for u in G.predecessors(v):
                Pi[v][i] = extend_and_merge(Pi[v][i], Pi[u][i - 1], G[u][v], r, t)

    return Pi


def extend_and_merge(R, Q, e, r, t):
    for p in Q:
        if p[0] is None:
            q = (e['Type'],
                 tuple(p_c + e_c for p_c, e_c in zip(p[1], e['c'])),
                 tuple(p_b + e_b for p_b, e_b in zip(p[2], e['b'])),
                 p, e)
        else:
            q = (str(p[0]) + e['Type'],
                 tuple(p_c + e_c for p_c, e_c in zip(p[1], e['c'])),
                 tuple(p_b + e_b for p_b, e_b in zip(p[2], e['b'])),
                 p, e)
        for i in range(len(q[1])):
            if q[1][i] > t[i]:
                continue
        pos_q = pos(q, r)
        if pos_q not in R or R[pos_q][2][0] < q[2][0]:
            R[pos_q] = q
    return R


all_costs, all_benefits = costs_benefits(G)
c_min = np.min(all_costs, axis=0)
b_max = np.max(all_benefits, axis=0)

pareto_set = get_pareto(G, 0, r, t)

pareto_set
