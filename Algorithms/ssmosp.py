import networkx as nx
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


def SSMOSP(G: nx.DiGraph, start: str, bc: List, bc_thresholds: np.array, approx_ratios: List) -> Dict:
    # Todo: SSMOSP Algorithm, fill in details, trans graph to dynamic

    pareto = defaultdict(dict)

    for i in range(1, G.number_of_nodes()):
        for v in G.nodes():
            if is_end_node(v):
                continue

    for u in G.predecessors(v):
        e = G.edges[(u, v)]

        for _, p in pareto[u].items():
            if len(p['operations']) >= 3:
                continue

            p_prime = extend_path(p, e)

            pos = compute_position(p_prime['bc'], approx_ratios)

            if pos not in pareto[v] or compare_paths(p_prime['bc'], pareto[v][pos]['bc']):
                pareto[v][pos] = p_prime

    return pareto


def extend_path(p: Dict, e: Dict) -> Dict:
    q = p.copy()

    # Todo: "Extend" part for SSMOSP

    return q


def is_end_node(node: Tuple) -> bool:
    return all(flag == 1 for _, flag in node)


def compute_position(bc: np.array, approx_ratios: List) -> Tuple:
    return tuple(int(np.log(cost) / scale) for cost, scale in zip(bc, approx_ratios))


def compare_paths(bc1: np.array, bc2: np.array) -> bool:
    # Todo: Add exact objective
    return all(b1 <= b2 for b1, b2 in zip(bc1, bc2))
