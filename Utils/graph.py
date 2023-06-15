import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Tuple


def add_node(G: nx.DiGraph, label: Tuple) -> None:
    G.add_node(label)


def add_edge(G: nx.DiGraph, source_node: Tuple, target_node: Tuple, operation: str) -> None:
    G.add_edge(source_node, target_node, operation=operation)


def create_path(node: Tuple) -> Dict:
    path = {'node': node, 'operations': [], 'bc': np.zeros(6)}
    return path


def create_node_label(universal_schema: pd.DataFrame) -> Tuple:
    return tuple((i, 0) for i in range(len(universal_schema.columns)))


