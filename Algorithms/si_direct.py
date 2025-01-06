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
from collections import defaultdict, Counter
from multiprocessing import Pool

sys.path.append("../")
import Dataset.Kaggle.others.movie_objectives as movie_objectives
import Trainer.house_random_forest as house_random_forest
import Utils.sample_nodes as sample
from concurrent.futures import ProcessPoolExecutor

Data = "../Dataset/Mental/"
max_length = 6
# dataset = Data + "1013/"
dataset = Data + "results/ml" + str(max_length) + "/"
dataset = dataset.replace('/', '\\')
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


def get_objectives(node_id, model, cluster_file='../Dataset/OpenData/House/processed/house_clustered.csv'):
    cluster_file = cluster_file.replace('/', '\\')
    node = G.nodes[node_id]
    df = movie_objectives.surrogate_inputs(node, cluster_file)
    model_objectives = model.predict(df)[0]
    # Movie
    # feature_objectives = movie_objectives.feature_objectives(node, cluster_file)
    # House
    df_table = sample.process_data(node['Label'], cluster_file)
    X, y, _ = house_random_forest.process_data(df_table)
    feature_objectives = house_random_forest.feature_objs(X, y)
    feature_objectives = list(feature_objectives)

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


def get_cluster_counts_modsnet(clustered_file='../Dataset/ModsNet/processed/graph_clustered.txt'.replace('/', '\\')):
    clustered_data = pd.read_csv(clustered_file, sep="\t", header=None, names=["connection", "cluster"])

    # Count edges for each cluster
    cluster_counts = Counter(clustered_data["cluster"])

    # Convert to dictionary
    cluster_count_dict = dict(cluster_counts)

    return cluster_count_dict


def get_cluster_counts(file_path='../Dataset/Mental/uni_table_clustered.csv'):
    data = pd.read_csv(file_path)

    if 'cluster' not in data.columns:
        raise ValueError("The input file does not contain a 'Cluster' column.")
    cluster_counts = data['cluster'].value_counts().to_dict()

    return cluster_counts


def process_batch(batch, cluster_count_dict, model_path, record):
    # Load the model inside the worker process
    model = joblib.load(model_path)

    batch_data = []
    batch_ids = []
    results = []

    for index, row in batch.iterrows():
        node_id = row['Id']
        print(node_id)

        #modsnet
        # metric_columns = ["precision@5", "precision@10", "recall@5", "recall@10", "ndcg@5", "ndcg@10"]
        #mental
        metric_columns = ['accuracy','precision','recall','f1','auc','time']

        if node_id in record["Id"].values:
            metrics = record.loc[record["Id"] == node_id, metric_columns].iloc[0].tolist()
            results.append((node_id, metrics, []))
            continue

        label = eval(row['Label'])
        features, clusters = label

        # Check the condition: if the 3rd digit in clusters (index 2) is 0 (ModsNet)
        # if clusters[2] == 0:
        #     results.append((node_id, [0, 0, 0, 0, 0, 0], []))
        #     continue

        # Prepare input data for the model
        data = {}
        data["active_items"] = features.count(1)
        data["active_values"] = clusters.count(1)
        data["num_rows"] = sum(count for cluster_id, count in cluster_count_dict.items() if clusters[cluster_id])
        data["num_cols"] = sum(features) + 1
        # modsnet
        # data["num_cols"] = sum(features[:9]) + (8 if any(features[9:]) else 0)

        # Add feature and cluster states
        for i, state in enumerate(features):
            data[f'feature_{i + 1}'] = state
        for i, state in enumerate(clusters):
            data[f'cluster_{i + 1}'] = state

        batch_data.append(data)
        batch_ids.append(node_id)

    # Perform model inference for the batch (if there's data)
    if batch_data:
        batch_df = pd.DataFrame(batch_data)
        batch_objectives = model.predict(batch_df)

        for node_id, objectives in zip(batch_ids, batch_objectives):
            results.append((node_id, objectives.tolist(), []))

    return results


def get_objectives_batch(cluster_count_dict, batch_size=1000,
                           model_path='../Surrogate/Mental/mental_surrogate.joblib'.replace('/','\\'),
                           record_path='../Surrogate/Mental/sample_nodes.csv'.replace('/', '\\')):

    global G
    results = []
    record = pd.read_csv(record_path)

    # Split nodes_df into batches
    batches = [nodes_df[start:start + batch_size] for start in range(0, len(nodes_df), batch_size)]

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_batch, batch, cluster_count_dict, model_path, record)
            for batch in batches
        ]
        for future in futures:
            results.extend(future.result())

    # Update the graph with results
    for node_id, model_objectives, feature_objectives in results:
        G.nodes[node_id]['model_objectives'] = model_objectives
        G.nodes[node_id]['feature_objectives'] = feature_objectives

    print("Graph updated with node attributes: 'model_objectives' and 'feature_objectives'.")
    return G


def cal_costs_benefits(args):
    edge, node_data = args
    print(edge)
    u, v = edge

    sm_objs, sf_objs = node_data[u]['model_objectives'], node_data[u]['feature_objectives']
    tm_objs, tf_objs = node_data[v]['model_objectives'], node_data[v]['feature_objectives']

    # Movie
    # time = tm_objs[0] - sm_objs[0]
    # accuracy = tm_objs[1] - sm_objs[1]
    # complexity = tm_objs[2] - sm_objs[2]
    # fisher = tf_objs[0] - sf_objs[0]
    # mutual_info = tf_objs[1] - sf_objs[1]
    # vif = tf_objs[2] - sf_objs[2]
    # costs = [vif, time, complexity]
    # benefits = [fisher, mutual_info, accuracy]

    # House
    # fisher = tf_objs[0] - sf_objs[0]
    # mutual_info = tf_objs[1] - sf_objs[1]
    # f1 = tm_objs[1] - sm_objs[1]
    # accuracy = tm_objs[0] - sm_objs[0]
    # training_time = tm_objs[2] - sm_objs[2]
    # costs = [training_time]
    # benefits = [fisher, mutual_info, f1, accuracy]

    # ModsNet
    # precision5 = tm_objs[0] - sm_objs[0]
    # precision10 = tm_objs[1] - sm_objs[1]
    # recall5 = tm_objs[2] - sm_objs[2]
    # recall10 = tm_objs[3] - sm_objs[3]
    # ndcg5 = tm_objs[4] - sm_objs[4]
    # ndcg10 = tm_objs[5] - sm_objs[5]
    # costs = []
    # benefits = [precision5, precision10, recall5, recall10, ndcg5, ndcg10]

    # ModsNet
    accuracy = tm_objs[0] - sm_objs[0]
    precision = tm_objs[1] - sm_objs[1]
    recall = tm_objs[2] - sm_objs[2]
    f1 = tm_objs[3] - sm_objs[3]
    auc = tm_objs[4] - sm_objs[4]
    time = tm_objs[5] - sm_objs[5]
    costs = [time]
    benefits = [accuracy, precision, recall, f1, auc]

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
    num_m = len(G.nodes[0]['model_objectives'])
    num_f = len(G.nodes[0]['feature_objectives'])

    model_objectives_mins = [min(G.nodes[node]['model_objectives'][i] for node in G.nodes()) for i in range(num_m)]
    feature_objectives_mins = [min(G.nodes[node]['feature_objectives'][i] for node in G.nodes()) for i in range(num_f)]

    model_objectives_maxs = [max(G.nodes[node]['model_objectives'][i] for node in G.nodes()) for i in range(num_m)]
    feature_objectives_maxs = [max(G.nodes[node]['feature_objectives'][i] for node in G.nodes()) for i in range(num_f)]

    # Movie
    # c_min = [feature_objectives_mins[2], model_objectives_mins[0], model_objectives_mins[2]]
    # b_max = [feature_objectives_maxs[0], feature_objectives_maxs[1], model_objectives_maxs[1]]
    # House
    c_min = [model_objectives_mins[2]]
    b_max = [feature_objectives_maxs[0], feature_objectives_maxs[1], model_objectives_maxs[1], model_objectives_maxs[0]]
    # modsnet
    # c_min = []
    # b_max = [model_objectives_maxs[0], model_objectives_maxs[1],  model_objectives_maxs[2],
    #           model_objectives_maxs[3],  model_objectives_maxs[4], model_objectives_maxs[5]]
    # mental
    # c_min = [model_objectives_mins[5]]
    # b_max = [model_objectives_maxs[0], model_objectives_maxs[1], model_objectives_maxs[2],
    #          model_objectives_maxs[3], model_objectives_maxs[4]]

    c_min = [0.0001 if c <= 0 else c for c in c_min]

    return c_min, b_max


def get_cmin(G):
    model_objectives_mins = [min(G.nodes[node]['model_objectives'][i] for node in G.nodes()) for i in range(3)]

    c_min = [model_objectives_mins[1], model_objectives_mins[2], model_objectives_mins[0]]
    # set c_min[i] to 0.0001 if c_min <= 0
    c_min = [0.0001 if c <= 0 else c for c in c_min]

    return c_min


def pos(q: tuple, r: list, c_min, b_max):
    pos_q = []

    # Costs
    for i in range(len(q[1])):
        pos_q.append(math.floor(math.log(q[1][i] / c_min[i], r[i])))

    # Benefits
    for i in range(len(q[2])-1):
        if q[2][i] == 0:
            pos_q.append(0)
            continue
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
    Pi = defaultdict(lambda: defaultdict(dict))
    # Movie
    # c = [G.nodes[s]['feature_objectives'][2], G.nodes[s]['model_objectives'][0], G.nodes[s]['model_objectives'][2]]
    # b = [G.nodes[s]['feature_objectives'][0], G.nodes[s]['feature_objectives'][1], G.nodes[s]['model_objectives'][1]]
    # House
    # c = [G.nodes[s]['model_objectives'][2]]
    # b = [G.nodes[s]['feature_objectives'][0], G.nodes[s]['feature_objectives'][1], G.nodes[s]['model_objectives'][1], G.nodes[s]['model_objectives'][0]]
    # modsnet
    # c = []
    # b = [G.nodes[s]['model_objectives'][0], G.nodes[s]['model_objectives'][1], G.nodes[s]['model_objectives'][2],
    #      G.nodes[s]['model_objectives'][3], G.nodes[s]['model_objectives'][4], G.nodes[s]['model_objectives'][5]]
    # mental
    c = [G.nodes[s]['model_objectives'][5]]
    b = [G.nodes[s]['model_objectives'][0], G.nodes[s]['model_objectives'][1], G.nodes[s]['model_objectives'][2],
         G.nodes[s]['model_objectives'][3], G.nodes[s]['model_objectives'][4]]

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
    logging.info(f"max_length: {max_length}")
    epsilon = 0.02

    # Movie
    # r = [1 + epsilon, 1 + epsilon, 1 + epsilon, 1 - epsilon, 1 - epsilon, 1]
    # t = [5, 1.5, 555]
    # House
    # r = [1 + epsilon, 1 - epsilon, 1 - epsilon, 1 - epsilon, 1]
    # t = [2]
    # model_path = '../Surrogate/House/house_surrogate.joblib'
    # ModsNet
    # r = [1 - epsilon, 1 - epsilon, 1 - epsilon, 1 - epsilon, 1 - epsilon, 1]
    # t = []
    # mental
    r = [1 - epsilon, 1 - epsilon, 1 - epsilon, 1 - epsilon, 1 - epsilon, 1]
    t = [0.45]
    # model_path = model_path.replace("/", "\\")


    start = time.time()
    logging.info("Nodes objectives...")
    # nodes_objectives(G, model_path)
    # cluter_counts = get_cluster_counts()
    # G = get_objectives_batch(cluter_counts)
    # pickle.dump(G, open(dataset + 'objectives.gpickle', 'wb'))
    # logging.info("Edges costs/benefits...")
    # costs_benefits(G)
    # end = time.time()
    # logging.info(f"Cost/Benefit Calculation Time: {end - start}")
    # pickle.dump(G, open(dataset + 'costs.gpickle', 'wb'))


    G = pickle.load(open(dataset + 'costs.gpickle', 'rb'))
    logging.info(f"epsilon: {epsilon}")
    c_min, b_max = get_cmin_bmax(G)
    logging.info(f"c_min: {c_min}, b_max: {b_max}")
    logging.info("Getting pareto set...")
    start = time.time()
    pareto_set = get_pareto(G, 0, r, t, c_min, b_max, max_length)
    end = time.time()
    logging.info(f"ApxMODis Search Time: {end - start}")

    pareto_dict = dict(pareto_set)
    pareto_json = json.dumps(pareto_dict, indent=4)
    with open(dataset + 'si_pareto_' + str(epsilon) + '.json', 'w') as json_file:
        json_file.write(pareto_json)

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