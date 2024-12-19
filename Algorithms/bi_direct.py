import ast
import json
import logging
import math
import pickle
import sys
import time
import joblib
import pandas as pd
import Trainer.movie_gradient_boosting as mgb_movie

from Algorithms.si_direct import get_cmin_bmax, get_cmin
from Utils.graph_igraph_table import pad_tuple
import Utils.sample_nodes as sample
import Trainer.house_random_forest as house_random_forest

sys.path.append("../")
import Dataset.Kaggle.others.movie_objectives as movie_objectives

Data = "../Dataset/OpenData/House/"
max_length = 2

# dataset = Data + "1108/"
dataset = Data + "results/ml" + str(max_length) + "/"
dataset = dataset.replace('/', '\\')
logging.basicConfig(filename=Data+'log.txt', level=logging.INFO, format='%(message)s')


# Calculate a Box, which is a tuple of normalized costs and benefits
def cal_box(path, epsilon, c_min, b_max):
    box = []

    # Costs
    for i in range(len(path["costs"])):
        box.append(0) if path["costs"][i] == 0 else \
            box.append(math.floor(math.log(path["costs"][i] / c_min[i], 1 + epsilon[i])))

    # Benefits
    for i in range(len(path["benefits"])):
        box.append(0) if path["benefits"][i] == 0 else \
            box.append(math.floor(math.log(path["benefits"][i] / b_max[i], 1 - epsilon[i + len(path["costs"])])))

    return tuple(box)


def cal_box_cost_only(path, epsilon, c_min):
    box = []

    # Costs
    for i in range(len(path["costs"])):
        box.append(0) if path["costs"][i] == 0 else \
            box.append(math.floor(math.log(path["costs"][i] / c_min[i], 1 + epsilon[i])))

    return tuple(box)


def costs_benefits(state, model_path='../Surrogate/House/house_surrogate.joblib',
                   cluster_file='../Dataset/OpenData/House/processed/house_clustered.csv'):
    cluster_file = cluster_file.replace('/', '\\')

    node = {}
    # node['Label'] = str(pad_tuple(str(state)))
    node['Label'] = str(state)
    df = movie_objectives.surrogate_inputs(node, cluster_file)

    model = joblib.load(model_path)
    model_objectives = model.predict(df)[0]

    # Movie
    # feature_objectives = movie_objectives.feature_objectives(node, cluster_file)
    # costs = [feature_objectives[2], model_objectives[0], model_objectives[2]]
    # benefits = [feature_objectives[0], feature_objectives[1], model_objectives[1]]

    # House
    df_table = sample.process_data(node['Label'], cluster_file)
    X, y, _ = house_random_forest.process_data(df_table)
    feature_objectives = house_random_forest.feature_objs(X, y)
    feature_objectives = list(feature_objectives)
    costs = [model_objectives[2]]
    benefits = [feature_objectives[0], feature_objectives[1], model_objectives[1], model_objectives[0]]

    return [costs, benefits]


def costs(state, model_path='../Surrogate/HuggingFace/hf_surrogate.joblib',
                   cluster_file=Data+'clustered_table.csv'):
    node = {}
    node['Label'] = str(state)
    df = movie_objectives.surrogate_inputs(node, cluster_file)

    model = joblib.load(model_path)
    model_objectives = list(model.predict(df)[0])

    costs = [model_objectives[1], model_objectives[2], model_objectives[0]]
    costs = [0.0001 if c <= 0 else c for c in costs]

    return costs


# Check is path p2 dominates path p1
def is_dominated(b1, b2, p1, p2, epsilon):
    b1 = ast.literal_eval(b1)
    b2 = ast.literal_eval(b2)

    # Case 1: Same box, compare the last benefit
    if b1 == b2:
        if p1['benefits'] is None or p2['benefits'] is None:
            return p1['costs'][-1] <= p2['costs'][-1]
        else:
            return p1['benefits'][-1] < p2['benefits'][-1]

    # Case 2: Different boxes, compare the box
    for i in range(len(b1)):
        # if b2[i] > (1 + epsilon[i]) * b1[i]:
        # we already consider epsilon in the box calculation
        if b2[i] > b1[i]:
            return False

    return True


def update_pareto(pareto_set: dict, box, path, epsilon, prun_set):
    added = False

    # Case 1: The box already exists in the Pareto set
    if box in pareto_set.keys():
        ex_path = pareto_set[box]
        # 1.1 existing path > new path, do nothing
        if is_dominated(box, box, path, ex_path, epsilon):
            return pareto_set, added, prun_set
        # 1.2 new path >= existing path, replace the existing path
        pareto_set[box] = path
        return pareto_set, added, prun_set

    # Case2: The box does not exist in the Pareto set
    to_remove = []
    for ex_box, ex_path in pareto_set.items():
        # 2.1 exist an existing box > new box, do nothing
        if is_dominated(box, ex_box, path, ex_path, epsilon):
            return pareto_set, added, prun_set
        # 2.2 new box > this existing box, mark this existing box to be removed
        if is_dominated(ex_box, box, ex_path, path, epsilon):
            to_remove.append(ex_box)

    # Remove boxes that are dominated
    for b in to_remove:
        del pareto_set[b]

    prun_set.update(to_remove)

    # Add the new box
    pareto_set[box] = path
    added = True

    return pareto_set, added, prun_set


def obj2cb(G, node):
    # Movie
    # costs = [G.nodes[node]['feature_objectives'][2],
    #          G.nodes[node]['model_objectives'][0],
    #          G.nodes[node]['model_objectives'][2]]
    #
    # benefits = [G.nodes[node]['feature_objectives'][0],
    #             G.nodes[node]['feature_objectives'][1],
    #             G.nodes[node]['model_objectives'][1]]

    # House
    costs = [G.nodes[node]['model_objectives'][2]]

    benefits = [G.nodes[node]['feature_objectives'][0],
                G.nodes[node]['feature_objectives'][1],
                G.nodes[node]['model_objectives'][1],
                G.nodes[node]['model_objectives'][0]]

    return [costs, benefits]


def obj2cb_cost_only(G, node):
    costs = [G.nodes[node]['model_objectives'][1],
                G.nodes[node]['model_objectives'][2],
                G.nodes[node]['model_objectives'][0]]

    return costs


# Spawn new paths from the last node in the path
def spawn(G, path, epsilon, c_min, b_max, direction="F"):
    current_node = path['nodes'][-1]  # the last node in the path
    neighbors = list(G.neighbors(current_node)) if direction == "F" else list(G.predecessors(current_node))

    extend_paths = {}
    for neighbor in neighbors:
        new_path = {'nodes': path['nodes'] + [neighbor],
                    'costs': obj2cb(G, neighbor)[0],
                    'benefits': obj2cb(G, neighbor)[1]}
        box = str(cal_box(new_path, epsilon, c_min, b_max))
        extend_paths[box] = new_path
    return extend_paths


def spawn_cost_only(G, path, epsilon, c_min, direction="F"):
    current_node = path['nodes'][-1]  # the last node in the path
    neighbors = list(G.neighbors(current_node)) if direction == "F" else list(G.predecessors(current_node))

    extend_paths = {}
    for neighbor in neighbors:
        new_path = {'nodes': path['nodes'] + [neighbor],
                    'costs': obj2cb_cost_only(G, neighbor)}
        box = str(cal_box(new_path, epsilon, c_min))
        extend_paths[box] = new_path
    return extend_paths


def spawn_state(state, direction="F"):
    columns, rows = state
    neighbors = []

    # Helper function to ensure at least one "1" is present in the tuple
    def ensure_one_present(tpl):
        return 1 in tpl

    # columns tuple
    for i in range(len(columns)):
        if (direction == "F" and columns[i] == 1) or (direction == "B" and columns[i] == 0):
            new_columns = list(columns)
            new_columns[i] = 1 - columns[i]
            new_state = (tuple(new_columns), rows)

            # If the new state has no "1" in the columns tuple, add a "1" to the columns tuple
            if ensure_one_present(new_state[0]) and not ensure_one_present(new_state[1]):
                new_rows = list(rows)
                new_rows[0] = 1
                new_state = (tuple(new_columns), tuple(new_rows))
            # If the new state has no "1" in the rows tuple, add a "1" to the rows tuple
            if ensure_one_present(new_state[1]) and not ensure_one_present(new_state[0]):
                new_columns = list(columns)
                new_columns[0] = 1
                new_state = (tuple(new_columns), rows)
            # If the new state has "1" in both the columns and rows tuple, add it to the neighbors
            if ensure_one_present(new_state[0]) and ensure_one_present(new_state[1]):
                neighbors.append(new_state)

    # rows tuple
    for i in range(len(rows)):
        if (direction == "F" and rows[i] == 1) or (direction == "B" and rows[i] == 0):
            new_rows = list(rows)
            new_rows[i] = 1 - rows[i]
            new_state = (columns, tuple(new_rows))

            # If the new state has no "1" in the columns tuple, add a "1" to the columns tuple
            if ensure_one_present(new_state[1]) and not ensure_one_present(new_state[0]):
                new_columns = list(columns)
                new_columns[0] = 1
                new_state = (tuple(new_columns), tuple(new_rows))
            # If the new state has no "1" in the rows tuple, add a "1" to the rows tuple
            if ensure_one_present(new_state[0]) and not ensure_one_present(new_state[1]):
                new_rows = list(rows)
                new_rows[0] = 1
                new_state = (columns, tuple(new_rows))
            # If the new state has "1" in both the columns and rows tuple, add it to the neighbors
            if ensure_one_present(new_state[0]) and ensure_one_present(new_state[1]):
                neighbors.append(new_state)

    return neighbors


def bi_directional_search(G, pareto_set, start_node, end_node, epsilon, c_min, b_max):
    sandwich_bounds = set()
    prun_set = set()

    visitedF = set()
    visitedB = set()

    fs_path = {'nodes': [start_node], 'costs': obj2cb(G, start_node)[0], 'benefits': obj2cb(G, start_node)[1]}
    bs_path = {'nodes': [end_node], 'costs': obj2cb(G, end_node)[0], 'benefits': obj2cb(G, end_node)[1]}

    pathsF = {str(cal_box(fs_path, epsilon, c_min, b_max)): fs_path}
    pathsB = {str(cal_box(bs_path, epsilon, c_min, b_max)): bs_path}

    while pathsF or pathsB:
        new_pathsF = {}
        new_pathsB = {}

        # Forward exploration
        for boxF, pathF in pathsF.items():
            current_node = pathF['nodes'][-1]
            visitedF.add(current_node)
            print(current_node)

            extend_paths = spawn(G, pathF, epsilon, c_min, b_max, direction="F")
            for box, new_path in extend_paths.items():
                if box in prun_set:
                    continue

                prun = False
                for bound in sandwich_bounds:
                    # check if the new box is dominated by the upper bound and dominates the lower bound
                    if is_dominated(box, bound[1], new_path, pareto_set.get(bound[1], None), epsilon) and \
                            is_dominated(bound[0], box, pareto_set.get(bound[0], None), new_path, epsilon):
                        prun = True
                        break
                if not prun:
                    pareto_set, added, prun_set = update_pareto(pareto_set, box, new_path, epsilon, prun_set)
                    if added:
                        new_pathsF[box] = new_path

        # Backward exploration
        for boxB, pathB in pathsB.items():
            current_node = pathB['nodes'][-1]
            visitedB.add(current_node)
            print(current_node)

            extend_paths = spawn(G, pathB, epsilon, c_min, b_max, direction="B")
            for box, new_path in extend_paths.items():
                if box in prun_set:
                    continue

                prun = False
                for bound in sandwich_bounds:
                    # check if the new box is dominated by the upper bound and dominates the lower bound
                    if is_dominated(box, bound[1], new_path, pareto_set.get(bound[1], None), epsilon) and \
                            is_dominated(bound[0], box, pareto_set.get(bound[0], None), new_path, epsilon):
                        prun = True
                        break
                if not prun:
                    pareto_set, added, prun_set = update_pareto(pareto_set, box, new_path, epsilon, prun_set)
                    if added:
                        new_pathsB[box] = new_path

        # Terminate if the two searches meet
        if visitedF & visitedB:
            break

        # Update sandwich bounds based on the updated Pareto set
        for boxF, pathF in new_pathsF.items():
            for boxB, pathB in new_pathsB.items():
                if is_dominated(boxF, boxB, pathF, pathB, epsilon) and boxF[-1] == boxB[-1]:
                    sandwich_bounds.add((boxF, boxB))
                elif is_dominated(boxB, boxF, pathB, pathF, epsilon) and boxB[-1] == boxF[-1]:
                    sandwich_bounds.add((boxB, boxF))

        pathsF = new_pathsF
        pathsB = new_pathsB

    return pareto_set


def bi_directional_search_state(start_state, end_state, epsilon, c_min, b_max, max_length):
    pareto_set = {}
    sandwich_bounds = set()
    prun_set = set()

    fs_path = {'nodes': [start_state],
               'costs': costs_benefits(start_state)[0],
               'benefits': costs_benefits(start_state)[1]}
    bs_path = {'nodes': [end_state],
               'costs': 0,
               'benefits': 0}

    pathsF = {str(cal_box(fs_path, epsilon, c_min, b_max)): fs_path}
    pathsB = {str((0, 0, 0, 0, 0, 0)): bs_path}

    while pathsF or pathsB:
        new_pathsF = {}
        new_pathsB = {}

        # Forward exploration
        print(f"pathsF: {len(pathsF)}")
        for boxF, pathF in pathsF.items():
            if len(pathF['nodes']) > max_length:
                continue

            current_state = pathF['nodes'][-1]
            print(current_state)

            next_states = spawn_state(current_state, direction="F")
            for next_state in next_states:
                new_path = {'nodes': pathF['nodes'] + [next_state],
                            'costs': costs_benefits(next_state)[0],
                            'benefits': costs_benefits(next_state)[1]}
                box = str(cal_box(new_path, epsilon, c_min, b_max))
                if box in prun_set:
                    continue

                prun = False
                for bound in sandwich_bounds:
                    if is_dominated(box, bound[1], new_path, pareto_set.get(bound[1], None), epsilon) and \
                            is_dominated(bound[0], box, pareto_set.get(bound[0], None), new_path, epsilon):
                        prun = True
                        break
                if not prun:
                    pareto_set, added, prun_set = update_pareto(pareto_set, box, new_path, epsilon, prun_set)
                    if added:
                        new_pathsF[box] = new_path

        # Backward exploration
        print(f"pathsB: {len(pathsB)}")
        for boxB, pathB in pathsB.items():
            if len(pathB['nodes']) > max_length:
                continue

            current_state = pathB['nodes'][-1]
            print(current_state)

            next_states = spawn_state(current_state, direction="B")
            for next_state in next_states:
                new_path = {'nodes': pathB['nodes'] + [next_state],
                            'costs': costs_benefits(next_state)[0],
                            'benefits': costs_benefits(next_state)[1]}
                box = str(cal_box(new_path, epsilon, c_min, b_max))
                if box in prun_set:
                    continue

                prun = False
                for bound in sandwich_bounds:
                    if is_dominated(box, bound[1], new_path, pareto_set.get(bound[1], None), epsilon) and \
                            is_dominated(bound[0], box, pareto_set.get(bound[0], None), new_path, epsilon):
                        prun = True
                        break
                if not prun:
                    pareto_set, added, prun_set = update_pareto(pareto_set, box, new_path, epsilon, prun_set)
                    if added:
                        new_pathsB[box] = new_path

        # Termination condition
        if set(pathF['nodes'][-1] for pathF in pathsF.values()) & set(pathB['nodes'][-1] for pathB in pathsB.values()):
            break

        # Update sandwich bounds
        for boxF, pathF in new_pathsF.items():
            for boxB, pathB in new_pathsB.items():
                # check the last element of the box to ensure them in the same layer on the most important objective
                if is_dominated(boxF, boxB, pathF, pathB, epsilon) and boxF[-1] == boxB[-1]:
                    sandwich_bounds.add((boxF, boxB))
                elif is_dominated(boxB, boxF, pathB, pathF, epsilon) and boxB[-1] == boxF[-1]:
                    sandwich_bounds.add((boxB, boxF))

        pathsF = new_pathsF
        pathsB = new_pathsB

    return pareto_set


def bi_directional_search_state_cost_only(start_state, end_state, epsilon, c_min, max_length):
    pareto_set = {}
    sandwich_bounds = set()
    prun_set = set()

    fs_path = {'nodes': [start_state],
               'costs': costs(start_state)}
    bs_path = {'nodes': [end_state],
               'costs': 0}

    pathsF = {str(cal_box_cost_only(fs_path, epsilon, c_min)): fs_path}
    pathsB = {str((0, 0, 0, 0, 0, 0)): bs_path}

    while pathsF or pathsB:
        new_pathsF = {}
        new_pathsB = {}

        # Forward exploration
        print(f"pathsF: {len(pathsF)}")
        for boxF, pathF in pathsF.items():
            if len(pathF['nodes']) > max_length:
                continue

            current_state = pathF['nodes'][-1]
            print(current_state)

            next_states = spawn_state(current_state, direction="F")
            for next_state in next_states:
                new_path = {'nodes': pathF['nodes'] + [next_state],
                            'costs': costs(next_state)}
                box = str(cal_box(new_path, epsilon, c_min))
                if box in prun_set:
                    continue

                prun = False
                for bound in sandwich_bounds:
                    if is_dominated(box, bound[1], new_path, pareto_set.get(bound[1], None), epsilon) and \
                            is_dominated(bound[0], box, pareto_set.get(bound[0], None), new_path, epsilon):
                        prun = True
                        break
                if not prun:
                    pareto_set, added, prun_set = update_pareto(pareto_set, box, new_path, epsilon, prun_set)
                    if added:
                        new_pathsF[box] = new_path

        # Backward exploration
        print(f"pathsB: {len(pathsB)}")
        for boxB, pathB in pathsB.items():
            if len(pathB['nodes']) > max_length:
                continue

            current_state = pathB['nodes'][-1]
            print(current_state)

            next_states = spawn_state(current_state, direction="B")
            for next_state in next_states:
                new_path = {'nodes': pathB['nodes'] + [next_state],
                            'costs': costs(next_state)}
                box = str(cal_box_cost_only(new_path, epsilon, c_min))
                if box in prun_set:
                    continue

                prun = False
                for bound in sandwich_bounds:
                    if is_dominated(box, bound[1], new_path, pareto_set.get(bound[1], None), epsilon) and \
                            is_dominated(bound[0], box, pareto_set.get(bound[0], None), new_path, epsilon):
                        prun = True
                        break
                if not prun:
                    pareto_set, added, prun_set = update_pareto(pareto_set, box, new_path, epsilon, prun_set)
                    if added:
                        new_pathsB[box] = new_path

        # Termination condition
        if set(pathF['nodes'][-1] for pathF in pathsF.values()) & set(pathB['nodes'][-1] for pathB in pathsB.values()):
            break

        # Update sandwich bounds
        for boxF, pathF in new_pathsF.items():
            for boxB, pathB in new_pathsB.items():
                # check the last element of the box to ensure them in the same layer on the most important objective
                if is_dominated(boxF, boxB, pathF, pathB, epsilon) and boxF[-1] == boxB[-1]:
                    sandwich_bounds.add((boxF, boxB))
                elif is_dominated(boxB, boxF, pathB, pathF, epsilon) and boxB[-1] == boxF[-1]:
                    sandwich_bounds.add((boxB, boxF))

        pathsF = new_pathsF
        pathsB = new_pathsB

    return pareto_set


def pre_clusters(df, target, classif=True):
    if not classif:
        return df['cluster'].value_counts().head(2).index.tolist()

    classes = df[target].unique()
    clusters = {}

    for target_class in classes:
        dominant_cluster = df[df[target] == target_class].groupby('cluster').size().idxmax()
        clusters[target_class] = dominant_cluster

    return clusters


def main():
    G = pickle.load(open('../Dataset/OpenData/House/results/ml6/costs.gpickle', 'rb'))

    # start_node = 0
    # end_node = 37653
    e = 0.02
    epsilon = [e] * 5
    # epsilon = [e] * 5
    feature = 20
    cluster = 7

    if "Kaggle" in Data or "Scale" in Data:
        c_min, b_max = get_cmin_bmax(G)
        clusters = pd.read_csv('../Dataset/Kaggle/others/movie_clustered_table.csv')
        clusters = mgb_movie.preprocess_data(clusters)
        pre = pre_clusters(clusters, 'gross_class')
        indices = [pre[i] for i in pre.keys()]

        start_state = (tuple([1] * feature), tuple([1] * cluster))
        end_state = (tuple([0] * feature), tuple(1 if i in indices else 0 for i in range(cluster)))

        start_time = time.time()
        # pareto = bi_directional_search(G, pareto_set, start_node, end_node, epsilon, c_min, b_max)
        pareto = bi_directional_search_state(start_state, end_state, epsilon, c_min, b_max, max_length)
        end_time = time.time()

    elif "House" in Data:
        c_min, b_max = get_cmin_bmax(G)
        clusters = pd.read_csv(Data + 'processed/house_clustered.csv')
        X, y, _ = house_random_forest.process_data(clusters)
        clusters = pd.concat([X, y], axis=1)
        pre = pre_clusters(clusters, 'PRICE_CLASS')
        indices = [pre[i] for i in pre.keys()]

        start_state = (tuple([1] * feature), tuple([1] * cluster))
        end_state = (tuple([0] * feature), tuple(1 if i in indices else 0 for i in range(cluster)))

        start_time = time.time()
        pareto = bi_directional_search_state(start_state, end_state, epsilon, c_min, b_max, max_length)
        end_time = time.time()
    elif "HuggingFace" in Data:
        c_min = get_cmin(G)
        # clusters = pd.read_csv(Data + 'clustered_table.csv')
        # clusters = mgb.preprocess_data(clusters)
        # pre = pre_clusters(clusters, 'gross_class')
        # indices = [pre[i] for i in pre.keys()]

        # clusters = pd.read_csv(Data + 'clustered_table.csv')
        # indices = pre_clusters(clusters, None, None)

        # start_state = (tuple([1] * 11), tuple([1] * 11))
        # end_state = (tuple([0] * 11), tuple(1 if i in indices else 0 for i in range(11)))

        # start_state = (tuple([1] * 12), tuple([1] * 10))
        # end_state = (tuple([0] * 12), tuple(1 if i in indices else 0 for i in range(10)))

    logging.info(f"epsilon: {epsilon}")
    logging.info(f"max_length: {max_length}")
    logging.info(f"Search time: {end_time - start_time}")
    logging.info(f"Pareto set size: {len(pareto)}")
    pareto_json = json.dumps(pareto, indent=4)
    with open(dataset + '/no' + str(e) + '.json', 'w') as json_file:
        json_file.write(pareto_json)


if __name__ == '__main__':
    main()
