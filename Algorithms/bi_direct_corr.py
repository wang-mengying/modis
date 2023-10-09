import ast
import json
import logging
import math
import pickle
import sys
import time
import pandas as pd

import joblib

from Algorithms.si_direct import get_cmin_bmax

sys.path.append("../")
import Dataset.Kaggle.others.movie_objectives as movie_objectives
import Utils.correlation_analysis as correlation_analysis

dataset = "../Dataset/Kaggle/others/d7m8/"
logging.basicConfig(filename=dataset + 'bi_direct/log_bi.txt', level=logging.INFO, format='%(message)s')
records = pd.read_csv('../Surrogate/Movie/sample_nodes.csv')
relations = correlation_analysis.gat_relations(dataset + 'nodes.json')


# Calculate a Box, which is a tuple of normalized costs and benefits
def cal_box(path, epsilon, c_min, b_max):
    box = []

    # Costs
    for i in range(len(path["costs"])):
        box.append(0) if path["costs"][i] == 0 or path["costs"][i] ==None else \
            box.append(math.floor(math.log(path["costs"][i] / c_min[i], 1 + epsilon[i])))

    # Benefits
    for i in range(len(path["benefits"])):
        box.append(0) if path["benefits"][i] == 0 or path["costs"][i] == None else \
            box.append(math.floor(math.log(path["benefits"][i] / b_max[i], 1 - epsilon[i + len(path["costs"])])))

    return tuple(box)


def costs_benefits(state, model_path='../Surrogate/Kaggle/movie_surrogate.joblib',
                   cluster_file='../Dataset/Kaggle/others/movie_clustered_table.csv'):
    node = {}
    node['Label'] = str(state)
    df = movie_objectives.surrogate_inputs(node, cluster_file)

    model = joblib.load(model_path)
    model_objectives = list(model.predict(df)[0])
    feature_objectives = movie_objectives.feature_objectives(node, cluster_file)

    costs = [feature_objectives[2], model_objectives[0], model_objectives[2]]
    benefits = [feature_objectives[0], feature_objectives[1], model_objectives[1]]

    return [costs, benefits]


def costs_benefits_part(state, cluster_file='../Dataset/Kaggle/others/movie_clustered_table.csv'):
    node = {}
    node['Label'] = str(state)

    feature_objectives = movie_objectives.feature_objectives(node, cluster_file)

    costs = [feature_objectives[2], None, None]
    benefits = [feature_objectives[0], feature_objectives[1], None]

    return [costs, benefits]


def find_related_objective(obj, relations):
    related_obj = None
    for relation in relations.keys():
        if relation[0] == obj:
            related_obj = relation[1]
            return related_obj
        if relation[1] == obj:
            related_obj = relation[0]
            return related_obj

    return related_obj


def fill_missing_objectives(state):
    costs, benefits = costs_benefits_part(state)
    c = ['vif', 'training_time', 'complexity']
    b = ['fisher', 'mutual_info', 'accuracy']

    node = {}
    node['Label'] = str(state)
    active_items = eval(node['Label'])[0].count(1)
    active_clusters = eval(node['Label'])[1].count(1)

    # if no missing value, return the original objectives
    if None not in costs and None not in benefits:
        return [costs, benefits]

    # if the node['Label'] is in the records, fill in missing objectives by the records
    if node['Label'] in records['Label'].values:
        node_info = records[records['Label'] == node['Label']].iloc[0]
        for i in range(len(c)):
            if costs[i] is None:
                costs[i] = node_info[c[i]]
        for i in range(len(b)):
            if benefits[i] is None:
                benefits[i] = node_info[b[i]]

    if None not in costs and None not in benefits:
        return [costs, benefits]

    # if the node['Label'] is not in the records, fill in missing objectives by the relation
    obj_has_relation = set()
    for relation in relations.keys():
        obj_has_relation.add(relation[0])
        obj_has_relation.add(relation[1])

    for i in range(len(c)):
        if costs[i] is None and c[i] in obj_has_relation:
            # find the related objective and its value
            related_obj = find_related_objective(c[i], relations)
            if related_obj == 'active_items':
                related_obj_value = active_items
            elif related_obj == 'active_clusters':
                related_obj_value = active_clusters
            else:
                related_obj_value = costs[c.index(related_obj)] if related_obj in c else benefits[b.index(related_obj)]

            # find the closed larger and smaller values of the related objective in the records
            larger_rows = records[records[related_obj] > related_obj_value]
            smaller_rows = records[records[related_obj] < related_obj_value]

            larger_value = larger_rows.sort_values(by=related_obj).iloc[0][c[i]] if len(larger_rows) > 0 else None
            smaller_value = smaller_rows.sort_values(by=related_obj, ascending=False).iloc[0][c[i]] if len(
                smaller_rows) > 0 else None

            if larger_value and smaller_value:
                costs[i] = (larger_value + smaller_value) / 2
            elif larger_value:
                costs[i] = larger_value
            elif smaller_value:
                costs[i] = smaller_value

    for i in range(len(b)):
        if benefits[i] is None and b[i] in obj_has_relation:
            # find the related objective and its value
            related_obj = find_related_objective(b[i], relations)
            related_obj_value = costs[c.index(related_obj)] if related_obj in c else benefits[b.index(related_obj)]

            # find the closed larger and smaller values of the related objective in the records
            larger_rows = records[records[related_obj] > related_obj_value]
            smaller_rows = records[records[related_obj] < related_obj_value]

            larger_value = larger_rows.sort_values(by=related_obj).iloc[0][b[i]] if len(larger_rows) > 0 else None
            smaller_value = smaller_rows.sort_values(by=related_obj, ascending=False).iloc[0][b[i]] if len(
                smaller_rows) > 0 else None

            if larger_value and smaller_value:
                benefits[i] = (larger_value + smaller_value) / 2
            elif larger_value:
                benefits[i] = larger_value
            elif smaller_value:
                benefits[i] = smaller_value

    return [costs, benefits]


# Check is path p2 dominates path p1
def is_dominated(b1, b2, p1, p2, epsilon):
    b1 = ast.literal_eval(b1)
    b2 = ast.literal_eval(b2)

    # Case 1: Same box, compare the last benefit
    if b1 == b2:
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
    costs = [G.nodes[node]['feature_objectives'][2],
             G.nodes[node]['model_objectives'][0],
             G.nodes[node]['model_objectives'][2]]

    benefits = [G.nodes[node]['feature_objectives'][0],
                G.nodes[node]['feature_objectives'][1],
                G.nodes[node]['model_objectives'][1]]

    return [costs, benefits]


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


def bi_directional_search_state(start_state, end_state, epsilon, c_min, b_max, max_length=20):
    pareto_set = {}
    sandwich_bounds = set()
    prun_set = set()

    fs_path = {'nodes': [start_state],
               'costs': fill_missing_objectives(start_state)[0],
               'benefits': fill_missing_objectives(start_state)[1]}
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
            if len(pathF['nodes']) > max_length/2:
                continue

            current_state = pathF['nodes'][-1]
            print(current_state)

            next_states = spawn_state(current_state, direction="F")
            for next_state in next_states:
                new_path = {'nodes': pathF['nodes'] + [next_state],
                            'costs': fill_missing_objectives(next_state)[0],
                            'benefits': fill_missing_objectives(next_state)[1]}
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
            if len(pathB['nodes']) > max_length/2:
                continue

            current_state = pathB['nodes'][-1]
            print(current_state)

            next_states = spawn_state(current_state, direction="B")
            for next_state in next_states:
                new_path = {'nodes': pathB['nodes'] + [next_state],
                            'costs': fill_missing_objectives(next_state)[0],
                            'benefits': fill_missing_objectives(next_state)[1]}
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


def main():
    G = pickle.load(open(dataset + 'costs.gpickle', 'rb'))

    start_node = 0
    end_node = 37653
    max_length = 20
    epsilon = [0.5] * 5 + [0.0001]
    c_min, b_max = get_cmin_bmax(G)
    pareto_set = {}

    start_time = time.time()
    # pareto = bi_directional_search(G, pareto_set, start_node, end_node, epsilon, c_min, b_max)
    pareto = bi_directional_search_state((tuple([1] * 11), tuple([1] * 11)),
                                         (tuple([0] * 11), tuple([0] * 11)), epsilon, c_min, b_max, max_length)
    end_time = time.time()

    logging.info(f"epsilon: {epsilon}")
    logging.info(f"max_length: {max_length}")
    logging.info(f"Search time: {end_time - start_time}")
    logging.info(f"Pareto set size: {len(pareto)}")
    pareto_json = json.dumps(pareto, indent=4)
    with open(dataset + 'bi_direct/pareto.json', 'w') as json_file:
        json_file.write(pareto_json)


if __name__ == '__main__':
    main()
