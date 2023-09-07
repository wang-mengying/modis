import ast
import json
import logging
import math
import pickle
import time

from Algorithms.si_direct import get_cmin_bmax

dataset = "../Dataset/Movie/others/d7m8/"
logging.basicConfig(filename=dataset + 'bi_direct/log_bi.txt', level=logging.INFO, format='%(message)s')


# Calculate a Box, which is a tuple of normalized costs and benefits
def cal_box(path, epsilon, c_min, b_max):
    box = []

    # Costs
    for i in range(len(path["costs"])):
        box.append(math.floor(math.log(path["costs"][i] / c_min[i], 1 + epsilon[i])))

    # Benefits
    for i in range(len(path["benefits"])):
        box.append(math.floor(math.log(path["benefits"][i] / b_max[i], 1 - epsilon[i + len(path["costs"])])))

    return tuple(box)


# Check is path p2 dominates path p1
def is_dominated(b1, b2, p1, p2, epsilon):
    b1 = ast.literal_eval(b1)
    b2 = ast.literal_eval(b2)

    # Case 1: Same box, compare the last benefit
    if b1 == b2:
        return p1['benefits'][-1] < p2['benefits'][-1]

    # Case 2: Different boxes, compare the box
    for i in range(len(b1)):
        if b2[i] > (1 + epsilon[i]) * b1[i]:
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

                dominated = False
                for sb in sandwich_bounds:
                    if is_dominated(box, sb[1], new_path, pareto_set.get(sb[1], None), epsilon):
                        dominated = True
                        break
                if not dominated:
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

                dominated = False
                for sb in sandwich_bounds:
                    if is_dominated(box, sb[0], new_path, pareto_set.get(sb[0], None), epsilon):
                        dominated = True
                        break
                if not dominated:
                    pareto_set, added, prun_set = update_pareto(pareto_set, box, new_path, epsilon, prun_set)
                    if added:
                        new_pathsF[box] = new_path

        # Terminate if the two searches meet
        if visitedF & visitedB:
            break

        # Update sandwich bounds based on the updated Pareto set
        for boxF, pathF in new_pathsF.items():
            for boxB, pathB in new_pathsB.items():
                if is_dominated(boxF, boxB, pathF, pathB, epsilon) or is_dominated(boxB, boxF, pathB, pathF, epsilon):
                    sandwich_bounds.add((boxF, boxB))

        pathsF = new_pathsF
        pathsB = new_pathsB

    return pareto_set


def main():
    G = pickle.load(open(dataset + 'costs.gpickle', 'rb'))

    start_node = 0
    end_node = 37653
    epsilon = [0.1] * 5 + [0.0001]
    c_min, b_max = get_cmin_bmax(G)
    pareto_set = {}

    start_time = time.time()
    pareto = bi_directional_search(G, pareto_set, start_node, end_node, epsilon, c_min, b_max)
    end_time = time.time()
    logging.info(f"epsilon: {epsilon}")
    logging.info(f"Search time: {end_time - start_time}")
    logging.info(f"Pareto set size: {len(pareto)}")
    pareto_json = json.dumps(pareto, indent=4)
    with open(dataset + 'bi_direct/pareto.json', 'w') as json_file:
        json_file.write(pareto_json)


if __name__ == '__main__':
    main()
