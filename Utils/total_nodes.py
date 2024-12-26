import math
from scipy.special import comb
from itertools import combinations


def igraph_add(L):
    total = 1
    for t_i in L:
        subset_count = sum(comb(t_i, k) for k in range(t_i - math.ceil(t_i / 10), t_i))
        total *= subset_count + 2
    return int(total)


def igraph_drop(L):
    n = len(L)
    total = 0
    max_drop_items = math.ceil(n - 0.7 * n)
    for i in range(n - max_drop_items, n+1):
        for active_items in combinations(range(n), i):
            num_combinations = 1
            for item in active_items:
                t = L[item]
                num_values = sum(comb(t, j, exact=True) for j in range(t - math.ceil(t/10), t + 1))
                num_combinations *= num_values
            total += num_combinations
    return total


def igraph_table(L, k, d, m):
    n = len(L)

    # states from "drop"
    drop_states = sum(comb(n, i) for i in range(math.floor(d * n), n + 1))
    print(math.floor(d * n), n + 1)

    # states from "modify"
    modify_states = sum(comb(k, i) for i in range(math.ceil(m * k), k + 1))
    print(math.ceil(m * k), k + 1)

    total = drop_states * modify_states

    return int(total)


schema = [13] * 10
cluster = 13

print(igraph_table(schema, cluster, 0.7, 0.7))

# 2824752490, 2792361600, 130384