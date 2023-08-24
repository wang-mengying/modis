import sys

import joblib
import pandas as pd
import numpy as np

sys.path.append("../")
import Utils.sample_nodes as sample
import Utils.objectives as objectives
import Trainer.movie_gradient_boosting as mgb


def count_cluster(file_path):
    df = pd.read_csv(file_path)
    cluster_counts = df['cluster'].value_counts().sort_index()
    cluster_counts = cluster_counts.to_dict()

    return cluster_counts


def surrogate_inputs(node, file_path):
    features, clusters = eval(node['Label'])

    active_items = features.count(1)
    active_values = features.count(1)

    clusters_counts = count_cluster(file_path)
    num_rows = sum([clusters_counts[i] for i, value in enumerate(clusters) if value == 1])

    data = {
        'active_items': [active_items],
        'active_values': [active_values],
        'num_rows': [num_rows],
        'num_cols': [active_items + 1],
    }

    for i, state in enumerate(features):
        data[f'feature_{i + 1}'] = state
    for i, state in enumerate(clusters):
        data[f'cluster_{i + 1}'] = state

    df = pd.DataFrame(data)

    return df


def feature_objectives(node, clustered_file):
    df = sample.process_data(node['Label'], clustered_file, clustered_file)

    bins = [0, 50000000, 150000000, np.inf]
    labels = ['Low', 'Medium', 'High']
    df['gross_class'] = pd.cut(df['worldwide_gross'], bins=bins, labels=labels)

    X = df.drop(['worldwide_gross', 'gross_class', 'cluster'], axis=1)

    fisher = objectives.fisher_score(X, df['gross_class'])
    mi = objectives.mutual_info(X, df['worldwide_gross'])
    vif = objectives.vif(X)

    return [fisher, mi, vif]


def main():
    node = {'Id': 0, 'Label': '((1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1), (1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0))', 'stratum': 0}
    file_path = 'movie_clustered_table.csv'
    df = surrogate_inputs(node, file_path)

    model_path = '../../../Surrogate/Movie/movie_surrogate.joblib'
    model = joblib.load(model_path)

    model_objs = model.predict(df)[0]
    print(model_objs)
    feature_objs = feature_objectives(node, 'movie_clustered_table.csv')
    print(feature_objs)


if __name__ == '__main__':
    main()
