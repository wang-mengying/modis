import sys
import pandas as pd
import numpy as np

from Trainer import house_random_forest

sys.path.append("../")
import Trainer.movie_gradient_boosting as mgb
import Dataset.Kaggle.others.movie_objectives as movie_objectives
import Trainer.avocado_linear_regression as alr
import Trainer.school_logistic_regression as slr

import warnings
warnings.filterwarnings("ignore")


def extract_counts(label):
    items, values = eval(label)
    return sum(items), sum(values)


def get_sample(df, n_samples=10000):
    df[['active_items', 'active_values']] = df['Label'].apply(extract_counts).apply(pd.Series)
    df['stratum'] = df['active_items'].astype(str) + '-' + df['active_values'].astype(str)

    sample = df.groupby('stratum').apply(
        lambda x: x.sample(int(np.rint(n_samples * len(x) / len(df))), random_state=1)).reset_index(drop=True)

    # If get less than 300 samples, sample randomly from the remaining nodes
    if len(sample) < n_samples:
        remaining = df.loc[~df.index.isin(sample.index)]
        additional_sample = remaining.sample(n=n_samples - len(sample), random_state=1)
        sample = pd.concat([sample, additional_sample]).reset_index(drop=True)

    return sample


def process_data(node_label, clustered_file):
    # df_o = pd.read_csv(original_file)
    df_c = pd.read_csv(clustered_file)
    items, values = eval(node_label)

    # Drop columns
    inactive_cols = [i for i, x in enumerate(items) if x == 0]
    df = df_c.drop(df_c.columns[inactive_cols], axis=1, inplace=False)

    # Drop rows
    inactive_rows = [i for i, x in enumerate(values) if x == 0]
    drop_rows = df_c[df_c['cluster'].isin(inactive_rows)].index
    df.drop(drop_rows, inplace=True)
    df.drop(['cluster'], axis=1, inplace=True)

    # df.to_csv(f'{path}{node_id}.csv', index=False)
    return df


def cal_objectives_movie(df_sample, clustered_file):
    df_sample['num_rows'] = 0
    df_sample['num_cols'] = 0

    df_sample['accuracy'] = 0.0
    df_sample['complexity'] = 0.0
    df_sample['training_time'] = 0.0

    df_sample['fisher'] = 0.0
    df_sample['mutual_info'] = 0.0
    df_sample['vif'] = 0.0

    for i, row in df_sample.iterrows():
        node_id = row['Id']
        print(f"Processing node {node_id}.")

        node_label = row['Label']
        df = process_data(node_label, clustered_file)

        df_sample.loc[i, 'num_rows'] = df.shape[0]
        df_sample.loc[i, 'num_cols'] = df.shape[1]

        df = mgb.preprocess_data(df)
        try:
            training_time, accuracy, complexity = mgb.train_and_evaluate_model(df)
        except Exception as e:
            print(f"Error raised. {e}")
            training_time, accuracy, complexity = 0.0, 0.0, 0.0
            continue
        fisher, mutual_info, vif = movie_objectives.feature_objectives(row, clustered_file)

        df_sample.loc[i, 'accuracy'] = accuracy
        df_sample.loc[i, 'complexity'] = complexity/(df_sample.loc[i, 'num_cols'] - 1)
        df_sample.loc[i, 'training_time'] = training_time
        df_sample.loc[i, 'fisher'] = fisher
        df_sample.loc[i, 'mutual_info'] = mutual_info
        df_sample.loc[i, 'vif'] = vif

    return df_sample


def cal_objectives_avocado(df_sample, original_file, clustered_file):
    df_sample['num_rows'] = 0
    df_sample['num_cols'] = 0

    df_sample['mse'] = 0.0
    df_sample['mae'] = 0.0
    df_sample['training_time'] = 0.0

    for i, row in df_sample.iterrows():
        node_id = row['Id']
        print(f"Processing node {node_id}.")

        node_label = row['Label']
        df = process_data(node_label, clustered_file)

        df_sample.loc[i, 'num_rows'] = df.shape[0]
        df_sample.loc[i, 'num_cols'] = df.shape[1]

        df = alr.pre_processing(df)
        mse, mae, training_time = alr.train_test(df)
        # try:
        #     mse, mae, training_time = alr.train_test(df)
        # except Exception as e:
        #     print(f"Error raised. {e}")
        #     mse, mae, training_time = 0.0, 0.0, 0.0
        #     continue

        df_sample.loc[i, 'mse'] = mse
        df_sample.loc[i, 'mae'] = mae
        df_sample.loc[i, 'training_time'] = training_time

    return df_sample


def cal_objectives_house(df_sample, original_file, clustered_file):
    df_sample['num_rows'] = 0
    df_sample['num_cols'] = 0

    df_sample['accuracy'] = 0.0
    df_sample['f1'] = 0.0
    df_sample['training_time'] = 0.0
    df_sample['fisher'] = 0.0
    df_sample['mutual_info'] = 0.0

    for i, row in df_sample.iterrows():
        node_id = row['Id']
        print(f"Processing node {node_id}.")

        node_label = row['Label']
        df = process_data(node_label, clustered_file)

        df_sample.loc[i, 'num_rows'] = df.shape[0]
        df_sample.loc[i, 'num_cols'] = df.shape[1]

        X, y, _ = house_random_forest.process_data(df)

        try:
            accuracy, f1, training_time = house_random_forest.train_and_save_model(X, y)
        except Exception as e:
            print(f"Error raised. {e}")
            accuracy, f1, training_time = 0.0, 0.0, 0.0
            continue

        fisher, mi = house_random_forest.feature_objs(X, y)

        df_sample.loc[i, 'accuracy'] = accuracy
        df_sample.loc[i, 'f1'] = f1
        df_sample.loc[i, 'training_time'] = training_time
        df_sample.loc[i, 'fisher'] = fisher
        df_sample.loc[i, 'mutual_info'] = mi

    return df_sample


def cal_objectives_school(df_sample, original_file, clustered_file):
    df_sample['num_rows'] = 0
    df_sample['num_cols'] = 0

    df_sample['accuracy'] = 0.0
    df_sample['f1'] = 0.0
    df_sample['training_time'] = 0.0

    for i, row in df_sample.iterrows():
        node_id = row['Id']
        print(f"Processing node {node_id}.")

        node_label = row['Label']
        df = process_data(node_label, clustered_file)

        df_sample.loc[i, 'num_rows'] = df.shape[0]
        df_sample.loc[i, 'num_cols'] = df.shape[1]

        preprocessor = slr.preprocess_data_with_mapper(df)
        try:
            f1, accuracy, training_time = slr.train_and_evaluate_model(df)
        except Exception as e:
            print(f"Error raised. {e}")
            f1, accuracy, training_time = 0.0, 0.0, 0.0
            continue

        df_sample.loc[i, 'accuracy'] = accuracy
        df_sample.loc[i, 'f1'] = f1
        df_sample.loc[i, 'training_time'] = training_time

    return df_sample


def movie():
    dataset = "../Dataset/Kaggle/"
    surrogate = "../Surrogate/Kaggle/"
    df = pd.read_csv(dataset + 'others/d7m7/nodes.csv')

    print("Sampling ......")
    sample = get_sample(df)
    sample.to_csv(surrogate + 'sample_nodes.csv', index=False)

    print("Start training ......")
    sample = cal_objectives_movie(sample, dataset + 'processed/movie_filtered.csv',
                            dataset + 'others/movie_clustered_table.csv')

    sample.to_csv(surrogate + 'sample_nodes.csv', index=False)


def avocado():
    dataset = "../Dataset/HuggingFace/"
    surrogate = "../Surrogate/HuggingFace/"
    df = pd.read_csv(dataset + 'results/ml2/nodes.csv')

    print("Sampling ......")
    sample = get_sample(df)
    sample.to_csv(surrogate + 'sample_nodes.csv', index=False)

    print("Start training ......")
    sample = cal_objectives_avocado(sample, dataset + 'extra/avocado_full.csv',
                                    dataset + 'house_clustered.csv')

    sample.to_csv(surrogate + 'sample_nodes.csv', index=False)


def opendata(topic, original_file, clustered_file):
    dataset = "../Dataset/OpenData/" + topic + "/"
    surrogate = "../Surrogate/" + topic + "/"
    df = pd.read_csv(dataset + 'results/ml6/nodes.csv')

    print("Sampling ......")
    sample = get_sample(df)
    sample.to_csv(surrogate + 'sample_nodes.csv', index=False)

    print("Start training ......")
    if topic == "School":
        sample = cal_objectives_school(sample, dataset + original_file, dataset + clustered_file)
    elif topic == "House":
        sample = cal_objectives_house(sample, dataset + original_file, dataset + clustered_file)

    sample.to_csv(surrogate + 'sample_nodes.csv', index=False)


def main():
    topic = "House"
    if topic == "Movie":
        movie()
    elif topic == "Avocado":
        avocado()
    elif topic == "House":
        opendata(topic, "processed/house_filtered.csv", "processed/house_clustered.csv")
    elif topic == "School":
        opendata(topic, "processed/school_filtered.csv", "processed/school_clustered.csv")


if __name__ == '__main__':
    main()
