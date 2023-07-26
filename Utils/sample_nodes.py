import time
import sys
import pandas as pd
import numpy as np

sys.path.append("../")
import Trainer.movie_gradient_boosting as mgb

import warnings
warnings.filterwarnings("ignore")


# Count of active items/values
def extract_counts(label):
    items, values = eval(label)
    return sum(items), sum(values)


def get_sample(df, n_samples=500):
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


def process_data(node_label, original_file, clustered_file):
    df_o = pd.read_csv(original_file)
    df_c = pd.read_csv(clustered_file)

    items, values = eval(node_label)

    # Drop columns
    inactive_cols = [i for i, x in enumerate(items) if x == 0]
    df = df_o.drop(df_o.columns[inactive_cols], axis=1, inplace=False)

    # Drop rows
    inactive_rows = [i for i, x in enumerate(values) if x == 0]
    drop_rows = df_c[df_c['cluster'].isin(inactive_rows)].index
    df.drop(drop_rows, inplace=True)

    # df.to_csv(f'{path}{node_id}.csv', index=False)
    return df


def train_model(df_sample, original_file, clustered_file):
    df_sample['num_rows'] = 0
    df_sample['num_cols'] = 0
    df_sample['accuracy'] = 0.0
    df_sample['time'] = 0.0

    for i, row in df_sample.iterrows():
        node_id = row['Id']
        print(f"Processing node {node_id}.")

        node_label = row['Label']
        df = process_data(node_label, original_file, clustered_file)

        df_sample.loc[i, 'num_rows'] = df.shape[0]
        df_sample.loc[i, 'num_cols'] = df.shape[1]

        df = mgb.preprocess_data(df)
        start_time = time.time()
        accuracy = mgb.train_and_evaluate_model(df)
        running_time = time.time() - start_time

        df_sample.loc[i, 'accuracy'] = accuracy
        df_sample.loc[i, 'time'] = running_time

    return df_sample


def main():
    dataset = "../Dataset/Movie/"
    surrogate = "../Surrogate/Movie/"
    df = pd.read_csv(dataset + 'others/d7m7/nodes.csv')

    print("Sampling ......")
    sample = get_sample(df)
    sample.to_csv(surrogate + 'sample_nodes.csv', index=False)

    print("Start training ......")
    sample = train_model(sample, dataset + 'processed/movie_filtered.csv',
                         dataset + 'others/movie_clustered_table.csv')

    sample.to_csv(surrogate + 'sample_nodes.csv', index=False)


if __name__ == '__main__':
    main()
