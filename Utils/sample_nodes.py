import pandas as pd
import numpy as np


# Count of active items/values
def extract_counts(label):
    items, values = eval(label)
    return sum(items), sum(values)


def get_sample(df, n_samples=300):
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


def process_data(node_label, node_id, original_file, clustered_file, path):
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

    df.to_csv(f'{path}{node_id}.csv', index=False)


def process_all(df_sample, original_file, clustered_file, path):
    for _, row in df_sample.iterrows():
        node_id = row['Id']
        node_label = row['Label']
        process_data(node_label, node_id, original_file, clustered_file, path)

    print("Processing complete.")


def main():
    path = "../Dataset/Movie/"
    df = pd.read_csv(path + 'others/d7m7/nodes.csv')

    sample = get_sample(df)
    sample.to_csv(path + 'others/sample_nodes.csv', index=False)

    # label = "((1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1), (0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0))"
    # id = 1575
    process_all(sample, path + 'processed/movie_filtered.csv',
                path + 'others/movie_clustered_table.csv',
                path + 'others/sample/')


if __name__ == '__main__':
    main()
