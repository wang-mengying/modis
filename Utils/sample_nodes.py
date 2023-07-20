import pandas as pd
import numpy as np


# Count of active items/values
def extract_counts(label):
    items, values = eval(label)
    return sum(items), sum(values)


def get_sample(df, n_samples=300):
    df[['active_items', 'active_values']] = df['Label'].apply(extract_counts).apply(pd.Series)
    df['stratum'] = df['active_items'].astype(str) + '-' + df['active_values'].astype(str)

    sample = df.groupby('stratum').apply(lambda x: x.sample(int(np.rint(n_samples*len(x)/len(df))), random_state=1)).reset_index(drop=True)

    # If get less than 300 samples, sample randomly from the remaining nodes
    if len(sample) < n_samples:
        remaining = df.loc[~df.index.isin(sample.index)]
        additional_sample = remaining.sample(n=n_samples-len(sample), random_state=1)
        sample = pd.concat([sample, additional_sample]).reset_index(drop=True)

    return sample


def main():
    path = "../Dataset/Movie/others/"
    df = pd.read_csv(path + 'nodes.csv')

    sample = get_sample(df)
    sample.to_csv(path + 'sample_nodes.csv', index=False)


if __name__ == '__main__':
    main()