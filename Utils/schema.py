import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# Same_name(col1, col2) if col1 == col2 or col1 is part of col2, vice versa
def same_name(col1, col2):
    same = (col1.lower() in col2.lower()) or (col1.lower() in col2.lower())

    return same


# col1 equals col2: same_name(col1, col2) and overlap by at least 0.3.
def col_equal(df1, df2, col1, col2, t_overlap=0.1):
    if not same_name(col1, col2):
        return False

    val1 = df1[col1].dropna().unique()
    val2 = df2[col2].dropna().unique()
    overlap = len(set(val1) & set(val2)) / len(set(val1))

    equal = overlap > t_overlap

    return equal


# Merge df_original and df_extra with conditions
def merge_table(df_original, df_extra, target_column):
    include_target = False
    for col in df_extra.columns:
        if same_name(target_column, col):
            include_target = True
            break

    merge_keys = list(set(df_original.columns.tolist()) & set(df_extra.columns.tolist()))
    if include_target:
        df_original = pd.merge(df_original, df_extra, how='outer', on=merge_keys)
    else:
        for col_o in df_original.select_dtypes(include=['object', 'category']).columns:
            for col_e in df_extra.select_dtypes(include=['object', 'category']).columns:
                if col_equal(df_original, df_extra, col_o, col_e, 0.7):
                    df_original = pd.merge(df_original, df_extra, how='outer', on=merge_keys)
                break

    # Remove rows whose target_column is null.
    df_original.dropna(axis=0, subset=[str(target_column)], inplace=True)

    return df_original


def row_clean(df, thr_null=0.5, iqr_factor=1.5):
    # Remove rows with more than 50%(thr_null) missing values.
    df_cleaned = df.dropna(thresh=df.shape[1] * thr_null, inplace=False)

    # Remove rows with outliers by IQR method
    Q1 = df_cleaned.quantile(0.25)
    Q3 = df_cleaned.quantile(0.75)
    IQR = Q3 - Q1
    df_cleaned = df_cleaned[~((df_cleaned < (Q1 - iqr_factor * IQR)) |
                              (df_cleaned > (Q3 + iqr_factor * IQR))).any(axis=1)]

    return df_cleaned


def main():
    dataset = '../Dataset/Movie/'
    original_path = dataset + 'movie_original.csv'
    extra_path = dataset + 'extra'
    df_original = pd.read_csv(original_path)

    # Table-level Merge
    for file in os.listdir(extra_path):
        if file.endswith('.csv'):
            file_path = os.path.join(extra_path, file)
            df_extra = pd.read_csv(file_path)

            df_original = merge_table(df_original, df_extra, "worldwide_gross")
    df_original.to_csv(dataset + 'processed/movie_merged.csv', index=False)

    # Row-level Clean
    df_cleaned = row_clean(df_original, thr_null=0.5, iqr_factor=2)
    df_cleaned.to_csv(dataset + 'processed/movie_cleaned.csv', index=False)


if __name__ == '__main__':
    main()
