import os
import pandas as pd
import sys
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
from sklearn.preprocessing import LabelEncoder

sys.path.append("../")
import Trainer.movie_gradient_boosting as mgb
import Trainer.avocado_linear_regression as alr
import Trainer.school_logistic_regression as slr
import Trainer.house_random_forest as hrf

import warnings
warnings.filterwarnings("ignore")


def same_name(col1, col2, t_overlap=0.9):
    col1 = col1.lower()
    col2 = col2.lower()

    similar = string_overlap(col1, col2) > t_overlap
    # (col1 in col2) or (col2 in col1)

    return similar


def longest_common_subsequence(s1, s2):
    if len(s1) == 0 or len(s2) == 0:
        return 0

    dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]

    for i in range(len(s1) + 1):
        for j in range(len(s2) + 1):
            if i == 0 or j == 0:
                dp[i][j] = 0
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[len(s1)][len(s2)]


def string_overlap(s1, s2):
    s1 = s1.lower()
    s2 = s2.lower()
    lcs_length = longest_common_subsequence(s1, s2)
    max_length = max(len(s1), len(s2))
    return lcs_length / max_length


def col_equal(df1, df2, col1, col2, t_overlap=0.1):
    # if df1[col1].dtype != df2[col2].dtype:
    #     return False
    if not same_name(col1, col2):
        return False

    val1 = df1[col1].dropna().unique()
    val2 = df2[col2].dropna().unique()

    if len(val1) == 0 or len(val2) == 0:
        return False

    overlap = len(set(val1) & set(val2)) / len(set(val1))
    equal = overlap > t_overlap

    return equal


def merge_table(df_original, df_extra, target_column):
    include_target = False
    for col in df_extra.columns:
        if col_equal(df_original, df_extra, target_column, col, 0.8):
            include_target = True
            break

    merge_keys = list(set(df_original.columns.tolist()) & set(df_extra.columns.tolist()))
    if include_target:
        df_original = pd.merge(df_original, df_extra, how='outer', on=merge_keys)
    else:
        for col_o in df_original.select_dtypes(include=['object', 'category']).columns:
            for col_e in df_extra.select_dtypes(include=['object', 'category']).columns:
                if col_equal(df_original, df_extra, col_o, col_e, 0.7) and df_original[col_o].dtype == df_extra[col_e].dtype:
                    try:
                        df_original = pd.merge(df_original, df_extra, how='outer', on=merge_keys)
                    except:
                        df_original = pd.merge(df_original, df_extra, how='outer', left_on=col_o, right_on=col_e)
                break

    # Remove rows with missing target values
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


def col_numeric(X, y, percentile=90):
    selector_f_classif = SelectPercentile(f_classif, percentile=percentile)
    selector_f_classif.fit_transform(X, y)

    num_cols = X.columns[selector_f_classif.get_support()]

    return num_cols


def col_categorical(X, y, percentile=80):
    cat_encoded = X.apply(LabelEncoder().fit_transform)

    selector_chi2 = SelectPercentile(chi2, percentile=percentile)
    selector_chi2.fit_transform(cat_encoded, y)

    cat_cols = X.columns[selector_chi2.get_support()]

    return cat_cols


def movie(dataset):
    original_path = dataset + 'processed/movie_original.csv'
    extra_path = dataset + 'extra'
    df_original = pd.read_csv(original_path)

    # Table-level Merge
    for file in os.listdir(extra_path):
        if file.endswith('.csv'):
            file_path = os.path.join(extra_path, file)
            print(file_path)
            df_extra = pd.read_csv(file_path, encoding_errors='ignore', error_bad_lines=False)

            df_original = merge_table(df_original, df_extra, "worldwide_gross")
    df_original.to_csv(dataset + 'processed/movie_merged.csv', index=False)

    # Row-level Clean
    df_cleaned = row_clean(df_original, thr_null=0.5, iqr_factor=2)
    df_cleaned.to_csv(dataset + 'processed/movie_cleaned.csv', index=False)

    # Column-level Filter
    df = pd.read_csv(dataset + 'processed/movie_cleaned.csv')
    categorical_cols = ['genres', 'director_professions', 'movie_title', 'director_name']
    df_num = mgb.preprocess_data(df.drop(categorical_cols, axis=1))

    X_num = df_num.drop(['worldwide_gross', 'gross_class'], axis=1)
    X_cat = df_cleaned[categorical_cols]
    y = df_num['gross_class']

    num_cols = col_numeric(X_num, y)
    cat_cols = col_categorical(X_cat, y)

    df_filtered = pd.concat([pd.DataFrame(df, columns=num_cols),
                             pd.DataFrame(df, columns=cat_cols),
                             pd.DataFrame(df, columns=['worldwide_gross']), ], axis=1)
    df_filtered.to_csv(dataset + 'processed/movie_filtered.csv', index=False)


def avocado(dataset):
    original_path = dataset + 'housing.csv'
    extra_path = dataset + 'extra'
    df_original = pd.read_csv(original_path)

    # Table-level Merge
    for file in os.listdir(extra_path):
        if file.endswith('.csv'):
            file_path = os.path.join(extra_path, file)
            print(file_path)
            df_extra = pd.read_csv(file_path, encoding_errors='ignore', error_bad_lines=False)

            df_original = merge_table(df_original, df_extra, "AveragePrice")
    df_original.to_csv(dataset + 'processed/avocado_merged.csv', index=False)

    # Row-level Clean
    df_cleaned = row_clean(df_original, thr_null=0.5, iqr_factor=2)
    df_cleaned.to_csv(dataset + 'processed/avocado_cleaned.csv', index=False)

    # Column-level Filter
    df = pd.read_csv(dataset + 'processed/avocado_cleaned.csv')
    # categorical_cols = ['type', 'region']
    # df_num = alr.pre_processing(df.drop(categorical_cols, axis=1))
    #
    # X_num = df_num.drop(['AveragePrice'], axis=1)
    # X_cat = df_cleaned[categorical_cols]
    # y = df_num['AveragePrice']
    #
    # num_cols = col_numeric(X_num, y)
    # cat_cols = col_categorical(X_cat, y)
    #
    # df_filtered = pd.concat([pd.DataFrame(df, columns=num_cols),
    #                             pd.DataFrame(df, columns=cat_cols),
    #                             pd.DataFrame(df, columns=['AveragePrice']), ], axis=1)
    df.to_csv(dataset + 'processed/avocado_filtered.csv', index=False)


def school(dataset):
    original_path = dataset + 'School/processed/school_original.csv'
    extra_path = dataset + 'extra'
    df_original = pd.read_csv(original_path)

    # Table-level Merge
    for file in os.listdir(extra_path):
        if file.endswith('.csv'):
            file_path = os.path.join(extra_path, file)
            print(file_path)
            df_extra = pd.read_csv(file_path, encoding_errors='ignore', error_bad_lines=False)
            df_original = merge_table(df_original, df_extra, "class")
    df_original.to_csv(dataset + 'School/processed/school_filtered.csv', index=False)


def house(dataset):
    original_path = dataset + 'House/processed/house_original.csv'
    extra_path = dataset + 'extra'
    df_original = pd.read_csv(original_path)

    # # Row-level Clean
    # df_cleaned = row_clean(df_original, thr_null=0.5, iqr_factor=2)
    # df_cleaned.to_csv(dataset + 'House/processed/house_cleaned.csv', index=False)
    #
    # # Column-level Filter
    # df = pd.read_csv(dataset + 'House/processed/house_cleaned.csv')
    cols_to_drop = ['SOLD DATE', 'Unnamed: 7', 'NEXT OPEN HOUSE START TIME', 'NEXT OPEN HOUSE END TIME', 'ADDRESS', 'URL (SEE https://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)',
                    'MLS#']
    df = df_original.drop(columns=cols_to_drop, errors='ignore')
    df.to_csv(dataset + 'House/processed/house_filtered.csv', index=False)


def main():
    topic = "house"

    if topic == "movie":
        dataset = "../Dataset/Kaggle/"
        movie(dataset)
    elif topic == "avocado":
        dataset = "../Dataset/HuggingFace/"
        avocado(dataset)
    elif topic == "school":
        dataset = "../Dataset/OpenData/"
        school(dataset)
    elif topic == "house":
        dataset = "../Dataset/OpenData/"
        house(dataset)


if __name__ == '__main__':
    main()
