import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from Trainer.school_logistic_regression import preprocess_data_with_mapper as preprocess_data


import warnings
warnings.filterwarnings("ignore")

school_data = pd.read_csv('./processed/school_filtered.csv')
target = 'class'
min_clusters = 12
max_clusters = 30


def optimal_kmeans(feature, max_clusters):
    best_score = -1
    best_clusters = 0
    best_labels = None

    for clusters in range(min_clusters, max_clusters + 1):
        print('Trying {} clusters'.format(clusters))
        kmeans = KMeans(n_clusters=clusters, random_state=123)
        kmeans.fit(feature)
        labels = kmeans.labels_
        score = silhouette_score(feature, labels)

        if score > best_score:
            best_score, best_clusters, best_labels = score, clusters, labels

    return best_clusters, best_labels


def transform_data_to_dataframe(df, preprocessor):
    # Fit and transform the data
    transformed_data = preprocessor.fit_transform(df)

    # Getting new column names after one-hot encoding
    # The preprocessor is a ColumnTransformer with transformers for numerical and categorical data
    num_cols = preprocessor.named_transformers_['num'].get_feature_names_out()
    cat_cols = preprocessor.named_transformers_['cat'].get_feature_names_out()

    # Combine all column names
    new_columns = list(num_cols) + list(cat_cols)

    # Create a DataFrame with the transformed data
    transformed_df = pd.DataFrame(transformed_data, columns=new_columns)

    return transformed_df


numeric_columns = school_data.select_dtypes(include=['int64', 'float64']).columns
data_processed = preprocess_data(school_data)
# data_processed = transform_data_to_dataframe(school_data, data_processed)
# data_notarget = data_processed.drop(target, axis=1)
clusters, labels = optimal_kmeans(data_processed, max_clusters)
print(clusters)

data = pd.read_csv('./processed/school_filtered.csv')
data['cluster'] = labels

data.to_csv('./processed/school_clustered.csv', index=False)