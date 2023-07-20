import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
import json

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('../processed/movie_filtered.csv')
target = 'worldwide_gross'
min_clusters = 10
max_clusters = 30


def optimal_kmeans(feature, max_clusters):
    best_score = -1
    best_clusters = 0
    best_labels = None

    for clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=clusters, random_state=123)
        kmeans.fit(feature)
        labels = kmeans.labels_
        score = silhouette_score(feature, labels)

        if score > best_score:
            best_score, best_clusters, best_labels = score, clusters, labels

    return best_clusters, best_labels


# Numerical features, directly K-Means
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

# 'director_birthYear' and 'director_deathYear'
data['director_birthYear'] = pd.to_numeric(data['director_birthYear'], errors='coerce')
data['director_deathYear'] = pd.to_numeric(data['director_deathYear'], errors='coerce')
data['director_birthYear'] = data['director_birthYear'].fillna(data['director_birthYear'].mean())
data['director_deathYear'] = data['director_deathYear'].fillna(2023)


# 'genres', split and count the frequency, then one hot encoding and K-Means
genres = data['genres'].str.split(',', expand=True).stack().value_counts()
data['genres'] = data['genres'].apply(lambda x: sum([genres[g] for g in x.split(',')]))


# 'movie_title', count the number of words
data['movie_title'] = data['movie_title'].apply(lambda x: len(x.split()))

# 'director_name', frequency counts and the first letter
director_name_df = data[['director_name']].copy()
director_name_df['director_name_freq'] = director_name_df.groupby('director_name')['director_name'].transform('count')
director_name_df['director_name_letter'] = director_name_df['director_name'].apply(lambda x: x[0].lower())
director_name_df['director_name_letter'] = LabelEncoder().fit_transform(director_name_df['director_name_letter'])

director_name_df['director_name'] = director_name_df['director_name_freq'] + director_name_df['director_name_letter']
data['director_name'] = director_name_df['director_name']

data = data.fillna(data.mean())
clusters, labels = optimal_kmeans(data, max_clusters)
data['cluster'] = labels
print(clusters)

data.to_csv('movie_clustered_table.csv', index=False)