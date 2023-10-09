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
min_clusters = 5
max_clusters = 25
cluster_k = {}


def optimal_kmeans(feature, min_clusters, max_clusters):
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


def clustering(data, col):
    clusters, labels = optimal_kmeans(data[[col, target]], min_clusters, max_clusters)
    data[col] = labels
    cluster_k[col] = clusters


# Numerical features, directly K-Means
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

numerical_cols.remove(target)

for col in numerical_cols:
    data[col] = data[col].fillna(data[col].mean())
    clustering(data, col)

# 'director_birthYear' and 'director_deathYear'
data['director_birthYear'] = pd.to_numeric(data['director_birthYear'], errors='coerce')
data['director_deathYear'] = pd.to_numeric(data['director_deathYear'], errors='coerce')
data['director_birthYear'] = data['director_birthYear'].fillna(data['director_birthYear'].mean())
data['director_deathYear'] = data['director_deathYear'].fillna(2023)

clustering(data, 'director_birthYear')
clustering(data, 'director_deathYear')

# 'genres', split and count the frequency, then one hot encoding and K-Means
genres = data['genres'].str.split(',', expand=True).stack().value_counts()
data['genres'] = data['genres'].apply(lambda x: ', '.join(sorted([g for g in x.split(',') if genres[g] > 5])))

vectorizer = CountVectorizer(tokenizer=lambda x: x.split(', '))
genres_encoded = vectorizer.fit_transform(data['genres'])
genres_encoded_df = pd.DataFrame(genres_encoded.toarray(), columns=vectorizer.get_feature_names_out())
concatenated_df = pd.concat([genres_encoded_df, data['worldwide_gross']], axis=1)
clusters, labels = optimal_kmeans(concatenated_df, min_clusters, max_clusters)
data['genres'] = labels
cluster_k['genres'] = clusters

# 'movie_title', count the number of words
data['movie_title'] = data['movie_title'].apply(lambda x: len(x.split()))
clustering(data, 'movie_title')

# 'director_name', frequency counts and the first letter
director_name_df = data[['director_name']].copy()
director_name_df['director_name_freq'] = director_name_df.groupby('director_name')['director_name'].transform('count')
director_name_df['director_name_letter'] = director_name_df['director_name'].apply(lambda x: x[0].lower())
director_name_df['director_name_letter'] = LabelEncoder().fit_transform(director_name_df['director_name_letter'])

director_name_df['director_name'] = director_name_df['director_name_freq'] + director_name_df['director_name_letter']
clustering(pd.concat([director_name_df, data[target]], axis=1), 'director_name')

data['director_name'] = director_name_df['director_name']

# Save the processed result
data.to_csv('movie_clustered.csv', index=False)

with open('cluster_k.json', 'w') as json_file:
    json.dump(cluster_k, json_file)
