import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")
import Trainer.avocado_linear_regression as alr

data = pd.read_csv('./extra/avocado_full.csv')
target = 'AveragePrice'
min_clusters = 10
max_clusters = 20


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


data = data.fillna(data.mean())
data = alr.pre_processing(data)
clusters, labels = optimal_kmeans(data, max_clusters)
data['cluster'] = labels
print(clusters) #(10, 12)

data.to_csv('house_clustered.csv', index=False)