import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Load data from node.txt and edge.txt
node_file = "./graph_split/node.txt"
edge_file = "./graph_split/edge.txt"

# Read node data
nodes = pd.read_csv(node_file, sep="\t", header=None, names=["id", "type", "features"])
nodes["features"] = nodes["features"].apply(eval)

# Read edge data
edges = pd.read_csv(edge_file, sep="\t", header=None, names=["connection", "metrics"])
edges["metrics"] = edges["metrics"].apply(eval)

# Create a mapping of node id to features
node_feature_map = nodes.set_index("id")["features"].to_dict()

# Combine node features into edge features
def combine_node_features(connection, metrics):
    node1, node2 = eval(connection)
    features1 = node_feature_map.get(node1, [0] * 17)
    features2 = node_feature_map.get(node2, [0] * 17)
    return features1 + features2 + metrics

edges["combined_features"] = edges.apply(lambda row: combine_node_features(row["connection"], row["metrics"]), axis=1)

# Extract combined features for clustering
features = edges["combined_features"].tolist()
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine the optimal number of clusters
best_num_clusters = 10
best_score = -1

for num_clusters in range(10, 26):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    score = silhouette_score(scaled_features, clusters)
    print(f"Number of clusters: {num_clusters}, Silhouette Score: {score}")
    if score > best_score:
        best_num_clusters = num_clusters
        best_score = score

print(f"Optimal number of clusters: {best_num_clusters}")

# Perform clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=best_num_clusters, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# Add cluster labels to edges
edges["cluster"] = clusters

# Save the updated edges with cluster labels
output_file = "./processed/graph_clustered.txt"
edges[["connection", "cluster"]].to_csv(output_file, sep="\t", index=False, header=False)

print(f"Updated edges with connection and clusters saved to {output_file}.")
