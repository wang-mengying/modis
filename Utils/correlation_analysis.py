import json
import pandas as pd

def active_counts(label_str):
    label_tuple = eval(label_str)
    active_items = label_tuple[0].count(1)
    active_clusters = label_tuple[1].count(1)
    return active_items, active_clusters


def preprocessing_movie(json_file):
    with open(json_file, 'r') as f:
        nodes_data = json.load(f)

    active_items_list, active_clusters_list = zip(*[active_counts(node_info['Label']) for node_info in nodes_data.values()])
    data = {
        'time': [node_info['model_objectives'][0] for node_info in nodes_data.values()],
        'accuracy': [node_info['model_objectives'][1] for node_info in nodes_data.values()],
        'complexity': [node_info['model_objectives'][2] for node_info in nodes_data.values()],
        'fisher': [node_info['feature_objectives'][0] for node_info in nodes_data.values()],
        'mutual_info': [node_info['feature_objectives'][1] for node_info in nodes_data.values()],
        'vif': [node_info['feature_objectives'][2] for node_info in nodes_data.values()],
        'active_items': active_items_list,
        'active_clusters': active_clusters_list
    }
    df = pd.DataFrame(data)

    return df


def preprocessing_avocado(json_file):
    with open(json_file, 'r') as f:
        nodes_data = json.load(f)

    active_items_list, active_clusters_list = zip(*[active_counts(node_info['Label']) for node_info in nodes_data.values()])
    data = {
        'mse': [node_info['model_objectives'][0] for node_info in nodes_data.values()],
        'mae': [node_info['model_objectives'][1] for node_info in nodes_data.values()],
        'time': [node_info['model_objectives'][2] for node_info in nodes_data.values()],
        'active_items': active_items_list,
        'active_clusters': active_clusters_list
    }
    df = pd.DataFrame(data)

    return df


def preprocessing_house(json_file):
    with open(json_file, 'r') as f:
        nodes_data = json.load(f)

    active_items_list, active_clusters_list = zip(*[active_counts(node_info['Label']) for node_info in nodes_data.values()])
    data = {
        'accuracy': [node_info['model_objectives'][0] for node_info in nodes_data.values()],
        'f1': [node_info['model_objectives'][1] for node_info in nodes_data.values()],
        'time': [node_info['model_objectives'][2] for node_info in nodes_data.values()],
        'fisher': [node_info['feature_objectives'][0] for node_info in nodes_data.values()],
        'mutual_info': [node_info['feature_objectives'][1] for node_info in nodes_data.values()],
        'active_items': active_items_list,
        'active_clusters': active_clusters_list
    }
    df = pd.DataFrame(data)

    return df


def get_relations(json_file, task, threshold=0.75):
    if task == 'movie':
        df = preprocessing_movie(json_file)
    elif task == 'avocado':
        df = preprocessing_avocado(json_file)
    elif task == 'house':
        df = preprocessing_house(json_file)
    elif task == 'modsnet':
        data = pd.read_csv(json_file)
        numeric_columns = data.select_dtypes(include=['number']).columns
        df = data[numeric_columns]

    objectives = df.columns
    correlation_matrix = df.corr(method='spearman')
    strong_relations = {}

    for i, obj1 in enumerate(objectives):
        for j, obj2 in enumerate(objectives):
            if i < j and abs(correlation_matrix[obj1][obj2]) > threshold:
                relation_type = "P" if correlation_matrix[obj1][obj2] > 0 else "N"
                strong_relations[(obj1, obj2)] = relation_type

    return strong_relations


def main():
    # json_file = '../Dataset/Kaggle/others/d7m8/nodes.json'
    # json_file = '../Dataset/HuggingFace/results/ml2/nodes.json'
    # json_file = '../Dataset/OpenData/House/results/ml6/nodes.json'
    json_file = '../Surrogate/ModsNet/sample_nodes.csv'
    json_file = json_file.replace('/', '\\')

    strong_relations = get_relations(json_file, 'modsnet', threshold=0.93)
    print(strong_relations)


if __name__ == '__main__':
    main()

