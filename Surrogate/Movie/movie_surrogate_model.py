from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import joblib


# Transform 'Label' into separate binary feature columns
def transform_label(row):
    feature_states, cluster_states = eval(row['Label'])

    states = {}
    for i, state in enumerate(feature_states):
        states[f'feature_{i + 1}'] = state
    for i, state in enumerate(cluster_states):
        states[f'cluster_{i + 1}'] = state

    return pd.Series(states)


def get_X_y(df):
    label_data = df.apply(transform_label, axis=1)
    df = pd.concat([df, label_data], axis=1)

    df.drop(columns=['Label', 'stratum'], inplace=True)

    feature_cols = [col for col in df.columns if col not in ['Id', 'training_time', 'accuracy', 'complexity']]
    target_cols = ['training_time', 'accuracy', 'complexity']

    X = df[feature_cols]
    y = df[target_cols]

    # complexity = node/active_items
    y['relative_complexity'] = y['complexity'] / X['active_items']
    y['complexity'] = y['relative_complexity']
    y.drop('relative_complexity', axis=1, inplace=True)

    return X, y


def train_pred(X, y, model_path="movie_surrogate.joblib"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Multi-output Gradient Boosting model
    model = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)
    # model = joblib.load(model_path)

    y_pred = model.predict(X_test)

    metric = {'mse_time': mean_squared_error(y_test['training_time'], y_pred[:, 0]),
              'mse_accuracy': mean_squared_error(y_test['accuracy'], y_pred[:, 1]),
              'mse_complexity': mean_squared_error(y_test['complexity'], y_pred[:, 2]),
              'r2_time': r2_score(y_test['training_time'], y_pred[:, 0]),
              'r2_accuracy': r2_score(y_test['accuracy'], y_pred[:, 1]),
              'r2_complexity': r2_score(y_test['complexity'], y_pred[:, 2]),
              'mape_time': np.mean(np.abs((y_test['training_time'] - y_pred[:, 0]) / y_test['training_time'])) * 100,
              'mape_accuracy': np.mean(np.abs((y_test['accuracy'] - y_pred[:, 1]) / y_test['accuracy'])) * 100,
              "mape_complexity": np.mean(np.abs((y_test['complexity'] - y_pred[:, 2]) / y_test['complexity'])) * 100}

    return metric


def main():
    data = pd.read_csv('sample_nodes.csv')

    X, y = get_X_y(data)
    metrics = train_pred(X, y)

    for key, value in metrics.items():
        print(f"{key}: {value}")

    # mse_time: 0.01365497143443009
    # mse_accuracy: 0.0003146473320577315
    # mse_complexity: 46.49592482366314
    # r2_time: 0.798556067583752
    # r2_accuracy: 0.9626916950332823
    # r2_complexity: 0.9859198659682977
    # mape_time: 6.344140837105562
    # mape_accuracy: 1.7656642057062257
    # mape_complexity: 0.885800016908306


if __name__ == '__main__':
    main()
