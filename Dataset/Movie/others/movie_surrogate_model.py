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

    feature_cols = [col for col in df.columns if col not in ['Id', 'time', 'accuracy']]
    target_cols = ['time', 'accuracy']

    X = df[feature_cols]
    y = df[target_cols]

    return X, y


def train_pred(X, y, model_path="movie_surrogate.joblib"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Multi-output Gradient Boosting model
    model = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)
    # model = joblib.load(model_path)

    y_pred = model.predict(X_test)

    metric = {'mse_time': mean_squared_error(y_test['time'], y_pred[:, 0]),
              'mse_accuracy': mean_squared_error(y_test['accuracy'], y_pred[:, 1]),
              'r2_time': r2_score(y_test['time'], y_pred[:, 0]),
              'r2_accuracy': r2_score(y_test['accuracy'], y_pred[:, 1]),
              'mape_time': np.mean(np.abs((y_test['time'] - y_pred[:, 0]) / y_test['time'])) * 100,
              'mape_accuracy': np.mean(np.abs((y_test['accuracy'] - y_pred[:, 1]) / y_test['accuracy'])) * 100}

    return metric


def main():
    data = pd.read_csv('sample_nodes.csv')

    X, y = get_X_y(data)
    metrics = train_pred(X, y)

    for key, value in metrics.items():
        print(f"{key}: {value}")

    # mse_time: 0.013112148344765815
    # mse_accuracy: 0.0003146473320577315
    # r2_time: 0.8115984219329653
    # r2_accuracy: 0.9626916950332823
    # mape_time: 6.6159298918394835
    # mape_accuracy: 1.7656642057062257


if __name__ == '__main__':
    main()
