import pandas as pd
import numpy as np
import ast

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib

def transform_label(row):
    feature_states, cluster_states = eval(row['Label'])

    states = {}
    for i, state in enumerate(feature_states):
        states[f'feature_{i + 1}'] = state
    for i, state in enumerate(cluster_states):
        states[f'cluster_{i + 1}'] = state

    return pd.Series(states)

def load_and_preprocess_data(filepath):
    # Load the dataset
    data = pd.read_csv(filepath)

    # Transform the label to match the movie_surrogate_model.py format
    label_data = data.apply(transform_label, axis=1)
    data = pd.concat([data, label_data], axis=1)
    data.drop(columns=['Label', 'stratum'], inplace=True)

    return data

def split_data(data):
    feature_cols = [col for col in data.columns if col not in ['Id', 'mse', 'mae', 'training_time']]
    target_cols = ['mse', 'mae', 'training_time']

    X = data[feature_cols]
    y = data[target_cols]

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, min_samples_split=3, min_samples_leaf=2, subsample=0.9, random_state=42))
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse_scores = [mean_squared_error(y_test.iloc[:, i], y_pred[:, i]) for i in range(y_test.shape[1])]
    return mse_scores


def main():
    data = load_and_preprocess_data('sample_nodes.csv')
    X_train, X_test, y_train, y_test = split_data(data)
    trained_model = train_model(X_train, y_train)

    # Save the trained model for future inference
    joblib.dump(trained_model, 'hf_surrogate.joblib')
    mse_scores_revised = evaluate_model(trained_model, X_test, y_test)
    print(f"MSE for mse,mae, training_time: {mse_scores_revised[0]}, {mse_scores_revised[1]}, {mse_scores_revised[2]}")


if __name__ == "__main__":
    main()


#MSE for mse,mae, training_time: 0.00024320521323013735, 0.00028007296763602217, 6.600424584883627e-05