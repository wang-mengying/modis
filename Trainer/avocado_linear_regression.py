import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time


def pre_processing(avocado_data):
    # Replace all empty values
    avocado_data = avocado_data.replace('-', '0')
    avocado_data = avocado_data.replace('\\N', '0')
    avocado_data = avocado_data.replace(np.NaN, '0')

    avocado_data['Date'] = pd.to_datetime(avocado_data['Date'])
    avocado_data['month'] = avocado_data['Date'].dt.month
    avocado_data['day'] = avocado_data['Date'].dt.day
    avocado_data.drop(columns=['Date', 'year'], inplace=True)

    encoder = OneHotEncoder(drop='first', sparse=False)
    encoded_features = encoder.fit_transform(avocado_data[['type', 'region']])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['type', 'region']))
    avocado_data = pd.concat([avocado_data, encoded_df], axis=1)
    avocado_data.drop(columns=['type', 'region'], inplace=True)

    return avocado_data


def train_test(avocado_data):
    X = avocado_data.drop(columns='AveragePrice')
    y = avocado_data['AveragePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    start_time = time.time()
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    end_time = time.time()

    y_pred = regressor.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    training_time = end_time - start_time

    return mse, mae, training_time


def main():
    avocado_data = pd.read_csv('../Dataset/HuggingFace/processed/avocado_filtered.csv')
    print(f'Size: {avocado_data.shape}.')
    avocado_data = pre_processing(avocado_data)
    mse, mae, training_time = train_test(avocado_data)
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Training Time: {training_time:.4f} seconds")


if __name__ == '__main__':
    main()



