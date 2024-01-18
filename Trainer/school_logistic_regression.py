import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
import joblib


def preprocess_data(df):
    # Dropping the 'DBN' and 'School Name' columns
    df = df.drop(['DBN', 'School Name'], axis=1)

    # Identifying categorical and numerical columns
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'class']
    numerical_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]

    # Creating a column transformer with OneHotEncoder for categorical data and SimpleImputer for numerical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ])

    return preprocessor


def preprocess_data_with_mapper(df):
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'class']
    for col in categorical_cols:
        mapper = {value: i for i, value in enumerate(df[col].unique())}
        df[col] = df[col].map(mapper)

    numerical_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].mean())

    return df


def train_and_evaluate_model(df):
    # Preparing the features (X) and target (y)
    X = df.drop('class', axis=1)
    y = df['class']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Updating the pipeline with the new preprocessor
    # model = make_pipeline(preprocessor, LogisticRegression(max_iter=1000))
    model = LogisticRegression(max_iter=1000)

    # Training the model
    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start

    # Making predictions and evaluating the model
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    return f1, accuracy, training_time


def main():
    # file_path = '../Dataset/OpenData/School/processed/school_clustered.csv'
    file_path = '../Baselines/School/sksfm.csv'
    data = pd.read_csv(file_path)

    # preprocessor = preprocess_data(data)
    data = preprocess_data_with_mapper(data)
    f1, accuracy, training_time = train_and_evaluate_model(data)

    print(f'F1 Score: {f1}')
    print(f'Accuracy: {accuracy}')
    print(f'Training Time: {training_time}')
    print(f'Size: {data.shape}.')


if __name__ == "__main__":
    main()
