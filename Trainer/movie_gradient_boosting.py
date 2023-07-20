import pandas as pd
import numpy as np
import time
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import category_encoders as ce
from sklearn.preprocessing import MultiLabelBinarizer


# Handle cases where a column does not exist in df
def get_column(df, column):
    try:
        return df[column]
    except KeyError:
        return None


def preprocess_data(df):
    # Replace all empty values
    df = df.replace('-', '0')
    df = df.replace('\\N', '0')
    df = df.replace(np.NaN, '0')

    # Replace 'alive' with the current year in 'director_deathYear' column
    director_deathYear = get_column(df, 'director_deathYear')
    if director_deathYear is not None:
        df['director_deathYear'] = director_deathYear.replace('alive', '2023')
        df['director_deathYear'] = df['director_deathYear'].astype(int)

    # Convert 'director_birthYear' and 'director_deathYear' to integers
    director_birthYear = get_column(df, 'director_birthYear')
    if director_birthYear is not None:
        df['director_birthYear'] = director_birthYear.astype(int)

    # Convert 'production_date' to datetime and then extract the year
    if 'production_date' in df.columns:
        df['production_date'] = pd.to_datetime(df['production_date'])
        df['production_year'] = get_column(df, 'production_date').dt.year
        df = df.drop('production_date', axis=1)

    # Binary encode the 'movie_title' and 'director_name'columns
    binary_encode = ['movie_title', 'director_name']
    for column in binary_encode:
        if column in df.columns:
            bin_enc = ce.BinaryEncoder(cols=[column])
            df = bin_enc.fit_transform(df)

    # Multi-label binarization for 'genres' and 'director_professions'
    multi_encode = ['genres', 'director_professions']
    for column in multi_encode:
        if column in df.columns:
            df = multi_label_binarization(df, column)

    # Group 'worldwide_gross' and trans it into a classification problem
    bins = [0, 50000000, 150000000, np.inf]
    labels = ['Low', 'Medium', 'High']
    df['gross_class'] = pd.cut(df['worldwide_gross'], bins=bins, labels=labels)

    return df


def multi_label_binarization(df, column):
    df[column] = df[column].str.split(',')

    mlb = MultiLabelBinarizer()
    df_encoded = mlb.fit_transform(df[column])

    encoded_df = pd.DataFrame(df_encoded, columns=mlb.classes_)
    df = pd.concat([df, encoded_df], axis=1)

    df = df.drop(column, axis=1)

    return df


def train_and_evaluate_model(df):
    df = df.fillna(df.mean())
    df = df.dropna()

    X = df.drop(['worldwide_gross', 'gross_class'], axis=1)
    y = df['gross_class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def main():
    dataset_path = "../Dataset/Movie/"
    filename = sys.argv[1] if len(sys.argv) > 1 else 'processed/movie_merged.csv'
    path = dataset_path + filename
    df = pd.read_csv(path)

    print(f'Training file: {filename}.')
    print(f'Size: {df.shape}.')

    df = preprocess_data(df)
    start_time = time.time()
    accuracy = train_and_evaluate_model(df)
    running_time = time.time() - start_time

    print(f'Accuracy: {accuracy}.')
    print(f'Running time: {running_time} s.')


if __name__ == '__main__':
    main()
