import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import time

def preprocess_data(data):
    # Handle missing values
    data.fillna(data.median(numeric_only=True), inplace=True)
    data.fillna('Unknown', inplace=True)

    # Encode categorical variables
    categorical_cols = data.select_dtypes(include='object').columns
    label_encoders = {col: LabelEncoder() for col in categorical_cols}
    for col in categorical_cols:
        data[col] = label_encoders[col].fit_transform(data[col])

    # Normalize numerical columns
    numerical_cols = data.select_dtypes(include='number').columns.drop('Depression', errors='ignore')
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    X = data.drop(columns=['Depression'])
    y = data['Depression']

    return X, y


def train_model(X_train, y_train):
    model = LGBMClassifier(random_state=42, force_col_wise=True)
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    return accuracy, precision, recall, f1, roc_auc


def process(data):
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    start = time.time()
    model = train_model(X_train, y_train)
    end = time.time()
    train_time = end - start

    accuracy, precision, recall, f1, roc_auc = evaluate_model(model, X_test, y_test)

    return accuracy, precision, recall, f1, roc_auc, train_time


def main():
    # Preprocess the dataset
    # file_path = '../Dataset/Mental/uni_table.csv'
    file_path = '../Baselines/Mental/h2o.csv'
    data = pd.read_csv(file_path)
    print(f'Size: {data.shape}.')

    # Print evaluation results
    accuracy, precision, recall, f1, roc_auc, train_time = process(data)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    if roc_auc is not None:
        print(f"AUC: {roc_auc:.4f}")
    else:
        print("AUC could not be calculated as the model does not support probability predictions.")
    print(f"Training Time: {train_time:.4f} seconds")

# Run the main function
if __name__ == "__main__":
    main()
