import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import time

def process_data(housing_data):
    # Load the data
    # housing_data = pd.read_csv(filepath)
    
    # Output the size of the original file
    original_size = housing_data.shape
    
    # Drop columns with a large number of missing values
    cols_to_drop = ['SOLD DATE', 'Unnamed: 7', 'NEXT OPEN HOUSE START TIME', 'NEXT OPEN HOUSE END TIME']
    housing_data = housing_data.drop(columns=cols_to_drop, errors='ignore')
    
    # Impute numerical columns with their median values
    numerical_cols = housing_data.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        housing_data[col].fillna(housing_data[col].median(), inplace=True)

    # Impute categorical columns with the most frequent value
    categorical_cols = housing_data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        housing_data[col].fillna(housing_data[col].mode()[0], inplace=True)
        
    # Convert PRICE to classes
    low_threshold = housing_data['PRICE'].quantile(0.33)
    medium_threshold = housing_data['PRICE'].quantile(0.66)
    housing_data['PRICE_CLASS'] = housing_data['PRICE'].apply(lambda x: 'Low' if x <= low_threshold else ('Medium' if x <= medium_threshold else 'High'))

    cat_cols = ['SALE TYPE', 'CITY', 'STATE OR PROVINCE', 'ZIP OR POSTAL CODE', 'SOURCE', 'FAVORITE', 'INTERESTED', 'LOCATION', 'STATUS']
    for col in cat_cols:
        try:
            mapper = {k: v for v, k in enumerate(housing_data[col].unique())}
            housing_data[col] = housing_data[col].map(mapper)
        except:
            pass

    # Drop unnecessary columns and encode categorical columns
    cols_to_drop = ['ADDRESS', 'URL (SEE https://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)',
                    'MLS#', 'PRICE']
    housing_data = housing_data.drop(columns=cols_to_drop, errors='ignore')
    housing_data_encoded = pd.get_dummies(housing_data, drop_first=True)
    
    X = housing_data_encoded.drop(columns=['PRICE_CLASS_Low', 'PRICE_CLASS_Medium'])
    y = housing_data['PRICE_CLASS']
    
    return X, y, original_size

def train_and_save_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    rf_classifier = RandomForestClassifier(n_jobs=-1, random_state=42)
    
    start_time = time.time()
    rf_classifier.fit(X_train, y_train)
    end_time = time.time()
    
    y_pred = rf_classifier.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return accuracy, f1, end_time - start_time


def main():
    filepath = "../Dataset/OpenData/House/processed/house_filtered.csv"
    # filepath = "../Baselines/House/metam_mult.csv"
    housing_data = pd.read_csv(filepath)

    X, y, original_size = process_data(housing_data)
    accuracy, f1, training_time = train_and_save_model(X, y)

    print(f"Original File Size: {original_size[0]} rows, {original_size[1]} columns")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")


if __name__ == '__main__':
    main()
