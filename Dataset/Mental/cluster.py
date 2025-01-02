import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

# Load the original dataset
file_path = '../../Dataset/Mental/uni_table.csv'  # Replace with your file path
original_data = pd.read_csv(file_path)

# Create a copy for processing
processed_data = original_data.copy()

# Preprocessing
processed_data.fillna(processed_data.median(numeric_only=True), inplace=True)
processed_data.fillna('Unknown', inplace=True)

# Encode categorical variables
categorical_cols = processed_data.select_dtypes(include='object').columns
label_encoders = {col: LabelEncoder() for col in categorical_cols}
for col in categorical_cols:
    processed_data[col] = label_encoders[col].fit_transform(processed_data[col])

# Normalize numerical columns
numerical_cols = processed_data.select_dtypes(include=np.number).columns.drop('Depression')
scaler = StandardScaler()
processed_data[numerical_cols] = scaler.fit_transform(processed_data[numerical_cols])

# Prepare features for clustering
X = processed_data.drop(columns=['Depression', 'id', 'Name'])

# Apply KMeans clustering
kmeans = KMeans(n_clusters=12, random_state=42)
clusters = kmeans.fit_predict(X)

# Append the Cluster column to the original dataset
original_data['Cluster'] = clusters

# Save the updated dataset
output_file = '../../Dataset/Mental/uni_table_clustered.csv'  # Replace with your desired output path
original_data.to_csv(output_file, index=False)
print(f"Cluster numbers appended to the original file. Updated file saved to {output_file}")

