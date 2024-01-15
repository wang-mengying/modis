# Re-loading the data and re-applying the clustering process
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the data
housing_data = pd.read_csv('housing.csv')

# Drop columns with a large number of missing values
cols_to_drop = ['SOLD DATE', 'Unnamed: 7', 'NEXT OPEN HOUSE START TIME', 'NEXT OPEN HOUSE END TIME']
housing_data = housing_data.drop(columns=cols_to_drop)

# Impute numerical columns with their median values
numerical_cols = housing_data.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_cols:
    housing_data[col].fillna(housing_data[col].median(), inplace=True)

# Impute and encode categorical columns
categorical_cols = housing_data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    housing_data[col].fillna(housing_data[col].mode()[0], inplace=True)
    le = LabelEncoder()
    housing_data[col] = le.fit_transform(housing_data[col])

# Scaling the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(housing_data)

# Applying KMeans clustering
kmeans = KMeans(n_clusters=8, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

# Adding the cluster assignments to the original dataset
housing_data['cluster'] = clusters

# Saving to a CSV file
output_filepath = "clustered_table.csv"
housing_data.to_csv(output_filepath, index=False)

