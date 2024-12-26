from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import joblib


# Function to process the Label column
def transform_label(row):
    feature_states, cluster_states = eval(row['Label'])
    states = {}
    for i, state in enumerate(feature_states):
        states[f'feature_{i + 1}'] = state
    for i, state in enumerate(cluster_states):
        states[f'cluster_{i + 1}'] = state
    return pd.Series(states)

# Load the uploaded file
file_path = './sample_nodes.csv'
data = pd.read_csv(file_path)

# Process the dataset
label_data = data.apply(transform_label, axis=1)
processed_data = pd.concat([data, label_data], axis=1)

# Drop unused columns
processed_data.drop(columns=['Label', 'stratum'], inplace=True)

# Features and target selection
feature_cols = [col for col in processed_data.columns if col not in [
    'Id', 'precision@5', 'precision@10', 'recall@5', 'recall@10', 'ndcg@5', 'ndcg@10'
]]
target_cols = ['precision@5', 'precision@10', 'recall@5', 'recall@10', 'ndcg@5', 'ndcg@10']

X = processed_data[feature_cols]
y = processed_data[target_cols]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model setup: Multi-output Gradient Boosting
model = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))

# Train the model
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')

# Results
mse_df = pd.DataFrame({
    'Metric': target_cols,
    'MSE': mse
})

print(mse_df)

# Save the trained model as a joblib file
model_path = './modsnet_surrogate.joblib'
joblib.dump(model, model_path)


#          Metric       MSE
# 0   precision@5  0.003376
# 1  precision@10  0.002245
# 2      recall@5  0.000237
# 3     recall@10  0.000485
# 4        ndcg@5  0.002800
# 5       ndcg@10  0.001979

