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
processed_data.drop(columns=['Id', 'Label', 'stratum'], inplace=True)

# Features and target selection
target_cols = ['accuracy','precision','recall','f1','auc','time']
feature_cols = [col for col in processed_data.columns if col not in target_cols]

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
model_path = './mental_surrogate.joblib'
joblib.dump(model, model_path)

#       Metric       MSE
# 0   accuracy  0.000002
# 1  precision  0.000012
# 2     recall  0.000038
# 3         f1  0.000020
# 4        auc  0.000002
# 5       time  0.000325


