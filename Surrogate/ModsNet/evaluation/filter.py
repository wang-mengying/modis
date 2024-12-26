import pandas as pd
import ast

# Path to the sample_nodes.csv file
input_file = "./sample_nodes.csv"
output_file = "./filtered_ids.txt"

# Load the CSV file
data = pd.read_csv(input_file)

# Ensure the 'Label' column exists
if 'Label' not in data.columns:
    raise ValueError("The 'Label' column is missing from the input file.")

# Filter rows where the 3rd digit in the 2nd tuple is 0
def label_filter(label):
    try:
        items, values = eval(label)  # Safely evaluate the label string into two Python tuples
        if isinstance(values, (list, tuple)) and len(values) > 2 and values[2] == 0:
            return True
    except (ValueError, SyntaxError):
        pass
    return False

# Apply the filter to the dataframe
filtered_data = data[data['Label'].apply(label_filter)]

# Save the filtered IDs to a text file, one ID per line
with open(output_file, 'w') as f:
    for node_id in filtered_data['Id']:
        f.write(f"{node_id}\n")

print(f"Filtered IDs saved to {output_file}.")