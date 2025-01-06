import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters for file paths
length = 6  # Adjust this if needed
epsilon = 0.1  # Adjust this if needed
a_values = [0, 0.2, 0.4, 0.6, 0.8, 1]
input_files = [f"../Dataset/Kaggle/results/ml{length}/div{epsilon}{int(a*10)}.json" for a in a_values]

# Initialize variables
domain_selection_percentages = None
num_domains = 0

# Process each file
for j, (a, file) in enumerate(zip(a_values, input_files)):
    try:
        # Load the JSON data
        with open(file, 'r') as f:
            data = json.load(f)

        # Extract active domains (nodes[-1][-1]) for each dataset
        active_domains = [item["nodes"][-1][-1] for item in data.values() if item["nodes"]]

        # Dynamically determine the number of domains
        if num_domains == 0:
            num_domains = len(active_domains[0])  # Get the length of the first domain vector
            domain_selection_percentages = np.zeros((num_domains, len(a_values)))

        # Flatten all active domains to count total selections per domain
        active_matrix = np.array([list(map(int, domain)) for domain in active_domains])
        total_selections = np.sum(active_matrix)  # Total selections across all datasets
        domain_counts = np.sum(active_matrix, axis=0)  # Count selections per domain

        # Compute percentage contribution for each domain
        percentages = domain_counts / total_selections * 100 if total_selections > 0 else [0] * num_domains
        domain_selection_percentages[:, j] = percentages

    except FileNotFoundError:
        print(f"File not found: {file}")
        if domain_selection_percentages is not None:
            domain_selection_percentages[:, j] = [0] * num_domains  # Fill with zeros if file is missing

# Calculate standard deviations for each alpha
std_devs = np.std(domain_selection_percentages, axis=0)

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    domain_selection_percentages,
    annot=True,
    fmt=".1f",
    cmap="coolwarm",
    xticklabels=[f"{a:.1f}" for a in a_values],
    yticklabels=[f"Domain {i}" for i in range(num_domains)]
)

# Add standard deviation labels above the heatmap
for j, std_dev in enumerate(std_devs):
    plt.text(j + 0.5, -0.5, f"{std_dev:.2f}", ha="center", va="center", fontsize=10, color="black")

# Adjust axis labels and title
plt.xlabel("Alpha Values", fontsize=12)
plt.ylabel("Active Domains (Row Clusters)", fontsize=12)
plt.title("Percentage Contribution of Active Domains with Standard Deviations", fontsize=14)

# Save the plot
output_path = f"content_div.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Figure saved as {output_path}")

plt.tight_layout()
plt.show()

