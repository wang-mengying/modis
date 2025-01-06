import json
import matplotlib.pyplot as plt

# Parameters for file paths
length = 6  # Adjust this if needed
epsilon = 0.02  # Adjust this if needed

# a values
a_values = [0, 0.2, 0.4, 0.6, 0.8, 1]
input_files = [f"../Dataset/OpenData/House/results/ml{length}/div{epsilon}{int(a*10)}.json" for a in a_values]

# Metric to plot (e.g., b[1])
metric_index =2
all_y_values = []

# Read data and extract metric values for each file
for file in input_files:
    with open(file, 'r') as f:
        data = json.load(f)
    y_values = [item["benefits"][metric_index] for item in data.values() if item["benefits"]]
    all_y_values.append(y_values)

# Plot the data
plt.figure(figsize=(8, 6))
plt.boxplot(all_y_values, positions=a_values, widths=0.05, patch_artist=True, showmeans=True,
            boxprops=dict(facecolor="lightblue", color="blue"),
            meanprops=dict(marker="o", markerfacecolor="red", markeredgecolor="red"))

plt.xlim(-0.1, 1.1)

# Add labels and title
plt.xlabel("a values", fontsize=12)
plt.ylabel(f"Metric b[{metric_index}]", fontsize=12)
plt.title(f"Metric b[{metric_index}] vs. a values", fontsize=14)
plt.xticks(a_values)  # Set x-axis ticks to a values
plt.grid(axis="y", linestyle="--", alpha=0.7)

# plt.xlabel("$\\alpha$", fontsize=12)
# plt.ylabel("Accuracy", fontsize=12)
# plt.title(f"Metric b[{metric_index}] vs. a values", fontsize=14)
# plt.xticks(a_values)  # Set x-axis ticks to a values
# plt.grid(axis="y", linestyle="--", alpha=0.7)

# Save the plot
output_path = f"metric_b{metric_index}.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Figure saved as {output_path}")

# Show the plot
plt.tight_layout()
plt.show()
