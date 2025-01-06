import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters for file paths
a_values = [0, 0.2, 0.4, 0.6, 0.8, 1]
length1 = 6
epsilon1 = 0.02
input_files1 = [f"../../Dataset/OpenData/House/results/ml{length1}/div{epsilon1}{int(a*10)}.json" for a in a_values]

length2 = 6
epsilon2 = 0.1
input_files2 = [f"../../Dataset/Kaggle/results/ml{length2}/div{epsilon2}{int(a*10)}.json" for a in a_values]

# Initialize figure with different subplot widths
fig, axes = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1, 1.5]})
ax1, ax2 = axes

# Performance diversity - Boxplot
metric_index = 2
all_y_values = []

# Read data and extract metric values for each file
for file in input_files1:
    try:
        with open(file, 'r') as f:
            data = json.load(f)
        y_values = [item["benefits"][metric_index] for item in data.values() if item["benefits"]]
        all_y_values.append(y_values)
    except FileNotFoundError:
        print(f"File not found: {file}")
        all_y_values.append([])

# Plot the boxplot
ax1.boxplot(
    all_y_values,
    positions=a_values,
    widths=0.08,  # Narrower boxplot
    patch_artist=True,
    showmeans=True,
    boxprops=dict(facecolor="lightblue", color="blue"),
    meanprops=dict(marker="o", markerfacecolor="red", markeredgecolor="red"),
    medianprops=dict(color='coral', linewidth=2)
)
ax1.set_xlim(-0.1, 1.1)
ax1.set_xlabel("$\\alpha$", fontsize=14)
ax1.set_ylabel("Accuracy", fontsize=14)
# ax1.set_title('(a) $T_2$ Accuracy vs. $\\alpha$ Values', y=-0.225, fontdict={'size': 15, 'fontweight': 'bold'})
ax1.set_title('(a) $T_2$ Performance Diversity', y=-0.222, fontdict={'size': 15, 'fontweight': 'bold'})
ax1.set_xticks(a_values)
ax1.set_xticklabels(a_values, fontsize=12)  # Adjust font size for x-ticks
ax1.tick_params(axis='y', labelsize=12)  # Adjust font size for y-ticks
ax1.grid(axis="y", linestyle="--", alpha=0.7)

# Content diversity - Heatmap
domain_selection_percentages = None
num_domains = 0

# Process each file
for j, (a, file) in enumerate(zip(a_values, input_files2)):
    try:
        with open(file, 'r') as f:
            data = json.load(f)

        active_domains = [item["nodes"][-1][-1] for item in data.values() if item["nodes"]]
        if num_domains == 0:
            num_domains = len(active_domains[0])
            domain_selection_percentages = np.zeros((num_domains, len(a_values)))

        active_matrix = np.array([list(map(int, domain)) for domain in active_domains])
        total_selections = np.sum(active_matrix)
        domain_counts = np.sum(active_matrix, axis=0)

        percentages = domain_counts / total_selections * 100 if total_selections > 0 else [0] * num_domains
        domain_selection_percentages[:, j] = percentages
    except FileNotFoundError:
        print(f"File not found: {file}")
        if domain_selection_percentages is not None:
            domain_selection_percentages[:, j] = [0] * num_domains

# Calculate standard deviations for each alpha
std_devs = np.std(domain_selection_percentages, axis=0)

# Plot the heatmap
sns.heatmap(
    domain_selection_percentages,
    annot=True,
    fmt=".1f",
    cmap="coolwarm",
    xticklabels=[f"{a:.1f}" for a in a_values],
    yticklabels=[f"$A_{{{i}}}$" for i in range(num_domains)],
    ax=ax2,
    annot_kws={"size": 13.6},  # Adjust font size for heatmap numbers
)

# Adjust the color bar legend font size
cbar = ax2.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)

# Add standard deviation labels
for j, std_dev in enumerate(std_devs):
    ax2.text(j + 0.5, -0.5, f"{std_dev:.2f}", ha="center", va="center", fontsize=14.5, color="black")

# Add 'Std.' label
ax2.text(-0.2, -0.5, "Std.", ha="center", va="center", fontsize=14.5, fontweight="bold")

# Adjust axis labels and title
ax2.set_xlabel("$\\alpha$", fontsize=14)
ax2.set_ylabel("Active Domains", fontsize=14)
# ax2.set_title('(b) $T_1$ Percentage of A vs. $\\alpha$ Values', y=-0.225, fontdict={'size': 15, 'fontweight': 'bold'})
ax2.set_title('(b) $T_1$ Content Diversity', y=-0.222, fontdict={'size': 15, 'fontweight': 'bold'})
ax2.tick_params(axis='x', labelsize=12)  # Adjust font size for x-ticks
ax2.tick_params(axis='y', labelsize=12)  # Adjust font size for y-ticks

# Save and show the plot
plt.tight_layout()
output_path = "./output/divmodis.svg"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Figure saved as {output_path}")
plt.show()
