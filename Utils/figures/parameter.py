import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Original metric values for comparison
original_metrics = {
    "Precision@5": 0.7200,
    "Precision@10": 0.6600,
    "Recall@5": 0.1863,
    "Recall@10": 0.3217,
    "NDCG@5": 0.6923,
    "NDCG@10": 0.6646,
}

# Load the dataset
data = pd.read_csv('../../Dataset/ModsNet/ModsNet_Results.csv')

# Define a function to calculate percentage change

def calculate_percentage_change(df, original_values):
    for metric, original_value in original_values.items():
        metric_columns = [col for col in df.columns if metric.lower() in col.lower()]
        for col in metric_columns:
            df[col] = ((df[col] - original_value) / original_value) * 100
    return df

# Function to calculate overall percentage improvement
def calculate_overall_improvement_percentage(df):
    total_values = df.values.flatten()
    valid_values = total_values[~np.isnan(total_values)]  # Exclude NaNs
    improvements = valid_values[valid_values > 0]
    return (len(improvements) / len(valid_values)) * 100

# Filter data for the two heatmaps
heatmap1_data = data[data["maxl"] == 6].copy()
heatmap2_data = data[data["epsilon"] == 0.02].copy()

# Pivot data for heatmaps
heatmap1_pivot = heatmap1_data.pivot_table(
    index="epsilon",
    columns=["metric"],
    values=["ApxMODis", "BiMODis", "DivMODis", "NOMODis"]
)
heatmap1_pivot.columns = [
    f"{alg}*{metric}" for alg, metric in heatmap1_pivot.columns
]
heatmap1_pivot.reset_index(inplace=True)

heatmap2_pivot = heatmap2_data.pivot_table(
    index="maxl",
    columns=["metric"],
    values=["ApxMODis", "BiMODis", "DivMODis", "NOMODis"]
)
heatmap2_pivot.columns = [
    f"{alg}*{metric}" for alg, metric in heatmap2_pivot.columns
]
heatmap2_pivot.reset_index(inplace=True)

# Apply percentage change calculation
metric_mapping = {
    "precision@5": original_metrics["Precision@5"],
    "precision@10": original_metrics["Precision@10"],
    "recall@5": original_metrics["Recall@5"],
    "recall@10": original_metrics["Recall@10"],
    "ndcg@5": original_metrics["NDCG@5"],
    "ndcg@10": original_metrics["NDCG@10"],
}

heatmap1_pivot = calculate_percentage_change(heatmap1_pivot, metric_mapping)
heatmap2_pivot = calculate_percentage_change(heatmap2_pivot, metric_mapping)

# Calculate overall improvement percentages
overall_improvement_h1 = calculate_overall_improvement_percentage(heatmap1_pivot.set_index("epsilon"))
overall_improvement_h2 = calculate_overall_improvement_percentage(heatmap2_pivot.set_index("maxl"))

# Plot heatmaps
plt.figure(figsize=(16, 14))

# Heatmap 1
plt.subplot(2, 1, 2)
ax1 = sns.heatmap(
    heatmap1_pivot.set_index("epsilon"),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",  # Use coolwarm colormap
    cbar_kws={"label": "Percentage Change (%)", "pad": 0.02, "shrink": 0.9},
    center=0,  # Center the colormap at 0 to set white at the transition line
    annot_kws={"size": 11},
)
ax1.figure.axes[-1].yaxis.label.set_size(14)
plt.title(f"(b) $T_4$ Sensitivity with $\\epsilon$, fix $maxl = 6$ (Improved Results: {overall_improvement_h1:.2f}%)", fontsize=15, y=-1, fontweight='bold')
plt.xlabel("Algorithms*Metrics", fontsize=14)
plt.ylabel("Epsilon - $\\epsilon$", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.gca().yaxis.label.set_size(15)

# Add vertical lines to separate algorithms
for i in range(6, len(heatmap2_pivot.columns) - 1, 6):
    plt.axvline(x=i, color='black', linewidth=0.8)

# Heatmap 2
plt.subplot(2, 1, 1)
ax2 = sns.heatmap(
    heatmap2_pivot.set_index("maxl"),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",  # Use coolwarm colormap
    cbar_kws={"label": "Percentage Change (%)", "pad": 0.02, "shrink": 0.9},
    center=0,  # Center the colormap at 0 to set white at the transition line
    annot_kws={"size": 11}
)
ax2.figure.axes[-1].yaxis.label.set_size(14)
plt.title(f"(a) $T_4$ Sensitivity with $maxl$, fix $\\epsilon= 0.02$ (Improved Results: {overall_improvement_h2:.2f}%)", fontsize=15, y=-1, fontweight='bold')
plt.xlabel("Algorithms*Metrics", fontsize=14)
plt.ylabel("Maximum Length - $maxl$", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.gca().yaxis.label.set_size(15)

# Add vertical lines to separate algorithms
for i in range(6, len(heatmap2_pivot.columns) - 1, 6):
    plt.axvline(x=i, color='black', linewidth=0.8)

plt.tight_layout(h_pad=-12)
svg_path = "./output/parameter.svg"
png_path = "./output/parameter.png"
plt.savefig(svg_path, dpi=300, bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.show()
