import matplotlib.pyplot as plt
import pandas as pd
import os

# Zone names and file mapping
zones = [
    ("IT-NORD", "it-nord"),
    ("IT-CNOR", "it-cnor"),
    ("IT-CSUD", "it-csud"),
    ("IT-SUD", "it-sud"),
    ("IT-SICI", "it-sici"),
    ("IT-SARD", "it-sard"),
    ("IT-CALA", "it-cala"),
]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

for idx, (zone_name, zone_folder) in enumerate(zones):
    csv_path = os.path.join("results", zone_folder, "predictions_comparison.csv")
    if not os.path.exists(csv_path):
        axes[idx].set_visible(False)
        continue
    df = pd.read_csv(csv_path)
    # Scatter: Forecast (red)
    axes[idx].scatter(df["actual_mw"], df["predicted_forecast_mw"], color="red", alpha=0.5, s=10, label="Forecast")
    # Scatter: Actual (blue)
    axes[idx].scatter(df["actual_mw"], df["predicted_actual_mw"], color="blue", alpha=0.5, s=10, label="Actual")
    # 1:1 line
    min_val = min(df["actual_mw"].min(), df[["predicted_forecast_mw", "predicted_actual_mw"]].min().min())
    max_val = max(df["actual_mw"].max(), df[["predicted_forecast_mw", "predicted_actual_mw"]].max().max())
    axes[idx].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
    axes[idx].set_title(zone_name)
    axes[idx].set_xlabel("Actual MW")
    axes[idx].set_ylabel("Predicted MW")
    axes[idx].legend(fontsize=8)

# Hide any unused subplot
for j in range(len(zones), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig("results/scatter_grid.png", dpi=200)
plt.show()
