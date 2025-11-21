"""
Compare production forecasts (no day-ahead) vs evaluation forecasts (with day-ahead).

This demonstrates the accuracy difference between the two modes.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

zone = "IT-NORD"

print("=" * 80)
print(f"COMPARING FORECAST MODES FOR {zone}")
print("=" * 80)
print()

# Load evaluation results (with day-ahead)
eval_pred = pd.read_csv(f'results/{zone.lower()}/predictions.csv')
eval_pred['datetime'] = pd.to_datetime(eval_pred['datetime'])

# Load production forecast (without day-ahead, zero baseline)
prod_pred = pd.read_csv(f'results/{zone.lower()}/production_forecast_zero.csv')
prod_pred['datetime'] = pd.to_datetime(prod_pred['datetime'])

# Merge on datetime
comparison = eval_pred.merge(
    prod_pred[['datetime', 'predicted_mw']],
    on='datetime',
    suffixes=('_eval', '_prod')
)
comparison.rename(columns={
    'predicted_mw_eval': 'with_dayahead',
    'predicted_mw_prod': 'without_dayahead'
}, inplace=True)

# Calculate metrics
actual = comparison['actual_mw'].values
with_da = comparison['with_dayahead'].values
without_da = comparison['without_dayahead'].values

# MAE
mae_with = np.mean(np.abs(actual - with_da))
mae_without = np.mean(np.abs(actual - without_da))

# RMSE
rmse_with = np.sqrt(np.mean((actual - with_da)**2))
rmse_without = np.sqrt(np.mean((actual - without_da)**2))

# R²
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

r2_with = r2_score(actual, with_da)
r2_without = r2_score(actual, without_da)

print("ACCURACY COMPARISON")
print("-" * 80)
print(f"{'Metric':<20} {'With Day-Ahead':<20} {'Without Day-Ahead':<20} {'Difference':<15}")
print("-" * 80)
print(f"{'MAE (MW)':<20} {mae_with:>18.2f} {mae_without:>18.2f} {mae_without - mae_with:>13.2f}")
print(f"{'RMSE (MW)':<20} {rmse_with:>18.2f} {rmse_without:>18.2f} {rmse_without - rmse_with:>13.2f}")
print(f"{'R² Score':<20} {r2_with:>18.3f} {r2_without:>18.3f} {r2_without - r2_with:>13.3f}")
print("-" * 80)

# Performance degradation
mae_degradation = ((mae_without - mae_with) / mae_with) * 100
r2_degradation = ((r2_with - r2_without) / r2_with) * 100

print(f"\nPerformance Impact of Removing Day-Ahead:")
print(f"  MAE increased by {mae_degradation:.1f}%")
print(f"  R² decreased by {r2_degradation:.1f}%")
print()

if mae_degradation < 20:
    print("  ✅ MINIMAL IMPACT - Production mode works well!")
elif mae_degradation < 50:
    print("  ⚠️  MODERATE IMPACT - Consider using simple baseline or retraining without residual")
else:
    print("  ❌ SIGNIFICANT IMPACT - Recommend training new model without residual mode")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Time series comparison
ax1 = axes[0, 0]
ax1.plot(comparison['datetime'], actual, label='Actual', linewidth=2, alpha=0.8)
ax1.plot(comparison['datetime'], with_da, label='With day-ahead', linewidth=2, alpha=0.7)
ax1.plot(comparison['datetime'], without_da, label='Without day-ahead', linewidth=2, alpha=0.7)
ax1.set_xlabel('Date', fontsize=11)
ax1.set_ylabel('Power (MW)', fontsize=11)
ax1.set_title('Forecast Comparison: With vs Without Day-Ahead', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Scatter with day-ahead
ax2 = axes[0, 1]
ax2.scatter(actual, with_da, alpha=0.5, s=20, label='With day-ahead')
max_val = max(actual.max(), with_da.max())
ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect')
ax2.set_xlabel('Actual (MW)', fontsize=11)
ax2.set_ylabel('Predicted (MW)', fontsize=11)
ax2.set_title(f'With Day-Ahead (R²={r2_with:.3f})', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal', adjustable='box')

# Plot 3: Scatter without day-ahead
ax3 = axes[1, 0]
ax3.scatter(actual, without_da, alpha=0.5, s=20, label='Without day-ahead', color='orange')
max_val = max(actual.max(), without_da.max())
ax3.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect')
ax3.set_xlabel('Actual (MW)', fontsize=11)
ax3.set_ylabel('Predicted (MW)', fontsize=11)
ax3.set_title(f'Without Day-Ahead (R²={r2_without:.3f})', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal', adjustable='box')

# Plot 4: Error comparison
ax4 = axes[1, 1]
errors_with = np.abs(actual - with_da)
errors_without = np.abs(actual - without_da)
ax4.hist(errors_with, bins=30, alpha=0.6, label=f'With (MAE={mae_with:.1f})', density=True)
ax4.hist(errors_without, bins=30, alpha=0.6, label=f'Without (MAE={mae_without:.1f})', density=True, color='orange')
ax4.axvline(mae_with, color='blue', linestyle='--', linewidth=2)
ax4.axvline(mae_without, color='orange', linestyle='--', linewidth=2)
ax4.set_xlabel('Absolute Error (MW)', fontsize=11)
ax4.set_ylabel('Density', fontsize=11)
ax4.set_title('Error Distribution Comparison', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
output_path = f'results/{zone.lower()}/mode_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Saved comparison plot to {output_path}")
plt.close()

# Save comparison CSV
output_csv = f'results/{zone.lower()}/mode_comparison.csv'
comparison.to_csv(output_csv, index=False)
print(f"✅ Saved comparison data to {output_csv}")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print(f"For {zone}:")
print(f"  • With day-ahead baseline: MAE = {mae_with:.1f} MW, R² = {r2_with:.3f}")
print(f"  • Without day-ahead (zero): MAE = {mae_without:.1f} MW, R² = {r2_without:.3f}")
print()
print("Recommendation:")
if mae_degradation < 20:
    print(f"  ✅ Production mode (no day-ahead) is viable for {zone}")
    print(f"     Only {mae_degradation:.1f}% accuracy loss - acceptable for standalone operation")
else:
    print(f"  ⚠️  Consider training a dedicated model without --residual flag")
    print(f"     Current loss: {mae_degradation:.1f}% - may be improved with direct training")
print()
