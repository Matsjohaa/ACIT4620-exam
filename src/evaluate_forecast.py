"""
Evaluate the Encoder-Decoder model on test data (Oct 27 - Nov 10, 2025)
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

from model import EncoderDecoderCNNLSTM
from model_large import LargeEncoderDecoderCNNLSTM
from data_loader import (
    load_zone_data,
    WEATHER_FEATURES,
    ENGINEERED_FEATURES,
    TEMPORAL_FEATURES,
)


ZONES = ["IT-NORD", "IT-CNOR", "IT-CSUD", "IT-SUD", "IT-SICI", "IT-SARD", "IT-CALA"]
SEQ_LEN = 168
HORIZON = 336

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_norm_params(path):
    """Load normalization parameters."""
    data = np.load(path)
    return data["mean"], data["std"]


def build_test_sample(train_df, test_df):
    """
    Build encoder history + test horizon for Oct 27–Nov 10.
    Direct prediction mode (no day-ahead forecast needed).
    """
    train_df = train_df.sort_values("date").reset_index(drop=True)
    test_df = test_df.sort_values("date").reset_index(drop=True)

    assert len(test_df) == HORIZON, f"Expected {HORIZON} test hours, got {len(test_df)}"

    train_tail = train_df.tail(SEQ_LEN)

    features = WEATHER_FEATURES + ENGINEERED_FEATURES + TEMPORAL_FEATURES
    features = [f for f in features if f in train_tail.columns]

    X_enc_raw = train_tail[features].values         # [168, F]
    X_dec_raw = test_df[features].values            # [336, F]

    y_true_MW = test_df["actual"].values            # [336]

    inst_cap = test_df["installed_capacity_mw"].values[0]  # scalar
    dates = pd.to_datetime(test_df["date"].values)

    return X_enc_raw, X_dec_raw, y_true_MW, inst_cap, dates, len(features)


def metrics(y_true, y_pred):
    """Calculate MAE, RMSE, R² metrics."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return mae, rmse, r2


def apply_nighttime_zeroing(cf_pred, test_df):
    """
    Zero out predictions when solar radiation is negligible.
    Uses simple physics-based rule: if radiation < 0.001, solar production = 0.
    This prevents model from wasting capacity on trivial nighttime patterns.
    """
    test_df = test_df.sort_values("date").reset_index(drop=True)
    radiation = test_df["shortwave_radiation"].values
    
    # Simple rule: No radiation = No solar production
    # Threshold: 0.001 W/m² (effectively zero, allows for numerical precision)
    RADIATION_THRESHOLD = 1e-3
    zero_mask = radiation < RADIATION_THRESHOLD
    
    # Apply zeroing
    cf_pred_zeroed = cf_pred.copy()
    cf_pred_zeroed[zero_mask] = 0.0
    
    # Statistics
    n_zeroed = np.sum(zero_mask)
    n_daytime = len(cf_pred) - n_zeroed
    
    # Additional check: zero out any negative predictions (model artifact)
    negative_mask = cf_pred_zeroed < 0
    cf_pred_zeroed[negative_mask] = 0.0
    n_negative = np.sum(negative_mask)
    
    print(f"  → Zeroed {n_zeroed}/{len(cf_pred)} nighttime hours ({100*n_zeroed/len(cf_pred):.1f}%)")
    print(f"  → Active daytime hours: {n_daytime} ({100*n_daytime/len(cf_pred):.1f}%)")
    if n_negative > 0:
        print(f"  → Corrected {n_negative} negative predictions to zero")
    
    return cf_pred_zeroed


def load_model(zone, n_features, device, use_large_model=False):
    """Load the encoder-decoder model."""
    models_dir = Path("models") / zone.lower()
    model_path = models_dir / "model.pt"
    
    if use_large_model:
        model = LargeEncoderDecoderCNNLSTM(
            enc_sequence_length=SEQ_LEN,
            dec_sequence_length=HORIZON,
            n_features=n_features,
        ).to(device)
    else:
        model = EncoderDecoderCNNLSTM(
            enc_sequence_length=SEQ_LEN,
            dec_sequence_length=HORIZON,
            n_features=n_features,
        ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from {model_path}")
    return model


def save_predictions_csv(dates, y_true_MW, y_pred_MW, output_path):
    """
    Save hour-by-hour predictions to CSV.
    """
    df = pd.DataFrame({
        'datetime': dates,
        'actual_mw': y_true_MW,
        'predicted_mw': y_pred_MW,
        'error_mw': y_pred_MW - y_true_MW,
        'absolute_error_mw': np.abs(y_pred_MW - y_true_MW),
        'percent_error': 100 * (y_pred_MW - y_true_MW) / (y_true_MW + 1e-6)
    })
    df.to_csv(output_path, index=False)
    print(f"Saved predictions CSV to {output_path}")


def plot_forecast(zone, dates, y_true_MW, y_pred_MW, mae, rmse, r2, output_path):
    """
    Plot actual vs predicted forecast.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot actual and predicted with default matplotlib colors
    ax.plot(dates, y_true_MW, label="Actual", linewidth=2, alpha=0.8)
    ax.plot(dates, y_pred_MW, label="Predicted", linewidth=2, alpha=0.8)
    
    # Add metrics text box
    textstr = f'MAE: {mae:.2f} MW\nRMSE: {rmse:.2f} MW\nR²: {r2:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Power (MW)", fontsize=12)
    ax.set_title(f"{zone} – 14-Day Solar Forecast", 
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(DateFormatter("%b %d"))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved forecast plot to {output_path}")
    plt.close()


def plot_scatter(zone, y_true_MW, y_pred_MW, mae, rmse, r2, output_path):
    """
    Create scatter plot of predicted vs actual values.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(y_true_MW, y_pred_MW, alpha=0.5, s=20, edgecolors='none')
    
    # Perfect prediction line
    min_val = min(y_true_MW.min(), y_pred_MW.min())
    max_val = max(y_true_MW.max(), y_pred_MW.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    # Add metrics text box
    textstr = f'MAE: {mae:.2f} MW\nRMSE: {rmse:.2f} MW\nR²: {r2:.3f}\nSamples: {len(y_true_MW)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    ax.set_xlabel("Actual Power (MW)", fontsize=12)
    ax.set_ylabel("Predicted Power (MW)", fontsize=12)
    ax.set_title(f"{zone} – Predicted vs Actual", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved scatter plot to {output_path}")
    plt.close()


def plot_error_distribution(y_true_MW, y_pred_MW, output_path):
    """
    Plot error distribution histogram and time series.
    """
    errors = y_pred_MW - y_true_MW
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Error histogram
    axes[0, 0].hist(errors, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    axes[0, 0].axvline(errors.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.2f} MW')
    axes[0, 0].set_xlabel("Error (MW)", fontsize=11)
    axes[0, 0].set_ylabel("Frequency", fontsize=11)
    axes[0, 0].set_title("Error Distribution", fontsize=12, fontweight="bold")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Absolute error histogram
    abs_errors = np.abs(errors)
    axes[0, 1].hist(abs_errors, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 1].axvline(abs_errors.mean(), color='red', linestyle='--', linewidth=2, label=f'MAE: {abs_errors.mean():.2f} MW')
    axes[0, 1].set_xlabel("Absolute Error (MW)", fontsize=11)
    axes[0, 1].set_ylabel("Frequency", fontsize=11)
    axes[0, 1].set_title("Absolute Error Distribution", fontsize=12, fontweight="bold")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Error over time
    axes[1, 0].plot(errors, linewidth=1, alpha=0.7)
    axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].fill_between(range(len(errors)), errors, 0, alpha=0.3)
    axes[1, 0].set_xlabel("Hour", fontsize=11)
    axes[1, 0].set_ylabel("Error (MW)", fontsize=11)
    axes[1, 0].set_title("Error Over Time", fontsize=12, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Percentage error histogram
    pct_errors = 100 * errors / (y_true_MW + 1e-6)
    axes[1, 1].hist(pct_errors, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].axvline(pct_errors.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {pct_errors.mean():.1f}%')
    axes[1, 1].set_xlabel("Percentage Error (%)", fontsize=11)
    axes[1, 1].set_ylabel("Frequency", fontsize=11)
    axes[1, 1].set_title("Percentage Error Distribution", fontsize=12, fontweight="bold")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved error analysis plot to {output_path}")
    plt.close()


def plot_hourly_performance(dates, y_true_MW, y_pred_MW, output_path):
    """
    Analyze performance by hour of day.
    """
    hours = dates.hour.values
    errors = np.abs(y_pred_MW - y_true_MW)
    
    # Calculate mean absolute error per hour
    hourly_mae = []
    hourly_std = []
    for h in range(24):
        mask = hours == h
        if mask.sum() > 0:
            hourly_mae.append(errors[mask].mean())
            hourly_std.append(errors[mask].std())
        else:
            hourly_mae.append(0)
            hourly_std.append(0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. MAE by hour
    axes[0].bar(range(24), hourly_mae, alpha=0.7, edgecolor='black')
    axes[0].errorbar(range(24), hourly_mae, yerr=hourly_std, fmt='none', ecolor='red', capsize=3, alpha=0.5)
    axes[0].set_xlabel("Hour of Day", fontsize=11)
    axes[0].set_ylabel("Mean Absolute Error (MW)", fontsize=11)
    axes[0].set_title("MAE by Hour of Day", fontsize=12, fontweight="bold")
    axes[0].set_xticks(range(24))
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 2. Average actual vs predicted by hour
    hourly_actual = []
    hourly_pred = []
    for h in range(24):
        mask = hours == h
        if mask.sum() > 0:
            hourly_actual.append(y_true_MW[mask].mean())
            hourly_pred.append(y_pred_MW[mask].mean())
        else:
            hourly_actual.append(0)
            hourly_pred.append(0)
    
    x = range(24)
    width = 0.35
    axes[1].bar([i - width/2 for i in x], hourly_actual, width, label='Actual', alpha=0.7)
    axes[1].bar([i + width/2 for i in x], hourly_pred, width, label='Predicted', alpha=0.7)
    axes[1].set_xlabel("Hour of Day", fontsize=11)
    axes[1].set_ylabel("Average Power (MW)", fontsize=11)
    axes[1].set_title("Average Production by Hour", fontsize=12, fontweight="bold")
    axes[1].set_xticks(range(24))
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved hourly performance plot to {output_path}")
    plt.close()


def create_metrics_table(mae, rmse, r2, y_true_MW, y_pred_MW, output_path):
    """
    Create a comprehensive metrics table and save as CSV.
    """
    errors = y_pred_MW - y_true_MW
    abs_errors = np.abs(errors)
    pct_errors = 100 * errors / (y_true_MW + 1e-6)
    
    metrics = {
        'Metric': [
            'Mean Absolute Error (MAE)',
            'Root Mean Square Error (RMSE)',
            'R² Score',
            'Mean Error (Bias)',
            'Median Absolute Error',
            'Max Absolute Error',
            'Min Absolute Error',
            'Std Dev of Errors',
            'Mean Percentage Error',
            'Mean Absolute Percentage Error (MAPE)',
            'Number of Samples',
            'Actual Mean',
            'Actual Std Dev',
            'Predicted Mean',
            'Predicted Std Dev',
        ],
        'Value': [
            f'{mae:.2f} MW',
            f'{rmse:.2f} MW',
            f'{r2:.4f}',
            f'{errors.mean():.2f} MW',
            f'{np.median(abs_errors):.2f} MW',
            f'{abs_errors.max():.2f} MW',
            f'{abs_errors.min():.2f} MW',
            f'{errors.std():.2f} MW',
            f'{pct_errors.mean():.2f}%',
            f'{np.abs(pct_errors).mean():.2f}%',
            f'{len(y_true_MW)}',
            f'{y_true_MW.mean():.2f} MW',
            f'{y_true_MW.std():.2f} MW',
            f'{y_pred_MW.mean():.2f} MW',
            f'{y_pred_MW.std():.2f} MW',
        ]
    }
    
    df = pd.DataFrame(metrics)
    df.to_csv(output_path, index=False)
    print(f"Saved metrics table to {output_path}")
    
    # Also print to console
    print("\n" + "=" * 80)
    print("DETAILED METRICS")
    print("=" * 80)
    for metric, value in zip(metrics['Metric'], metrics['Value']):
        print(f"{metric:.<50} {value:>20}")
    print("=" * 80)


def evaluate_single_scenario(zone, train_df, test_df, model, mean, std, inst_cap, dates, scenario_name):
    """Evaluate a single test scenario (forecast or actual weather)."""
    # Build test sequences
    X_enc_raw, X_dec_raw, y_true_MW, _, _, n_features = build_test_sample(train_df, test_df)
    
    # Normalize
    mean_reshaped = mean.reshape(1, 1, -1)
    std_reshaped = std.reshape(1, 1, -1)
    X_enc = (X_enc_raw[None, ...] - mean_reshaped) / std_reshaped
    X_dec = (X_dec_raw[None, ...] - mean_reshaped) / std_reshaped
    
    # Predict (direct prediction, not residual)
    print(f"\nGenerating {scenario_name} predictions...")
    with torch.no_grad():
        Xe = torch.tensor(X_enc, dtype=torch.float32, device=device)
        Xd = torch.tensor(X_dec, dtype=torch.float32, device=device)
        cf_pred = model(Xe, Xd).cpu().numpy().reshape(-1)
    
    # Clip to valid range [0, 1.2] (capacity factor)
    cf_pred = np.clip(cf_pred, 0.0, 1.2)
    
    # Apply physics-based nighttime zeroing (radiation < 0.001 → production = 0)
    # This ensures predictions align with physical reality: no sun = no solar power
    cf_pred = apply_nighttime_zeroing(cf_pred, test_df.copy())
    
    # Convert to MW
    y_pred_MW = cf_pred * inst_cap
    
    # Calculate metrics
    mae, rmse, r2 = metrics(y_true_MW, y_pred_MW)
    
    return {
        'predictions': y_pred_MW,
        'actual': y_true_MW,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'dates': dates
    }


def save_comparison_csv(dates, results_forecast, results_actual, output_path):
    """Save comparison CSV with both forecast and actual weather predictions."""
    df = pd.DataFrame({
        'datetime': dates,
        'actual_mw': results_forecast['actual'],  # Same for both
        'predicted_forecast_mw': results_forecast['predictions'],
        'predicted_actual_mw': results_actual['predictions'],
        'error_forecast_mw': results_forecast['predictions'] - results_forecast['actual'],
        'error_actual_mw': results_actual['predictions'] - results_actual['actual'],
        'abs_error_forecast_mw': np.abs(results_forecast['predictions'] - results_forecast['actual']),
        'abs_error_actual_mw': np.abs(results_actual['predictions'] - results_actual['actual']),
    })
    df.to_csv(output_path, index=False)
    print(f"Saved comparison CSV to {output_path}")


def plot_forecast_comparison(zone, dates, results_forecast, results_actual, output_path):
    """Plot forecast comparison with different colors for forecast vs actual weather."""
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # Plot actual (black) - thicker for visibility
    ax.plot(dates, results_forecast['actual'], label="Actual Production", 
            linewidth=3, alpha=1.0, color='black', zorder=3)
    
    # Plot predictions with solid lines and high-contrast colors
    ax.plot(dates, results_forecast['predictions'], label=f"Predicted (Weather Forecast)", 
            linewidth=2.5, alpha=1.0, color='#FF0000', linestyle='-')  # Bright Red - solid
    ax.plot(dates, results_actual['predictions'], label=f"Predicted (Actual Weather)", 
            linewidth=2.5, alpha=1.0, color='#0066FF', linestyle='-')  # Bright Blue - solid
    
    # Add metrics text boxes with updated colors
    textstr_forecast = (f'Forecast Weather:\n'
                       f'MAE: {results_forecast["mae"]:.2f} MW\n'
                       f'R²: {results_forecast["r2"]:.3f}')
    textstr_actual = (f'Actual Weather:\n'
                     f'MAE: {results_actual["mae"]:.2f} MW\n'
                     f'R²: {results_actual["r2"]:.3f}')
    
    props_forecast = dict(boxstyle='round', facecolor='#FFE5E5', alpha=0.95, edgecolor='#FF0000', linewidth=2)
    props_actual = dict(boxstyle='round', facecolor='#E5ECFF', alpha=0.95, edgecolor='#0066FF', linewidth=2)
    
    ax.text(0.02, 0.98, textstr_forecast, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props_forecast, weight='bold')
    ax.text(0.02, 0.78, textstr_actual, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props_actual, weight='bold')
    
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Power (MW)", fontsize=12)
    ax.set_title(f"{zone} – 14-Day Solar Forecast Comparison", 
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(DateFormatter("%b %d"))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved forecast comparison plot to {output_path}")
    plt.close()


def plot_scatter_comparison(zone, results_forecast, results_actual, output_path):
    """Create side-by-side scatter plots for both scenarios."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Forecast weather scatter - bright red
    ax1.scatter(results_forecast['actual'], results_forecast['predictions'], 
                alpha=0.7, s=35, edgecolors='none', color='#FF0000')
    min_val = min(results_forecast['actual'].min(), results_forecast['predictions'].min())
    max_val = max(results_forecast['actual'].max(), results_forecast['predictions'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.7, label='Perfect prediction')
    
    textstr = (f'MAE: {results_forecast["mae"]:.2f} MW\n'
               f'RMSE: {results_forecast["rmse"]:.2f} MW\n'
               f'R²: {results_forecast["r2"]:.3f}')
    props = dict(boxstyle='round', facecolor='#FFE5E5', alpha=0.95, edgecolor='#FF0000', linewidth=2)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, weight='bold')
    
    ax1.set_xlabel("Actual Power (MW)", fontsize=11)
    ax1.set_ylabel("Predicted Power (MW)", fontsize=11)
    ax1.set_title("Weather Forecast", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # Actual weather scatter - bright blue
    ax2.scatter(results_actual['actual'], results_actual['predictions'], 
                alpha=0.7, s=35, edgecolors='none', color='#0066FF')
    min_val = min(results_actual['actual'].min(), results_actual['predictions'].min())
    max_val = max(results_actual['actual'].max(), results_actual['predictions'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.7, label='Perfect prediction')
    
    textstr = (f'MAE: {results_actual["mae"]:.2f} MW\n'
               f'RMSE: {results_actual["rmse"]:.2f} MW\n'
               f'R²: {results_actual["r2"]:.3f}')
    props = dict(boxstyle='round', facecolor='#E5ECFF', alpha=0.95, edgecolor='#0066FF', linewidth=2)
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, weight='bold')
    
    ax2.set_xlabel("Actual Power (MW)", fontsize=11)
    ax2.set_ylabel("Predicted Power (MW)", fontsize=11)
    ax2.set_title("Actual Weather", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    fig.suptitle(f"{zone} – Predicted vs Actual Comparison", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved scatter comparison plot to {output_path}")
    plt.close()


def plot_error_comparison(results_forecast, results_actual, output_path):
    """Plot error distribution comparison between forecast and actual weather."""
    errors_forecast = results_forecast['predictions'] - results_forecast['actual']
    errors_actual = results_actual['predictions'] - results_actual['actual']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Error histograms comparison
    bins = np.linspace(min(errors_forecast.min(), errors_actual.min()),
                      max(errors_forecast.max(), errors_actual.max()), 40)
    axes[0, 0].hist(errors_forecast, bins=bins, alpha=0.7, label='Forecast Weather', color='#FF0000', edgecolor='black')
    axes[0, 0].hist(errors_actual, bins=bins, alpha=0.7, label='Actual Weather', color='#0066FF', edgecolor='black')
    axes[0, 0].axvline(0, color='black', linestyle='--', linewidth=2, label='Zero error')
    axes[0, 0].axvline(errors_forecast.mean(), color='#FF0000', linestyle=':', linewidth=2.5)
    axes[0, 0].axvline(errors_actual.mean(), color='#0066FF', linestyle=':', linewidth=2.5)
    axes[0, 0].set_xlabel("Error (MW)", fontsize=11)
    axes[0, 0].set_ylabel("Frequency", fontsize=11)
    axes[0, 0].set_title("Error Distribution Comparison", fontsize=12, fontweight="bold")
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Absolute errors time series
    axes[0, 1].plot(np.abs(errors_forecast), alpha=0.9, linewidth=2, label='Forecast Weather', color='#FF0000')
    axes[0, 1].plot(np.abs(errors_actual), alpha=0.9, linewidth=2, label='Actual Weather', color='#0066FF')
    axes[0, 1].set_xlabel("Hour", fontsize=11)
    axes[0, 1].set_ylabel("Absolute Error (MW)", fontsize=11)
    axes[0, 1].set_title("Absolute Error Over Time", fontsize=12, fontweight="bold")
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Boxplot comparison
    box_forecast = axes[1, 0].boxplot([errors_forecast], positions=[1], 
                       labels=['Forecast'],
                       patch_artist=True,
                       widths=0.6,
                       boxprops=dict(facecolor='#FF0000', alpha=0.7, linewidth=2),
                       medianprops=dict(color='black', linewidth=2),
                       whiskerprops=dict(linewidth=2),
                       capprops=dict(linewidth=2))
    box_actual = axes[1, 0].boxplot([errors_actual], positions=[2], 
                       labels=['Actual'],
                       patch_artist=True,
                       widths=0.6,
                       boxprops=dict(facecolor='#0066FF', alpha=0.7, linewidth=2),
                       medianprops=dict(color='black', linewidth=2),
                       whiskerprops=dict(linewidth=2),
                       capprops=dict(linewidth=2))
    axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[1, 0].set_ylabel("Error (MW)", fontsize=11)
    axes[1, 0].set_title("Error Distribution Boxplot", fontsize=12, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. MAE comparison bar chart
    metrics_labels = ['MAE', 'RMSE']
    forecast_vals = [results_forecast['mae'], results_forecast['rmse']]
    actual_vals = [results_actual['mae'], results_actual['rmse']]
    
    x = np.arange(len(metrics_labels))
    width = 0.35
    axes[1, 1].bar(x - width/2, forecast_vals, width, label='Forecast Weather', color='#FF0000', alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1, 1].bar(x + width/2, actual_vals, width, label='Actual Weather', color='#0066FF', alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1, 1].set_ylabel("Error (MW)", fontsize=11)
    axes[1, 1].set_title("Metrics Comparison", fontsize=12, fontweight="bold")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics_labels)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved error comparison plot to {output_path}")
    plt.close()


def create_comparison_metrics_table(results_forecast, results_actual, output_path):
    """Create metrics comparison table."""
    metrics = {
        'Metric': [
            'Mean Absolute Error (MAE)',
            'Root Mean Square Error (RMSE)',
            'R² Score',
            'Mean Error (Bias)',
            'Median Absolute Error',
            'Std Dev of Errors',
        ],
        'Forecast Weather': [
            f"{results_forecast['mae']:.2f} MW",
            f"{results_forecast['rmse']:.2f} MW",
            f"{results_forecast['r2']:.4f}",
            f"{(results_forecast['predictions'] - results_forecast['actual']).mean():.2f} MW",
            f"{np.median(np.abs(results_forecast['predictions'] - results_forecast['actual'])):.2f} MW",
            f"{(results_forecast['predictions'] - results_forecast['actual']).std():.2f} MW",
        ],
        'Actual Weather': [
            f"{results_actual['mae']:.2f} MW",
            f"{results_actual['rmse']:.2f} MW",
            f"{results_actual['r2']:.4f}",
            f"{(results_actual['predictions'] - results_actual['actual']).mean():.2f} MW",
            f"{np.median(np.abs(results_actual['predictions'] - results_actual['actual'])):.2f} MW",
            f"{(results_actual['predictions'] - results_actual['actual']).std():.2f} MW",
        ]
    }
    df = pd.DataFrame(metrics)
    df.to_csv(output_path, index=False)
    print(f"Saved comparison metrics table to {output_path}")
    
    # Print to console
    print("\n" + "=" * 80)
    print("DETAILED METRICS COMPARISON")
    print("=" * 80)
    for i, row in df.iterrows():
        print(f"{row['Metric']:.<40} Forecast: {row['Forecast Weather']:>15}  Actual: {row['Actual Weather']:>15}")
    print("=" * 80)


def evaluate_zone(zone, use_large_model=False, use_actual_weather=False):
    """Evaluate a single zone with both forecast and actual weather."""
    print("\n" + "=" * 80)
    print(f"EVALUATING ENCODER-DECODER MODEL - {zone}")
    print("=" * 80)
    
    models_dir = Path("models") / zone.lower()
    results_dir = Path("results") / zone.lower()
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    print(f"\nUsing device: {device}")
    train_df = load_zone_data(zone, split="train")
    
    # Load test data for both scenarios
    test_df_forecast = load_zone_data(zone, split="test", use_actual_weather=False)
    test_df_actual = load_zone_data(zone, split="test", use_actual_weather=True)
    _, _, _, inst_cap, dates, n_features = build_test_sample(train_df, test_df_forecast)
    
    print(f"\nZone: {zone}")
    print(f"Installed capacity: {inst_cap:.2f} MW")
    print(f"Features: {n_features}")
    print(f"Forecast length: {len(dates)} hours")
    
    # Load normalization params and model
    mean, std = load_norm_params(models_dir / "norm.npz")
    model = load_model(zone, n_features, device, use_large_model)
    
    # Evaluate both scenarios
    results_forecast = evaluate_single_scenario(
        zone, train_df, test_df_forecast, model, mean, std, inst_cap, dates, "FORECAST WEATHER"
    )
    results_actual = evaluate_single_scenario(
        zone, train_df, test_df_actual, model, mean, std, inst_cap, dates, "ACTUAL WEATHER"
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    print(f"{'Scenario':<20} {'MAE (MW)':<12} {'RMSE (MW)':<12} {'R²':<10}")
    print("-" * 80)
    print(f"{'Forecast Weather':<20} {results_forecast['mae']:<12.2f} {results_forecast['rmse']:<12.2f} {results_forecast['r2']:<10.3f}")
    print(f"{'Actual Weather':<20} {results_actual['mae']:<12.2f} {results_actual['rmse']:<12.2f} {results_actual['r2']:<10.3f}")
    print("=" * 80)
    
    # Generate comprehensive reports
    print("\n" + "=" * 80)
    print("GENERATING REPORTS")
    print("=" * 80)
    
    # 1. Hour-by-hour CSV with both scenarios
    csv_path = results_dir / "predictions_comparison.csv"
    save_comparison_csv(dates, results_forecast, results_actual, csv_path)
    
    # 2. Comparison forecast time series plot
    forecast_path = results_dir / "forecast_comparison.png"
    plot_forecast_comparison(zone, dates, results_forecast, results_actual, forecast_path)
    
    # 3. Scatter plots side by side
    scatter_path = results_dir / "scatter_comparison.png"
    plot_scatter_comparison(zone, results_forecast, results_actual, scatter_path)
    
    # 4. Error analysis comparison
    error_path = results_dir / "error_comparison.png"
    plot_error_comparison(results_forecast, results_actual, error_path)
    
    # 5. Metrics table
    metrics_path = results_dir / "metrics_comparison.csv"
    create_comparison_metrics_table(results_forecast, results_actual, metrics_path)
    
    print("\n" + "=" * 80)
    print(f"{zone} - ALL REPORTS GENERATED SUCCESSFULLY")
    print("=" * 80)
    
    return {
        'zone': zone,
        'forecast_weather': {
            'mae': results_forecast['mae'],
            'rmse': results_forecast['rmse'],
            'r2': results_forecast['r2']
        },
        'actual_weather': {
            'mae': results_actual['mae'],
            'rmse': results_actual['rmse'],
            'r2': results_actual['r2']
        }
    }


def main():
    # Parse command line arguments
    zones_to_eval = ZONES
    use_large_model = False
    use_actual_weather = False
    
    if len(sys.argv) > 1:
        # First pass: collect flags
        for arg in sys.argv[1:]:
            if arg == "--large-model":
                use_large_model = True
            elif arg == "--actual-weather":
                use_actual_weather = True
        
        # Second pass: collect zones (skip flags)
        for i, arg in enumerate(sys.argv[1:], start=1):
            if arg == "--zones" and i < len(sys.argv):
                zones_to_eval = []
                for z in sys.argv[i+1:]:
                    if z.startswith("--"):
                        break
                    zones_to_eval.append(z.upper())
                break
    
    print("=" * 80)
    print("EVALUATING ALL ZONES")
    print("=" * 80)
    if use_large_model:
        print("Using LARGE model architecture (2.5M parameters)")
    else:
        print("Using small model architecture (242K parameters)")
    
    if use_actual_weather:
        print("Using ACTUAL OBSERVED WEATHER (best case scenario)")
    else:
        print("Using WEATHER FORECASTS (realistic scenario)")
    
    all_results = []
    for zone in zones_to_eval:
        try:
            result = evaluate_zone(zone, use_large_model, use_actual_weather)
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ ERROR evaluating {zone}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL ZONES")
    print("=" * 80)
    print(f"{'Zone':<12} {'Scenario':<20} {'MAE (MW)':<12} {'RMSE (MW)':<12} {'R²':<10}")
    print("-" * 80)
    for result in all_results:
        zone = result['zone']
        print(f"{zone:<12} {'Forecast Weather':<20} {result['forecast_weather']['mae']:<12.2f} {result['forecast_weather']['rmse']:<12.2f} {result['forecast_weather']['r2']:<10.3f}")
        print(f"{'':<12} {'Actual Weather':<20} {result['actual_weather']['mae']:<12.2f} {result['actual_weather']['rmse']:<12.2f} {result['actual_weather']['r2']:<10.3f}")
        print("-" * 80)
    print("=" * 80)
    
    # Save summary
    summary_data = []
    for result in all_results:
        summary_data.append({
            'zone': result['zone'],
            'scenario': 'forecast_weather',
            'mae_mw': result['forecast_weather']['mae'],
            'rmse_mw': result['forecast_weather']['rmse'],
            'r2': result['forecast_weather']['r2']
        })
        summary_data.append({
            'zone': result['zone'],
            'scenario': 'actual_weather',
            'mae_mw': result['actual_weather']['mae'],
            'rmse_mw': result['actual_weather']['rmse'],
            'r2': result['actual_weather']['r2']
        })
    summary_df = pd.DataFrame(summary_data)
    summary_path = Path("results") / "all_zones_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")
    print("\n✅ ALL EVALUATIONS COMPLETED!")


if __name__ == "__main__":
    main()
