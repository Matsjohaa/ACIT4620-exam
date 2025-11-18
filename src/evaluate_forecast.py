"""
Evaluate the Encoder-Decoder model on test data (Oct 27 - Nov 10, 2025)
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

from model import EncoderDecoderCNNLSTM
from data_loader import (
    load_zone_data,
    WEATHER_FEATURES,
    ENGINEERED_FEATURES,
    compute_day_ahead_capacity_factor,
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
    """
    train_df = train_df.sort_values("date").reset_index(drop=True)
    test_df = test_df.sort_values("date").reset_index(drop=True)

    assert len(test_df) == HORIZON, f"Expected {HORIZON} test hours, got {len(test_df)}"

    train_tail = train_df.tail(SEQ_LEN)

    features = WEATHER_FEATURES + ENGINEERED_FEATURES
    features = [f for f in features if f in train_tail.columns]

    X_enc_raw = train_tail[features].values         # [168, F]
    X_dec_raw = test_df[features].values            # [336, F]

    y_true_MW = test_df["actual"].values            # [336]

    day_ahead_cf_full = compute_day_ahead_capacity_factor(test_df)
    day_ahead_cf = day_ahead_cf_full                # [336]

    inst_cap = test_df["installed_capacity_mw"].values[0]  # scalar
    dates = pd.to_datetime(test_df["date"].values)

    return X_enc_raw, X_dec_raw, y_true_MW, day_ahead_cf, inst_cap, dates, len(features)


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
    Zero out predictions during nighttime hours and when solar radiation is negligible.
    """
    test_df = test_df.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(test_df["date"].values)
    hours = dates.hour.values
    
    radiation = test_df["shortwave_radiation"].values
    
    # Nighttime mask (hours outside 6-18)
    nighttime_mask = (hours < 6) | (hours > 18)
    
    # Low radiation mask (radiation <= 1e-3)
    low_radiation_mask = radiation <= 1e-3
    
    # Combine masks
    zero_mask = nighttime_mask | low_radiation_mask
    
    # Apply mask
    cf_pred_zeroed = cf_pred.copy()
    cf_pred_zeroed[zero_mask] = 0.0
    
    n_zeroed = np.sum(zero_mask)
    print(f"  → Zeroed {n_zeroed}/{len(cf_pred)} predictions ({100*n_zeroed/len(cf_pred):.1f}%)")
    
    return cf_pred_zeroed


def load_model(zone, n_features, device):
    """Load the encoder-decoder model."""
    models_dir = Path("models") / zone.lower()
    model_path = models_dir / "model.pt"
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


def evaluate_zone(zone):
    """Evaluate a single zone."""
    print("\n" + "=" * 80)
    print(f"EVALUATING ENCODER-DECODER MODEL - {zone}")
    print("=" * 80)
    
    models_dir = Path("models") / zone.lower()
    results_dir = Path("results") / zone.lower()
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    print(f"\nUsing device: {device}")
    train_df = load_zone_data(zone, split="train")
    test_df = load_zone_data(zone, split="test")
    
    # Build test sequences
    X_enc_raw, X_dec_raw, y_true_MW, day_ahead_cf, inst_cap, dates, n_features = build_test_sample(train_df, test_df)
    
    print(f"\nZone: {zone}")
    print(f"Installed capacity: {inst_cap:.2f} MW")
    print(f"Features: {n_features}")
    print(f"Forecast length: {len(y_true_MW)} hours")
    
    # Convert to capacity factor for residual calculations
    y_true_cf = y_true_MW / inst_cap
    
    # Load normalization params
    mean, std = load_norm_params(models_dir / "norm.npz")
    mean = mean.reshape(1, 1, -1)
    std = std.reshape(1, 1, -1)
    
    # Normalize
    X_enc = (X_enc_raw[None, ...] - mean) / std
    X_dec = (X_dec_raw[None, ...] - mean) / std
    
    # Load model
    model = load_model(zone, n_features, device)
    
    # Predict
    print("\nGenerating predictions...")
    with torch.no_grad():
        Xe = torch.tensor(X_enc, dtype=torch.float32, device=device)
        Xd = torch.tensor(X_dec, dtype=torch.float32, device=device)
        resid_cf = model(Xe, Xd).cpu().numpy().reshape(-1)
    
    # Add residual to day-ahead forecast
    cf_pred = np.clip(day_ahead_cf + resid_cf, 0.0, 1.2)
    
    # Apply nighttime zeroing
    cf_pred = apply_nighttime_zeroing(cf_pred, test_df.copy())
    
    # Convert to MW
    y_pred_MW = cf_pred * inst_cap
    
    # Calculate metrics
    mae, rmse, r2 = metrics(y_true_MW, y_pred_MW)
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"MAE:  {mae:.2f} MW")
    print(f"RMSE: {rmse:.2f} MW")
    print(f"R²:   {r2:.3f}")
    print("=" * 80)
    
    # Check residual correlation
    corr = np.corrcoef(y_true_cf - day_ahead_cf, resid_cf)[0, 1]
    print(f"\nResidual correlation: {corr:.4f}")
    
    # Generate comprehensive reports
    print("\n" + "=" * 80)
    print("GENERATING REPORTS")
    print("=" * 80)
    
    # 1. Hour-by-hour CSV
    csv_path = results_dir / "predictions.csv"
    save_predictions_csv(dates, y_true_MW, y_pred_MW, csv_path)
    
    # 2. Forecast time series plot
    forecast_path = results_dir / "forecast.png"
    plot_forecast(zone, dates, y_true_MW, y_pred_MW, mae, rmse, r2, forecast_path)
    
    # 3. Scatter plot
    scatter_path = results_dir / "scatter_plot.png"
    plot_scatter(zone, y_true_MW, y_pred_MW, mae, rmse, r2, scatter_path)
    
    # 4. Error analysis
    error_path = results_dir / "error_analysis.png"
    plot_error_distribution(y_true_MW, y_pred_MW, error_path)
    
    # 5. Hourly performance
    hourly_path = results_dir / "hourly_performance.png"
    plot_hourly_performance(dates, y_true_MW, y_pred_MW, hourly_path)
    
    # 6. Metrics table
    metrics_path = results_dir / "metrics.csv"
    create_metrics_table(mae, rmse, r2, y_true_MW, y_pred_MW, metrics_path)
    
    print("\n" + "=" * 80)
    print(f"{zone} - ALL REPORTS GENERATED SUCCESSFULLY")
    print("=" * 80)
    
    return {
        'zone': zone,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def main():
    print("=" * 80)
    print("EVALUATING ALL ZONES")
    print("=" * 80)
    
    all_results = []
    for zone in ZONES:
        try:
            result = evaluate_zone(zone)
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ ERROR evaluating {zone}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL ZONES")
    print("=" * 80)
    print(f"{'Zone':<12} {'MAE (MW)':<12} {'RMSE (MW)':<12} {'R²':<10}")
    print("-" * 80)
    for result in all_results:
        print(f"{result['zone']:<12} {result['mae']:<12.2f} {result['rmse']:<12.2f} {result['r2']:<10.3f}")
    print("=" * 80)
    
    # Save summary
    summary_df = pd.DataFrame(all_results)
    summary_path = Path("results") / "all_zones_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")
    print("\n✅ ALL EVALUATIONS COMPLETED!")


if __name__ == "__main__":
    main()
