"""
Improved PyTorch prediction script that combines training history with test data.
Uses last 168 hours from training as input to predict the 336-hour test period.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

sys.path.append(str(Path(__file__).parent))

from data_loader import load_all_zones, WEATHER_FEATURES, ENGINEERED_FEATURES
from model import SimpleCNNLSTM, CNNLSTM, get_device, calculate_metrics


def load_model(model_path, sequence_length=168, n_features=14, forecast_horizon=336, model_type='simple'):
    """Load a trained PyTorch model."""
    device = get_device()
    
    if model_type == 'simple':
        model = SimpleCNNLSTM(
            sequence_length=sequence_length,
            n_features=n_features,
            forecast_horizon=forecast_horizon
        ).to(device)
    else:
        model = CNNLSTM(
            sequence_length=sequence_length,
            n_features=n_features,
            forecast_horizon=forecast_horizon
        ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val Loss: {checkpoint['val_loss']:.5f}")
    print(f"  Val MAE: {checkpoint['val_mae']:.5f}")
    
    return model


def predict_test_period(model_path='models/best_model_pytorch.pt',
                        zones=None,
                        sequence_length=168,
                        forecast_horizon=336,  # 14 days
                        model_type='simple'):
    """
    Predict on test period using last 168 hours from training as input.
    """
    
    device = get_device()
    print("\n" + "="*80)
    print("PYTORCH MODEL EVALUATION - 14-DAY FORECAST")
    print("="*80)
    print(f"Device: {device}")
    print(f"Input: Last 168 hours from training (Oct 19-26, 2025)")
    print(f"Output: Test period prediction (Oct 27 - Nov 10, 2025)")
    
    # Load training and test data
    print("\n1. Loading data...")
    train_data = load_all_zones(split='train')
    test_data = load_all_zones(split='test')
    
    if zones:
        train_data = {k: v for k, v in train_data.items() if k in zones}
        test_data = {k: v for k, v in test_data.items() if k in zones}
    
    print(f"   Loaded {len(train_data)} zones for evaluation")
    
    # Load normalization parameters
    norm_params_path = Path('models/normalization_params_pytorch.npz')
    if not norm_params_path.exists():
        raise FileNotFoundError(f"Normalization parameters not found at {norm_params_path}")
    
    norm_params_data = np.load(norm_params_path)
    mean = norm_params_data['mean']
    std = norm_params_data['std']
    print(f"   Loaded normalization parameters")
    
    # Load model
    print("\n2. Loading model...")
    n_features = len(WEATHER_FEATURES) + len(ENGINEERED_FEATURES)  # weather + engineered (NO hour)
    model = load_model(model_path, sequence_length, n_features, forecast_horizon, model_type)
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Evaluate each zone
    print("\n3. Making predictions...")
    all_results = {}
    
    for zone in train_data.keys():
        if zone not in test_data:
            print(f"\n   {zone}: No test data available, skipping...")
            continue
            
        train_df = train_data[zone]
        test_df = test_data[zone]
        
        print(f"\n   {zone}:")
        print(f"   - Training data: {len(train_df)} records")
        print(f"   - Test data: {len(test_df)} records")
        
        # Get last 168 hours from training as input
        input_df = train_df.tail(sequence_length).copy()
        
        if len(input_df) < sequence_length:
            print(f"   ⚠️  Not enough training history ({len(input_df)} < {sequence_length}), skipping...")
            continue
        
        # Prepare input features (15 features: 13 weather + 2 engineered, NO hour)
        feature_cols = WEATHER_FEATURES + ENGINEERED_FEATURES
        X_input = input_df[feature_cols].values
        X_input = X_input.reshape(1, sequence_length, n_features)  # (1, 168, 15)
        
        # Normalize
        X_input_norm = (X_input - mean) / std
        
        # Convert to tensor and predict
        X_input_tensor = torch.FloatTensor(X_input_norm).to(device)
        
        with torch.no_grad():
            y_pred = model(X_input_tensor).cpu().numpy()[0]  # (336,)
        
        # Get actual values from test data
        y_actual_cf = test_df['capacity_factor'].values[:forecast_horizon]
        
        if len(y_actual_cf) < forecast_horizon:
            print(f"   ⚠️  Test data too short ({len(y_actual_cf)} < {forecast_horizon}), using available data...")
            y_pred = y_pred[:len(y_actual_cf)]
        
        # Get installed capacity for this zone
        installed_capacity_mw = test_df['installed_capacity_mw'].iloc[0]
        
        # Convert capacity factors to MW
        y_actual_mw = y_actual_cf * installed_capacity_mw
        y_pred_mw = y_pred * installed_capacity_mw
        
        # Calculate metrics on capacity factors (normalized, 0-1)
        metrics = calculate_metrics(y_actual_cf, y_pred)
        
        print(f"   - MAE:  {metrics['mae']:.5f} ({metrics['mae']*100:.2f}%)")
        print(f"   - RMSE: {metrics['rmse']:.5f}")
        print(f"   - MAPE: {metrics['mape']:.2f}%")
        print(f"   - R²:   {metrics['r2']:.4f}")
        
        all_results[zone] = {
            'metrics': metrics,
            'y_actual': y_actual_mw,  # Store MW for plotting
            'y_pred': y_pred_mw,      # Store MW for plotting
            'dates': test_df['date'].values[:len(y_actual_cf)],
            'installed_capacity_mw': installed_capacity_mw
        }
        
        # Plot predictions for this zone
        plot_zone_forecast(zone, y_actual_mw, y_pred_mw, test_df['date'].values[:len(y_actual_cf)], results_dir)
    
    if len(all_results) == 0:
        print("\n   ⚠️  No zones could be evaluated!")
        return
    
    # Save results
    print("\n4. Saving results...")
    
    # Save metrics to JSON
    metrics_summary = {
        zone: {k: float(v) for k, v in results['metrics'].items()}
        for zone, results in all_results.items()
    }
    
    metrics_file = results_dir / 'evaluation_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"   Saved metrics to {metrics_file}")
    
    # Create summary table CSV
    create_summary_table(all_results, results_dir)
    
    # Create summary plot
    plot_summary(all_results, results_dir)
    
    # Print summary table
    print("\n" + "="*80)
    print("EVALUATION SUMMARY - 14-DAY FORECAST (Oct 27 - Nov 10, 2025)")
    print("="*80)
    print(f"{'Zone':<12} {'MAE':<10} {'RMSE':<10} {'MAPE (%)':<10} {'R²':<10}")
    print("-"*80)
    for zone, results in all_results.items():
        m = results['metrics']
        print(f"{zone:<12} {m['mae']:<10.5f} {m['rmse']:<10.5f} {m['mape']:<10.2f} {m['r2']:<10.4f}")
    
    # Overall average
    avg_mae = np.mean([r['metrics']['mae'] for r in all_results.values()])
    avg_rmse = np.mean([r['metrics']['rmse'] for r in all_results.values()])
    avg_mape = np.mean([r['metrics']['mape'] for r in all_results.values()])
    avg_r2 = np.mean([r['metrics']['r2'] for r in all_results.values()])
    
    print("-"*80)
    print(f"{'AVERAGE':<12} {avg_mae:<10.5f} {avg_rmse:<10.5f} {avg_mape:<10.2f} {avg_r2:<10.4f}")
    print("="*80 + "\n")
    
    print(f"📊 Results saved to {results_dir}/")
    print(f"   - evaluation_metrics.json (detailed metrics)")
    print(f"   - evaluation_summary.csv (table)")
    print(f"   - evaluation_summary.png (bar charts)")
    print(f"   - {zone}_forecast.png (time series for each zone)")


def plot_zone_forecast(zone, y_actual, y_pred, dates, results_dir):
    """Plot forecast vs actual for a single zone."""
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    hours = np.arange(len(y_actual))
    ax.plot(hours, y_actual, label='Actual', linewidth=2, alpha=0.8, color='#2E86AB')
    ax.plot(hours, y_pred, label='Predicted', linewidth=2, alpha=0.8, color='#A23B72')
    
    ax.set_xlabel('Date (Oct-Nov 2025)', fontsize=12)
    ax.set_ylabel('Solar Production (MW)', fontsize=12)
    ax.set_title(f'{zone} - 14-Day Solar Production Forecast (Oct 27 - Nov 10, 2025)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add vertical lines for each day and labels
    day_labels = []
    day_positions = []
    for day in range(0, len(y_actual), 24):
        ax.axvline(day, color='gray', linestyle='--', alpha=0.2, linewidth=0.8)
        if day < len(y_actual):
            day_num = day // 24 + 1
            # Get the date for this day
            date_str = pd.to_datetime(dates[day]).strftime('%b %d')
            day_labels.append(date_str)
            day_positions.append(day + 12)  # Position label at noon
    
    # Set x-axis to show day labels
    ax.set_xticks(day_positions)
    ax.set_xticklabels(day_labels, rotation=45, ha='right')
    
    # Calculate and display metrics on plot
    mae = np.mean(np.abs(y_actual - y_pred))
    rmse = np.sqrt(np.mean((y_actual - y_pred) ** 2))
    r2 = 1 - (np.sum((y_actual - y_pred) ** 2) / np.sum((y_actual - np.mean(y_actual)) ** 2))
    
    metrics_text = f'MAE: {mae:.2f} MW | RMSE: {rmse:.2f} MW | R²: {r2:.4f}'
    ax.text(0.02, 0.02, metrics_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(results_dir / f'{zone}_forecast.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_table(all_results, results_dir):
    """Create CSV summary table."""
    
    rows = []
    for zone, results in all_results.items():
        m = results['metrics']
        rows.append({
            'Zone': zone,
            'MAE': f"{m['mae']:.5f}",
            'RMSE': f"{m['rmse']:.5f}",
            'MAPE (%)': f"{m['mape']:.2f}",
            'R²': f"{m['r2']:.4f}"
        })
    
    # Add average
    avg_mae = np.mean([r['metrics']['mae'] for r in all_results.values()])
    avg_rmse = np.mean([r['metrics']['rmse'] for r in all_results.values()])
    avg_mape = np.mean([r['metrics']['mape'] for r in all_results.values()])
    avg_r2 = np.mean([r['metrics']['r2'] for r in all_results.values()])
    
    rows.append({
        'Zone': 'AVERAGE',
        'MAE': f"{avg_mae:.5f}",
        'RMSE': f"{avg_rmse:.5f}",
        'MAPE (%)': f"{avg_mape:.2f}",
        'R²': f"{avg_r2:.4f}"
    })
    
    df = pd.DataFrame(rows)
    csv_file = results_dir / 'evaluation_summary.csv'
    df.to_csv(csv_file, index=False)
    print(f"   Saved summary table to {csv_file}")


def plot_summary(all_results, results_dir):
    """Create summary visualization of all zones."""
    
    zones = list(all_results.keys())
    metrics_names = ['MAE', 'RMSE', 'MAPE', 'R²']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for idx, metric_name in enumerate(metrics_names):
        metric_key = metric_name.lower().replace('²', '2').replace(' (%)', '')
        values = [all_results[zone]['metrics'][metric_key] for zone in zones]
        
        bars = axes[idx].bar(zones, values, color=colors[idx], alpha=0.7, edgecolor='black', linewidth=1.2)
        axes[idx].set_xlabel('Zone', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel(metric_name, fontsize=11, fontweight='bold')
        axes[idx].set_title(f'{metric_name} by Zone', fontsize=12, fontweight='bold')
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels on bars
        for bar, v in zip(bars, values):
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{v:.3f}',
                          ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add target line for MAE and MAPE
        if metric_name in ['MAE', 'MAPE']:
            target = 0.05 if metric_name == 'MAE' else 5.0
            axes[idx].axhline(target, color='red', linestyle='--', linewidth=2, 
                            label=f'Target (<{target})', alpha=0.7)
            axes[idx].legend(fontsize=9)
    
    plt.suptitle('14-Day Solar Forecast Evaluation (Oct 27 - Nov 10, 2025)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(results_dir / 'evaluation_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved summary plot to {results_dir / 'evaluation_summary.png'}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained PyTorch model on test period')
    parser.add_argument('--model-path', type=str, default='models/best_model_pytorch.pt',
                       help='Path to saved model (default: models/best_model_pytorch.pt)')
    parser.add_argument('--zones', type=str, nargs='+', default=None,
                       help='Zones to evaluate (default: all)')
    parser.add_argument('--model-type', type=str, default='simple',
                       choices=['simple', 'full'],
                       help='Model architecture (default: simple)')
    
    args = parser.parse_args()
    
    predict_test_period(
        model_path=args.model_path,
        zones=args.zones,
        model_type=args.model_type
    )
