"""
Fix IT-CALA train/test data by regenerating from updated raw energy data.
Only affects IT-CALA files - does not touch other zones.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add data scripts to path
sys.path.append(str(Path(__file__).parent / "data/scripts"))
from preprocessing import (
    load_energy_data,
    load_weather_forecast,
    load_weather_historic,
    merge_energy_weather,
    calculate_capacity_factor
)
from add_temporal_features import add_temporal_features


def fix_it_cala():
    """
    Regenerate IT-CALA train/test splits with updated energy data.
    Train/test split: Oct 26, 2025 23:00 (same as other zones)
    """
    print("=" * 80)
    print("FIXING IT-CALA TRAIN/TEST DATA")
    print("=" * 80)
    
    zone = "IT-CALA"
    train_end_date = pd.Timestamp("2025-10-26 23:00:00")
    test_start_date = pd.Timestamp("2025-10-27 00:00:00")
    test_end_date = pd.Timestamp("2025-11-09 23:00:00")
    
    print(f"\n1. Loading IT-CALA energy data...")
    df_energy = load_energy_data(zone)
    df_energy['date'] = pd.to_datetime(df_energy['date'])
    print(f"   Total energy records: {len(df_energy)}")
    print(f"   Range: {df_energy['date'].iloc[0]} to {df_energy['date'].iloc[-1]}")
    
    # Filter to only rows with actual data
    df_energy = df_energy[df_energy['actual'].notna()].copy()
    print(f"   Records with actual generation: {len(df_energy)}")
    
    # Load weather data
    print(f"\n2. Loading weather data...")
    df_weather_forecast = load_weather_forecast(zone)
    df_weather_historic = load_weather_historic(zone)
    df_weather_forecast['date'] = pd.to_datetime(df_weather_forecast['date'])
    df_weather_historic['date'] = pd.to_datetime(df_weather_historic['date'])
    print(f"   Forecast weather: {len(df_weather_forecast)} records")
    print(f"   Historic weather: {len(df_weather_historic)} records")
    
    # Create TRAIN set (historic weather)
    print(f"\n3. Creating train set (up to {train_end_date})...")
    df_energy_train = df_energy[df_energy['date'] <= train_end_date].copy()
    df_train = merge_energy_weather(df_energy_train, df_weather_historic)
    df_train = calculate_capacity_factor(df_train)
    df_train = add_temporal_features(df_train)
    df_train['zone'] = zone
    df_train = df_train.set_index('date')
    
    print(f"   Train: {len(df_train)} records")
    print(f"   Range: {df_train.index[0]} to {df_train.index[-1]}")
    print(f"   Columns: {len(df_train.columns)}")
    
    # Create TEST set (forecast weather)
    print(f"\n4. Creating test set with FORECAST weather...")
    df_energy_test = df_energy[
        (df_energy['date'] >= test_start_date) & 
        (df_energy['date'] <= test_end_date)
    ].copy()
    
    df_test_forecast = merge_energy_weather(df_energy_test, df_weather_forecast)
    df_test_forecast = calculate_capacity_factor(df_test_forecast)
    df_test_forecast = add_temporal_features(df_test_forecast)
    df_test_forecast['zone'] = zone
    df_test_forecast = df_test_forecast.set_index('date')
    
    print(f"   Test (forecast): {len(df_test_forecast)} records")
    if len(df_test_forecast) > 0:
        print(f"   Range: {df_test_forecast.index[0]} to {df_test_forecast.index[-1]}")
    print(f"   Columns: {len(df_test_forecast.columns)}")
    
    # Create TEST set (historic weather - actual weather scenario)
    print(f"\n5. Creating test set with HISTORIC weather...")
    df_test_historic = merge_energy_weather(df_energy_test, df_weather_historic)
    df_test_historic = calculate_capacity_factor(df_test_historic)
    df_test_historic = add_temporal_features(df_test_historic)
    df_test_historic['zone'] = zone
    df_test_historic = df_test_historic.set_index('date')
    
    print(f"   Test (historic): {len(df_test_historic)} records")
    if len(df_test_historic) > 0:
        print(f"   Range: {df_test_historic.index[0]} to {df_test_historic.index[-1]}")
    
    # Save files
    print(f"\n6. Saving files...")
    
    train_path = Path("data/processed/train/it_cala.csv")
    train_path.parent.mkdir(exist_ok=True, parents=True)
    df_train.to_csv(train_path)
    print(f"   ✓ Saved train: {train_path}")
    
    test_forecast_path = Path("data/processed/test/it_cala.csv")
    test_forecast_path.parent.mkdir(exist_ok=True, parents=True)
    df_test_forecast.to_csv(test_forecast_path)
    print(f"   ✓ Saved test (forecast): {test_forecast_path}")
    
    test_historic_path = Path("data/processed/test_actual_weather/it_cala.csv")
    test_historic_path.parent.mkdir(exist_ok=True, parents=True)
    df_test_historic.to_csv(test_historic_path)
    print(f"   ✓ Saved test (historic): {test_historic_path}")
    
    # Verification
    print(f"\n7. Verification...")
    train_check = pd.read_csv(train_path, index_col=0, parse_dates=True)
    test_f_check = pd.read_csv(test_forecast_path, index_col=0, parse_dates=True)
    test_h_check = pd.read_csv(test_historic_path, index_col=0, parse_dates=True)
    
    print(f"   Train: {len(train_check)} records, {len(train_check.columns)} columns")
    print(f"   Test (forecast): {len(test_f_check)} records, {len(test_f_check.columns)} columns")
    print(f"   Test (historic): {len(test_h_check)} records, {len(test_h_check.columns)} columns")
    
    # Check if we have 336 hours
    if len(test_f_check) == 336 and len(test_h_check) == 336:
        print(f"   ✅ Both test sets have 336 hours (14 days)")
    else:
        print(f"   ⚠️  Test sets: {len(test_f_check)} (forecast), {len(test_h_check)} (historic)")
    
    # Check temporal features
    required_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    has_temporal = all(col in train_check.columns for col in required_cols)
    if has_temporal:
        print(f"   ✅ Temporal features present")
    else:
        print(f"   ⚠️  Missing temporal features")
    
    print("\n" + "=" * 80)
    print("✅ IT-CALA DATA FIXED")
    print("=" * 80)
    print("\nNext step: Retrain IT-CALA model with updated data")
    print("Command: python src/train.py --zones IT-CALA --epochs 30 --attention --batch-size 16")


if __name__ == "__main__":
    fix_it_cala()
