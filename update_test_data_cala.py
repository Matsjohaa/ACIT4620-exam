"""
Update test data for IT-CALA to have 336 hours (14 days) for evaluation.
This script fetches forecast weather for test set and historic weather for test_actual_weather.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
import os

sys.path.append(str(Path(__file__).parent / "data/scripts"))
from preprocessing import (
    load_energy_data, 
    load_weather_forecast, 
    load_weather_historic,
    merge_energy_weather,
    calculate_capacity_factor
)

def update_cala_test_data():
    """
    Create proper test sets for IT-CALA with forecast and historic weather data.
    - test/: Uses forecast weather (realistic scenario)
    - test_actual_weather/: Uses historic weather (best-case scenario)
    Both cover the same 336-hour time period.
    """
    print("=" * 80)
    print("UPDATING IT-CALA TEST DATA")
    print("=" * 80)
    
    test_hours = 336
    zone = "IT-CALA"
    
    # Load energy data (full dataset)
    print("\n1. Loading IT-CALA energy data...")
    df_energy = load_energy_data(zone)
    print(f"   Total records: {len(df_energy)}")
    print(f"   Range: {df_energy['date'].iloc[0]} to {df_energy['date'].iloc[-1]}")
    
    # Calculate split point
    print(f"\n2. Determining split point ({test_hours} hours)...")
    if len(df_energy) < test_hours:
        print(f"❌ Not enough data! Only {len(df_energy)} hours available.")
        return
    
    split_idx = len(df_energy) - test_hours
    split_date = df_energy['date'].iloc[split_idx]
    print(f"   Split date: {split_date}")
    print(f"   Train: {df_energy['date'].iloc[0]} to {df_energy['date'].iloc[split_idx-1]}")
    print(f"   Test: {split_date} to {df_energy['date'].iloc[-1]}")
    
    # Load weather data
    print("\n3. Loading weather data...")
    df_weather_forecast = load_weather_forecast(zone)
    df_weather_historic = load_weather_historic(zone)
    print(f"   ✓ Forecast weather: {len(df_weather_forecast)} records")
    print(f"   ✓ Historic weather: {len(df_weather_historic)} records")
    
    # Split energy data
    df_energy_train = df_energy.iloc[:split_idx].copy()
    df_energy_test = df_energy.iloc[split_idx:].copy()
    
    # Create test set with FORECAST weather
    print("\n4. Creating test set with FORECAST weather...")
    df_test_forecast = merge_energy_weather(df_energy_test, df_weather_forecast)
    df_test_forecast = calculate_capacity_factor(df_test_forecast)
    df_test_forecast['zone'] = zone
    df_test_forecast = df_test_forecast.set_index('date')
    print(f"   Test (forecast): {len(df_test_forecast)} hours")
    
    # Create test set with HISTORIC weather
    print("\n5. Creating test set with HISTORIC weather...")
    df_test_historic = merge_energy_weather(df_energy_test, df_weather_historic)
    df_test_historic = calculate_capacity_factor(df_test_historic)
    df_test_historic['zone'] = zone
    df_test_historic = df_test_historic.set_index('date')
    print(f"   Test (historic): {len(df_test_historic)} hours")
    
    # Create updated train set (with historic weather)
    print("\n6. Creating updated train set...")
    df_train_updated = merge_energy_weather(df_energy_train, df_weather_historic)
    df_train_updated = calculate_capacity_factor(df_train_updated)
    df_train_updated['zone'] = zone
    df_train_updated = df_train_updated.set_index('date')
    print(f"   Train: {len(df_train_updated)} hours")
    
    # Save files
    print("\n7. Saving updated files...")
    
    # Test with forecast weather
    test_forecast_path = Path("data/processed/test/it_cala.csv")
    test_forecast_path.parent.mkdir(exist_ok=True, parents=True)
    df_test_forecast.to_csv(test_forecast_path)
    print(f"   ✓ Saved test (forecast): {test_forecast_path}")
    
    # Test with historic weather
    test_historic_path = Path("data/processed/test_actual_weather/it_cala.csv")
    test_historic_path.parent.mkdir(exist_ok=True, parents=True)
    df_test_historic.to_csv(test_historic_path)
    print(f"   ✓ Saved test (historic): {test_historic_path}")
    
    # Updated train
    train_updated_path = Path("data/processed/train/it_cala.csv")
    df_train_updated.to_csv(train_updated_path)
    print(f"   ✓ Saved updated train: {train_updated_path}")
    
    # Verify
    print("\n8. Verification...")
    test_f = pd.read_csv(test_forecast_path, index_col=0, parse_dates=True)
    test_h = pd.read_csv(test_historic_path, index_col=0, parse_dates=True)
    train_v = pd.read_csv(train_updated_path, index_col=0, parse_dates=True)
    
    print(f"   Test (forecast): {len(test_f)} hours ({test_f.index[0]} to {test_f.index[-1]})")
    print(f"   Test (historic): {len(test_h)} hours ({test_h.index[0]} to {test_h.index[-1]})")
    print(f"   Train: {len(train_v)} hours ({train_v.index[0]} to {train_v.index[-1]})")
    
    if len(test_f) == test_hours and len(test_h) == test_hours:
        print(f"   ✓ Both test sets have correct length ({test_hours} hours)")
    else:
        print(f"   ⚠ Warning: Expected {test_hours} hours, got {len(test_f)} (forecast) and {len(test_h)} (historic)")
    
    print("\n" + "=" * 80)
    print("✅ IT-CALA TEST DATA UPDATED SUCCESSFULLY")
    print("=" * 80)
    print("\nNote: You'll need to RETRAIN the IT-CALA model with the updated train set.")
    print("      The train set has been shortened to exclude the new test period.")

if __name__ == "__main__":
    update_cala_test_data()
