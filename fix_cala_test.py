"""
Fix IT-CALA test data: Create test sets with forecast/historic weather.
DOES NOT modify the train set - keeps it as-is.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent / "data/scripts"))
from preprocessing import (
    load_energy_data, 
    load_weather_forecast, 
    load_weather_historic,
    merge_energy_weather,
    calculate_capacity_factor
)
from add_temporal_features import add_temporal_features


def fix_cala_test_data():
    """
    Create proper test sets for IT-CALA:
    - test/: Uses forecast weather (realistic scenario)  
    - test_actual_weather/: Uses historic weather (best-case scenario)
    Both should have 336 hours starting from Oct 26, 2025 00:00.
    """
    print("=" * 80)
    print("FIXING IT-CALA TEST DATA")
    print("=" * 80)
    
    zone = "IT-CALA"
    
    # Define test period (after train ends at Oct 25, 2025 23:00)
    test_start = pd.Timestamp("2025-10-26 00:00:00")
    test_end = pd.Timestamp("2025-11-09 23:00:00")  # 336 hours = 14 days
    
    print(f"\n1. Target test period:")
    print(f"   Start: {test_start}")
    print(f"   End: {test_end}")
    print(f"   Duration: 336 hours (14 days)")
    
    # Load energy data
    print(f"\n2. Loading IT-CALA energy data...")
    df_energy_full = load_energy_data(zone)
    df_energy_full['date'] = pd.to_datetime(df_energy_full['date'])
    
    # Filter to test period
    df_energy_test = df_energy_full[
        (df_energy_full['date'] >= test_start) & 
        (df_energy_full['date'] <= test_end)
    ].copy()
    
    print(f"   Energy records in test period: {len(df_energy_test)}")
    if len(df_energy_test) > 0:
        print(f"   Range: {df_energy_test['date'].iloc[0]} to {df_energy_test['date'].iloc[-1]}")
    
    # Load weather data
    print(f"\n3. Loading weather data...")
    df_weather_forecast = load_weather_forecast(zone)
    df_weather_historic = load_weather_historic(zone)
    
    df_weather_forecast['date'] = pd.to_datetime(df_weather_forecast['date'])
    df_weather_historic['date'] = pd.to_datetime(df_weather_historic['date'])
    
    print(f"   Forecast weather: {len(df_weather_forecast)} records")
    print(f"   Forecast range: {df_weather_forecast['date'].iloc[0]} to {df_weather_forecast['date'].iloc[-1]}")
    print(f"   Historic weather: {len(df_weather_historic)} records")
    
    # Create test set with FORECAST weather
    print(f"\n4. Creating test set with FORECAST weather...")
    df_test_forecast = merge_energy_weather(df_energy_test, df_weather_forecast)
    df_test_forecast = calculate_capacity_factor(df_test_forecast)
    
    # Add temporal features
    df_test_forecast = add_temporal_features(df_test_forecast)
    df_test_forecast['zone'] = zone
    df_test_forecast = df_test_forecast.set_index('date')
    
    print(f"   Test (forecast): {len(df_test_forecast)} hours")
    if len(df_test_forecast) > 0:
        print(f"   Range: {df_test_forecast.index[0]} to {df_test_forecast.index[-1]}")
    
    # Create test set with HISTORIC weather
    print(f"\n5. Creating test set with HISTORIC weather...")
    df_test_historic = merge_energy_weather(df_energy_test, df_weather_historic)
    df_test_historic = calculate_capacity_factor(df_test_historic)
    
    # Add temporal features
    df_test_historic = add_temporal_features(df_test_historic)
    df_test_historic['zone'] = zone
    df_test_historic = df_test_historic.set_index('date')
    
    print(f"   Test (historic): {len(df_test_historic)} hours")
    if len(df_test_historic) > 0:
        print(f"   Range: {df_test_historic.index[0]} to {df_test_historic.index[-1]}")
    
    # Save files
    print(f"\n6. Saving test files...")
    
    test_forecast_path = Path("data/processed/test/it_cala.csv")
    test_forecast_path.parent.mkdir(exist_ok=True, parents=True)
    df_test_forecast.to_csv(test_forecast_path)
    print(f"   ✓ Saved: {test_forecast_path}")
    
    test_historic_path = Path("data/processed/test_actual_weather/it_cala.csv")
    test_historic_path.parent.mkdir(exist_ok=True, parents=True)
    df_test_historic.to_csv(test_historic_path)
    print(f"   ✓ Saved: {test_historic_path}")
    
    # Verification
    print(f"\n7. Verification...")
    test_f = pd.read_csv(test_forecast_path, index_col=0, parse_dates=True)
    test_h = pd.read_csv(test_historic_path, index_col=0, parse_dates=True)
    
    print(f"   Test (forecast): {len(test_f)} hours")
    print(f"   Test (historic): {len(test_h)} hours")
    
    if len(test_f) == 336 and len(test_h) == 336:
        print(f"   ✅ Both test sets have correct length (336 hours)")
    else:
        print(f"   ⚠️  Expected 336 hours, got {len(test_f)} (forecast) and {len(test_h)} (historic)")
    
    # Check train set was not modified
    train_path = Path("data/processed/train/it_cala.csv")
    train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
    print(f"\n   Train set: {len(train_df)} hours (unchanged)")
    print(f"   Train ends: {train_df.index[-1]}")
    
    print("\n" + "=" * 80)
    print("✅ IT-CALA TEST DATA FIXED")
    print("=" * 80)
    print("\n✓ Train set: UNCHANGED (as it should be)")
    print("✓ Test sets: Updated with proper forecast/historic weather")
    print("✓ Temporal features: Added to both test sets")


if __name__ == "__main__":
    fix_cala_test_data()
