"""
Generate test data using ACTUAL observed weather (not forecast).

This script creates test sets with actual weather observations to evaluate
model performance with perfect weather information. This helps us understand:
1. How much accuracy we lose due to weather forecast errors
2. The upper bound of model performance with perfect weather
3. Whether the model architecture is the limiting factor

Comparison:
- Standard test: Uses weather FORECAST (realistic scenario)
- Actual weather test: Uses weather OBSERVATIONS (best case scenario)

If performance is still poor with actual weather, the problem is the model/features.
If performance improves significantly, the problem is weather forecast quality.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Italian bidding zones
ITALY_ZONES = ['IT-NORD', 'IT-CNOR', 'IT-CSUD', 'IT-SUD', 'IT-SICI', 'IT-SARD', 'IT-CALA']


def load_energy_data(zone, data_dir='data/raw/energy'):
    """Load energy data and aggregate to hourly if needed."""
    zone_clean = zone.lower().replace('-', '_')
    energy_file = os.path.join(data_dir, f"{zone_clean}_solar.csv")
    
    if not os.path.exists(energy_file):
        raise FileNotFoundError(f"Energy file not found: {energy_file}")
    
    df = pd.read_csv(energy_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # Aggregate 15-min to hourly for post-2025 data
    post_2025 = df[df['date'] >= '2025-01-01']
    if len(post_2025) > 0:
        sample_date = post_2025.iloc[min(100, len(post_2025)-1)]['date'].date()
        records_per_day = len(post_2025[post_2025['date'].dt.date == sample_date])
        
        if records_per_day > 24:
            df = df.set_index('date').resample('1H').agg({
                'actual': 'mean',
                'day-ahead': 'mean',
                'intraday': 'mean',
                'installed_capacity_mw': 'first'
            }).reset_index()
    
    return df


def load_weather_historic(zone, data_dir='data/raw/weather/historic'):
    """Load historical/actual observed weather data."""
    zone_clean = zone.lower().replace('-', '_')
    weather_file = os.path.join(data_dir, f"{zone_clean}_weather_hourly.csv")
    
    if not os.path.exists(weather_file):
        raise FileNotFoundError(f"Weather file not found: {weather_file}")
    
    df = pd.read_csv(weather_file)
    
    if 'datetime' in df.columns:
        df['date'] = pd.to_datetime(df['datetime'])
        df = df.drop(columns=['datetime'])
    else:
        df['date'] = pd.to_datetime(df['date'])
    
    return df


def calculate_capacity_factor(df):
    """Calculate capacity factor and add engineered features."""
    df = df.copy()
    
    # Capacity factor
    df['capacity_factor'] = (df['actual'] / df['installed_capacity_mw']).clip(0, 1)
    
    # Engineered features
    df['hour'] = df['date'].dt.hour
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Cyclical encoding for hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Cyclical encoding for day of year
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    return df


def generate_actual_weather_test(zone, train_end_date='2025-10-26', test_days=14):
    """
    Generate test set using ACTUAL observed weather instead of forecasts.
    
    Args:
        zone: Zone name (e.g., 'IT-NORD')
        train_end_date: Last date of training data
        test_days: Number of days in test set (default: 14)
    
    Returns:
        DataFrame with test data using actual weather
    """
    print(f"\nGenerating ACTUAL WEATHER test data for {zone}...")
    
    # Load energy data
    energy_df = load_energy_data(zone)
    
    # Load ACTUAL weather (not forecast)
    weather_df = load_weather_historic(zone)
    
    # Define test period
    train_end = pd.to_datetime(train_end_date)
    test_start = train_end + pd.Timedelta(days=1)
    test_end = test_start + pd.Timedelta(days=test_days-1, hours=23)
    
    print(f"  Test period: {test_start.date()} to {test_end.date()}")
    
    # Filter to test period
    energy_test = energy_df[(energy_df['date'] >= test_start) & 
                           (energy_df['date'] <= test_end)].copy()
    
    weather_test = weather_df[(weather_df['date'] >= test_start) & 
                             (weather_df['date'] <= test_end)].copy()
    
    # Merge energy + actual weather
    test_df = pd.merge(
        energy_test[['date', 'actual', 'day-ahead', 'installed_capacity_mw']], 
        weather_test,
        on='date',
        how='inner'
    )
    
    # Remove rows with missing values
    test_df = test_df.dropna()
    
    # Add capacity factor and engineered features
    test_df = calculate_capacity_factor(test_df)
    
    # Weather features (same as training - use exact same column names!)
    weather_features = [
        'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
        'temperature_2m', 'apparent_temperature', 'cloudcover',
        'dewpoint_2m', 'precipitation', 'windspeed_10m',
        'windgusts_10m', 'winddirection_10m', 'surface_pressure',
        'relativehumidity_2m'
    ]
    
    # The raw data has slightly different names, so keep them as-is
    # (they already match the training data format)
    
    # Feature columns to keep
    keep_columns = ['date', 'actual', 'day-ahead', 'installed_capacity_mw', 
                   'capacity_factor'] + weather_features + [
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
    ]
    
    # Filter to available columns
    available_cols = [col for col in keep_columns if col in test_df.columns]
    test_df = test_df[available_cols]
    
    print(f"  ✓ Created test set: {len(test_df)} records")
    print(f"  ✓ Date range: {test_df['date'].min()} to {test_df['date'].max()}")
    print(f"  ✓ Features: {len([c for c in test_df.columns if c not in ['date', 'actual', 'day-ahead', 'installed_capacity_mw', 'capacity_factor']])} weather + engineered")
    
    # Check for missing data
    missing_dates = pd.date_range(test_start, test_end, freq='1H')
    missing_count = len(missing_dates) - len(test_df)
    if missing_count > 0:
        print(f"  ⚠ Warning: Missing {missing_count} hours of data in test period")
    
    return test_df


def main():
    """Generate actual weather test sets for all zones."""
    print("=" * 80)
    print("GENERATING TEST DATA WITH ACTUAL WEATHER OBSERVATIONS")
    print("=" * 80)
    print("\nThis creates test sets with observed weather (not forecasts)")
    print("to evaluate model performance with perfect weather information.\n")
    
    output_dir = 'data/processed/test_actual_weather'
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for zone in ITALY_ZONES:
        try:
            test_df = generate_actual_weather_test(zone)
            
            # Save to CSV
            zone_clean = zone.lower().replace('-', '_')
            output_path = os.path.join(output_dir, f"{zone_clean}.csv")
            test_df.to_csv(output_path, index=False)
            print(f"  ✓ Saved to {output_path}")
            
            results.append({
                'zone': zone,
                'records': len(test_df),
                'start': test_df['date'].min(),
                'end': test_df['date'].max(),
                'success': True
            })
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                'zone': zone,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    print(f"\n✓ Successfully created: {len(successful)} zones")
    for r in successful:
        print(f"  {r['zone']}: {r['records']} records ({r['start'].date()} to {r['end'].date()})")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)} zones")
        for r in failed:
            print(f"  {r['zone']}: {r['error']}")
    
    print(f"\nTest data saved to: {output_dir}")
    print("\nTo evaluate with actual weather, modify evaluate_forecast.py")
    print("to load from 'data/processed/test_actual_weather' instead of")
    print("'data/processed/test'.")
    print("=" * 80)


if __name__ == "__main__":
    main()
