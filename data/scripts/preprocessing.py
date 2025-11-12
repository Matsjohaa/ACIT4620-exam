"""
Data preprocessing module for Italian solar forecasting project.

This module handles:
- Loading and merging energy and weather data
- Calculating capacity factors
- Aligning timestamps
- Handling missing values
- Preparing data for CNN-LSTM model
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
    """
    Load energy data for a specific zone.
    
    After Jan 1, 2025, ENTSO-E data is in 15-minute intervals.
    This function aggregates to hourly to match historical data and weather data.
    
    Args:
        zone: Bidding zone name (e.g., 'IT-NORD')
        data_dir: Directory containing energy CSV files
    
    Returns:
        DataFrame with energy data (hourly resolution)
    """
    zone_clean = zone.lower().replace('-', '_')
    energy_file = os.path.join(data_dir, f"{zone_clean}_solar.csv")
    
    if not os.path.exists(energy_file):
        raise FileNotFoundError(f"Energy file not found: {energy_file}")
    
    df = pd.read_csv(energy_file)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    
    # Check if data has 15-minute intervals (>24 records per day)
    # Count records for a sample date after Jan 1, 2025
    post_2025 = df[df['date'] >= '2025-01-01']
    if len(post_2025) > 0:
        sample_date = post_2025.iloc[100]['date'].date() if len(post_2025) > 100 else post_2025.iloc[0]['date'].date()
        records_per_day = len(post_2025[post_2025['date'].dt.date == sample_date])
        
        if records_per_day > 24:
            # 15-minute intervals detected, aggregate to hourly
            # Resample to hourly using mean for generation values
            df_hourly = df.set_index('date').resample('1H').agg({
                'actual': 'mean',           # Average generation over the hour
                'day-ahead': 'mean',        # Average forecast
                'intraday': 'mean',         # Average intraday forecast
                'installed_capacity_mw': 'first'  # Capacity doesn't change within hour
            }).reset_index()
            
            df = df_hourly
    
    df['zone'] = zone
    
    return df


def load_weather_historic(zone, data_dir='data/raw/weather/historic'):
    """
    Load historical weather data for a zone.
    
    Args:
        zone: Bidding zone name (e.g., 'IT-NORD')
        data_dir: Directory containing weather data
    
    Returns:
        DataFrame with weather data
    """
    zone_clean = zone.lower().replace('-', '_')
    weather_file = os.path.join(data_dir, f"{zone_clean}_weather_hourly.csv")
    
    if not os.path.exists(weather_file):
        raise FileNotFoundError(f"Weather historic file not found: {weather_file}")
    
    df = pd.read_csv(weather_file)
    
    # Use 'datetime' column as the timestamp (not 'date' which is just the date part)
    if 'datetime' in df.columns:
        df['date'] = pd.to_datetime(df['datetime'])
        df = df.drop(columns=['datetime'])
    else:
        # Fallback if datetime doesn't exist
        df['date'] = pd.to_datetime(df['date'])
    
    # Convert to UTC to match energy data
    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.tz_localize('UTC')
    
    df['zone'] = zone
    
    return df


def load_weather_forecast(zone, data_dir='data/raw/weather/forecast'):
    """
    Load weather forecast data for a specific zone.
    
    Args:
        zone: Zone name (e.g., 'IT-NORD')
        data_dir: Directory containing weather forecast CSV files
    
    Returns:
        DataFrame with weather forecast data
    """
    filename = f"{zone.lower().replace('-', '_')}_weather_forecast.csv"
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Weather forecast file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    return df


def calculate_capacity_factor(df):
    """
    Calculate capacity factor from actual generation and installed capacity.
    
    Args:
        df: DataFrame with 'actual' and 'installed_capacity_mw' columns
    
    Returns:
        DataFrame with added 'capacity_factor' column
    """
    df = df.copy()
    
    # Calculate capacity factor
    df['capacity_factor'] = df['actual'] / df['installed_capacity_mw']
    
    # Clip to [0, 1] range (handle any numerical issues)
    df['capacity_factor'] = df['capacity_factor'].clip(0, 1)
    
    return df


def merge_energy_weather(energy_df, weather_df):
    """
    Merge energy and weather data by timestamp.
    
    Args:
        energy_df: DataFrame with energy data
        weather_df: DataFrame with weather data
    
    Returns:
        Merged DataFrame
    """
    # Ensure both dataframes have timezone-aware UTC timestamps
    if energy_df['date'].dt.tz is None:
        energy_df = energy_df.copy()
        energy_df['date'] = energy_df['date'].dt.tz_localize('UTC')
    elif str(energy_df['date'].dt.tz) != 'UTC':
        energy_df = energy_df.copy()
        energy_df['date'] = energy_df['date'].dt.tz_convert('UTC')
    
    if weather_df['date'].dt.tz is None:
        weather_df = weather_df.copy()
        weather_df['date'] = weather_df['date'].dt.tz_localize('UTC')
    elif str(weather_df['date'].dt.tz) != 'UTC':
        weather_df = weather_df.copy()
        weather_df['date'] = weather_df['date'].dt.tz_convert('UTC')
    
    # Merge on date
    merged = energy_df.merge(weather_df, on='date', how='inner', suffixes=('', '_weather'))
    
    # Drop duplicate zone column if it exists
    if 'zone_weather' in merged.columns:
        merged = merged.drop(columns=['zone_weather'])
    
    # Sort by date
    merged = merged.sort_values('date').reset_index(drop=True)
    
    return merged


def process_zone(zone, output_dir='data/processed'):
    """
    Process all data for a single zone.
    
    Args:
        zone: Zone name (e.g., 'IT-NORD')
        output_dir: Directory to save processed data
    
    Returns:
        DataFrame with processed data
    """
    print(f"\nProcessing {zone}...")
    
    # Load data
    print("  Loading energy data...")
    energy_df = load_energy_data(zone)
    print(f"    ✓ {len(energy_df):,} energy records")
    
    # Filter out rows where actual is NaN
    energy_df = energy_df[energy_df['actual'].notna()].copy()
    print(f"    ✓ {len(energy_df):,} records with actual generation")
    
    print("  Loading weather data...")
    weather_df = load_weather_historic(zone)
    print(f"    ✓ {len(weather_df):,} weather records")
    
    # Calculate capacity factor
    print("  Calculating capacity factors...")
    energy_df = calculate_capacity_factor(energy_df)
    
    # Merge energy and weather
    print("  Merging energy and weather data...")
    merged_df = merge_energy_weather(energy_df, weather_df)
    print(f"    ✓ {len(merged_df):,} merged records")
    
    # Add zone identifier
    merged_df['zone'] = zone
    
    # Sort by date
    merged_df = merged_df.sort_values('date').reset_index(drop=True)
    
    # Check data coverage
    date_range = (merged_df['date'].max() - merged_df['date'].min()).days
    print(f"    Date range: {merged_df['date'].min().date()} to {merged_df['date'].max().date()} ({date_range} days)")
    print(f"    Capacity factor range: {merged_df['capacity_factor'].min():.3f} to {merged_df['capacity_factor'].max():.3f}")
    
    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{zone.lower().replace('-', '_')}_processed.csv")
    merged_df.to_csv(output_file, index=False)
    print(f"    ✓ Saved to {output_file}")
    
    return merged_df


def process_all_zones(zones=None, output_dir='data/processed'):
    """
    Process data for all zones.
    
    Args:
        zones: List of zone names (default: all Italian zones)
        output_dir: Directory to save processed data
    
    Returns:
        Dictionary mapping zone names to DataFrames
    """
    if zones is None:
        zones = ITALY_ZONES
    
    print("="*70)
    print("PROCESSING ALL ZONES")
    print("="*70)
    
    results = {}
    
    for zone in zones:
        try:
            df = process_zone(zone, output_dir)
            results[zone] = df
        except Exception as e:
            print(f"  ✗ Error processing {zone}: {e}")
            continue
    
    print("\n" + "="*70)
    print(f"✓ Successfully processed {len(results)}/{len(zones)} zones")
    print("="*70)
    
    return results


def create_combined_dataset(zones=None, output_dir='data/processed'):
    """
    Create a single combined dataset with all zones.
    
    Args:
        zones: List of zone names (default: all Italian zones)
        output_dir: Directory to save combined data
    
    Returns:
        Combined DataFrame
    """
    if zones is None:
        zones = ITALY_ZONES
    
    print("\n" + "="*70)
    print("CREATING COMBINED DATASET")
    print("="*70)
    
    all_data = []
    
    for zone in zones:
        input_file = os.path.join(output_dir, f"{zone.lower().replace('-', '_')}_processed.csv")
        
        if os.path.exists(input_file):
            df = pd.read_csv(input_file)
            df['date'] = pd.to_datetime(df['date'])
            all_data.append(df)
            print(f"  ✓ Loaded {zone}: {len(df):,} records")
        else:
            print(f"  ⚠ {zone}: file not found, skipping")
    
    if not all_data:
        print("  ✗ No data files found!")
        return None
    
    # Combine all zones
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values(['zone', 'date']).reset_index(drop=True)
    
    # Save combined dataset
    output_file = os.path.join(output_dir, 'italy_all_zones_processed.csv')
    combined_df.to_csv(output_file, index=False)
    
    print(f"\n  ✓ Combined dataset: {len(combined_df):,} total records")
    print(f"  ✓ Zones: {combined_df['zone'].nunique()}")
    print(f"  ✓ Saved to: {output_file}")
    
    return combined_df


def get_data_summary(df):
    """
    Get summary statistics for processed data.
    
    Args:
        df: Processed DataFrame
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_records': len(df),
        'zones': df['zone'].nunique() if 'zone' in df.columns else 1,
        'date_range': {
            'start': df['date'].min(),
            'end': df['date'].max(),
            'days': (df['date'].max() - df['date'].min()).days
        },
        'missing_values': df.isnull().sum().to_dict(),
        'capacity_factor_stats': df['capacity_factor'].describe().to_dict() if 'capacity_factor' in df.columns else None
    }
    
    return summary


def create_train_test_split(df, train_end_date='2025-10-26', zone='IT-NORD', base_output_dir='data/processed'):
    """
    Split data into training and test sets.
    
    CRITICAL: Training uses HISTORIC weather, Test uses WEATHER FORECAST
    
    For this project:
    - Train: All data up to train_end_date with HISTORIC weather (actual observations)
    - Test: Oct 27 - Nov 10 with WEATHER FORECAST + actual generation (for validation)
    
    Test set simulates real forecasting scenario:
    - Uses weather FORECAST made on Oct 27 (14-day ahead prediction)
    - Has actual generation values for validation only
    - NO historical production features
    - NO day-ahead or intraday forecasts
    
    Args:
        df: Processed DataFrame with HISTORIC weather
        train_end_date: Last date to include in training (default: 2025-10-26)
        zone: Zone name for filename
        base_output_dir: Base directory (train/test folders created inside)
    
    Returns:
        Tuple of (train_df, test_df)
    """
    print(f"\nCreating train/test split for {zone} (train_end_date: {train_end_date})...")
    
    # Training set: Use the processed data with historic weather
    df['date'] = pd.to_datetime(df['date'], utc=True)
    train_end = pd.to_datetime(train_end_date, utc=True)
    train_df = df[df['date'] <= train_end].copy()
    
    print(f"  Training set: {len(train_df):,} records ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"  Training features: {len(train_df.columns)} columns (HISTORIC weather)")
    
    # Test set: Load energy data + WEATHER FORECAST (not historic)
    zone_clean = zone.lower().replace('-', '_')
    
    # Load weather FORECAST for test period
    forecast_file = f'data/raw/weather/forecast/{zone_clean}_weather_forecast.csv'
    if not os.path.exists(forecast_file):
        print(f"  ⚠ Weather forecast not found: {forecast_file}")
        print(f"  Creating empty test set")
        
        # Create separate train/test directories
        train_dir = os.path.join(base_output_dir, 'train')
        test_dir = os.path.join(base_output_dir, 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        train_path = os.path.join(train_dir, f"{zone_clean}.csv")
        train_df.to_csv(train_path, index=False)
        print(f"  ✓ Saved training data to {train_path}")
        
        return train_df, None
    
    weather_forecast = pd.read_csv(forecast_file)
    
    # Parse date from datetime column
    if 'datetime' in weather_forecast.columns:
        weather_forecast['date'] = pd.to_datetime(weather_forecast['datetime'])
        weather_forecast = weather_forecast.drop(columns=['datetime'])
    else:
        weather_forecast['date'] = pd.to_datetime(weather_forecast['date'])
    
    if weather_forecast['date'].dt.tz is None:
        weather_forecast['date'] = weather_forecast['date'].dt.tz_localize('UTC')
    
    # Load energy data for test period
    energy_file = f'data/raw/energy/{zone_clean}_solar.csv'
    energy_df = pd.read_csv(energy_file)
    energy_df['date'] = pd.to_datetime(energy_df['date'], utc=True)
    
    # Filter energy to test period
    test_start = train_end + pd.Timedelta(days=1)
    test_end = test_start + pd.Timedelta(days=13, hours=23)  # 14 days total
    
    energy_test = energy_df[(energy_df['date'] >= test_start) & (energy_df['date'] <= test_end)].copy()
    energy_test = energy_test[energy_test['actual'].notna()].copy()  # Only rows with actual data
    
    # Aggregate 15-min to hourly if needed
    if len(energy_test) > 0:
        sample_date = energy_test.iloc[min(10, len(energy_test)-1)]['date'].date()
        records_per_day = len(energy_test[energy_test['date'].dt.date == sample_date])
        
        if records_per_day > 24:
            energy_test = energy_test.set_index('date').resample('1H').agg({
                'actual': 'mean',
                'installed_capacity_mw': 'first'
            }).reset_index()
            energy_test = energy_test[energy_test['actual'].notna()]  # Remove any NaN from resampling
    
    # Merge energy + weather FORECAST
    test_df = pd.merge(energy_test[['date', 'actual', 'installed_capacity_mw']], 
                       weather_forecast, 
                       on='date', 
                       how='inner')
    
    if len(test_df) == 0:
        print(f"  ⚠ No matching dates between energy and weather forecast")
        print(f"    Energy: {energy_test['date'].min()} to {energy_test['date'].max()}")
        print(f"    Forecast: {weather_forecast['date'].min()} to {weather_forecast['date'].max()}")
    
    # Calculate capacity factor
    test_df['capacity_factor'] = (test_df['actual'] / test_df['installed_capacity_mw']).clip(0, 1)
    test_df['zone'] = zone
    
    # Sort by date
    test_df = test_df.sort_values('date').reset_index(drop=True)
    
    print(f"  Test set: {len(test_df):,} records ({test_df['date'].min().date()} to {test_df['date'].max().date()})")
    print(f"  Test features: {len(test_df.columns)} columns (WEATHER FORECAST + capacity)")
    
    # Create separate train/test directories
    train_dir = os.path.join(base_output_dir, 'train')
    test_dir = os.path.join(base_output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Save splits
    train_path = os.path.join(train_dir, f"{zone_clean}.csv")
    test_path = os.path.join(test_dir, f"{zone_clean}.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"  ✓ Saved training data to {train_path}")
    print(f"  ✓ Saved test data to {test_path}")
    
    return train_df, test_df


if __name__ == "__main__":
    import sys
    
    print("="*70)
    print("SOLAR FORECASTING - DATA PREPROCESSING")
    print("="*70)
    print("\nProject Goal: Train on data up to Oct 26, 2025")
    print("              Predict Oct 27 - Nov 10, 2025 (14 days)")
    print("              Validate against actual values")
    print("="*70)
    
    # Process all zones
    results = process_all_zones()
    
    # Create combined dataset
    combined_df = create_combined_dataset()
    
    print("\n" + "="*70)
    print("CREATING TRAIN/TEST SPLITS")
    print("="*70)
    print("\nTrain: Historical data + HISTORIC weather (for learning)")
    print("Test: WEATHER FORECAST + actual (for validation)")
    print("="*70)
    
    train_end_date = '2025-10-26'
    
    for zone in ITALY_ZONES:
        input_file = f'data/processed/{zone.lower().replace("-", "_")}_processed.csv'
        if os.path.exists(input_file):
            df = pd.read_csv(input_file)
            df['date'] = pd.to_datetime(df['date'])
            create_train_test_split(df, train_end_date, zone)
    
    # Split combined dataset
    if combined_df is not None:
        create_train_test_split(combined_df, train_end_date, 'italy_all_zones')
    
    # Show summary
    if combined_df is not None:
        print("\n" + "="*70)
        print("DATA SUMMARY")
        print("="*70)
        
        summary = get_data_summary(combined_df)
        print(f"\nTotal records: {summary['total_records']:,}")
        print(f"Zones: {summary['zones']}")
        print(f"Date range: {summary['date_range']['start'].date()} to {summary['date_range']['end'].date()}")
        print(f"Duration: {summary['date_range']['days']} days")
        
        if summary['capacity_factor_stats']:
            print(f"\nCapacity Factor Statistics:")
            print(f"  Mean: {summary['capacity_factor_stats']['mean']:.4f}")
            print(f"  Std:  {summary['capacity_factor_stats']['std']:.4f}")
            print(f"  Min:  {summary['capacity_factor_stats']['min']:.4f}")
            print(f"  Max:  {summary['capacity_factor_stats']['max']:.4f}")
        
        print("\n" + "="*70)
        print("✓ Preprocessing complete!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Run feature engineering: python src/features.py")
        print("  2. Create sequences for CNN-LSTM: python src/sequences.py")
        print("  3. Train model: python src/train.py")
        print("  4. Make predictions for Oct 28 - Nov 10")
        print("  5. Compare predictions vs actual values")
