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
    
    Args:
        zone: Zone name (e.g., 'IT-NORD')
        data_dir: Directory containing energy CSV files
    
    Returns:
        DataFrame with energy data
    """
    filename = f"{zone.lower().replace('-', '_')}_solar.csv"
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Energy file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    
    # Convert to timezone-naive for easier merging
    df['date'] = df['date'].dt.tz_localize(None)
    
    # Convert numeric columns
    for col in ['actual', 'day-ahead', 'intraday', 'installed_capacity_mw']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def load_weather_historic(zone, data_dir='data/raw/weather/historic'):
    """
    Load historical weather data for a specific zone.
    
    Args:
        zone: Zone name (e.g., 'IT-NORD')
        data_dir: Directory containing weather CSV files
    
    Returns:
        DataFrame with historical weather data
    """
    filename = f"{zone.lower().replace('-', '_')}_weather_historic.csv"
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Weather historic file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
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
    # Merge on date
    merged = energy_df.merge(weather_df, on='date', how='inner')
    
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
    
    # Check data coverage
    date_range = (merged_df['date'].max() - merged_df['date'].min()).days
    print(f"    Date range: {merged_df['date'].min().date()} to {merged_df['date'].max().date()} ({date_range} days)")
    
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


if __name__ == "__main__":
    # Process all zones
    results = process_all_zones()
    
    # Create combined dataset
    combined_df = create_combined_dataset()
    
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
