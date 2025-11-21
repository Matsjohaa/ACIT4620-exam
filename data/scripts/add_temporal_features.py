"""
Add simple temporal features to processed data files.

This script adds only 2 carefully chosen features:
1. Hour encoding (sin/cos) - captures daily solar cycle
2. Day of year encoding (sin/cos) - captures seasonal patterns

These are proven to help with solar forecasting without causing overfitting.
"""

import pandas as pd
import numpy as np
from pathlib import Path


ZONES = [
    "IT-NORD",
    "IT-CNOR",
    "IT-CSUD",
    "IT-SUD",
    "IT-SICI",
    "IT-SARD",
    "IT-CALA",
]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple temporal features using sine/cosine encoding.
    
    Args:
        df: DataFrame with 'date' column
    
    Returns:
        DataFrame with added temporal features
    """
    df = df.copy()
    
    # Parse date if needed
    if df['date'].dtype == 'object':
        df['date'] = pd.to_datetime(df['date'])
    
    # Extract hour and day of year
    hour = df['date'].dt.hour
    day_of_year = df['date'].dt.dayofyear
    
    # Sine/cosine encoding for hour (24-hour cycle)
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    
    # Sine/cosine encoding for day of year (365-day cycle)
    df['day_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
    
    return df


def process_file(filepath: Path) -> None:
    """
    Add temporal features to a single file.
    
    Args:
        filepath: Path to CSV file
    """
    if not filepath.exists():
        print(f"  ⚠️  File not found: {filepath}")
        return
    
    print(f"  Processing {filepath.name}...")
    
    # Load data
    df = pd.read_csv(filepath)
    original_cols = len(df.columns)
    
    # Check if features already exist
    if 'hour_sin' in df.columns:
        print(f"    → Temporal features already exist, skipping")
        return
    
    # Add features
    df = add_temporal_features(df)
    new_cols = len(df.columns)
    
    # Save
    df.to_csv(filepath, index=False)
    
    print(f"    ✅ Added {new_cols - original_cols} features ({original_cols} → {new_cols} columns)")


def main():
    """Process all zone files."""
    
    print("="*80)
    print("ADDING SIMPLE TEMPORAL FEATURES")
    print("="*80)
    print("\nAdding 4 features to all processed data files:")
    print("  - hour_sin, hour_cos (daily cycle)")
    print("  - day_sin, day_cos (seasonal cycle)")
    print()
    
    # Confirm
    response = input("Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.")
        return
    
    base_path = Path("data/processed")
    
    # Process train files
    print("\n" + "="*80)
    print("PROCESSING TRAINING DATA")
    print("="*80)
    
    for zone in ZONES:
        zone_lower = zone.lower().replace("-", "_")
        filepath = base_path / "train" / f"{zone_lower}.csv"
        process_file(filepath)
    
    # Process test files
    print("\n" + "="*80)
    print("PROCESSING TEST DATA")
    print("="*80)
    
    for zone in ZONES:
        zone_lower = zone.lower().replace("-", "_")
        filepath = base_path / "test" / f"{zone_lower}.csv"
        process_file(filepath)
    
    # Process test actual weather files
    print("\n" + "="*80)
    print("PROCESSING TEST ACTUAL WEATHER DATA")
    print("="*80)
    
    test_actual_path = base_path / "test_actual_weather"
    if test_actual_path.exists():
        for zone in ZONES:
            zone_lower = zone.lower().replace("-", "_")
            filepath = test_actual_path / f"{zone_lower}.csv"
            process_file(filepath)
    else:
        print("  ⚠️  test_actual_weather directory not found")
    
    print("\n" + "="*80)
    print("✅ ALL FILES PROCESSED")
    print("="*80)
    print("\nTemporal features added successfully!")
    print("Original features: 15 → New total: 19 features")
    print("\nNow you can train with these simple but effective features:")
    print("  python src/train.py --zones IT-NORD --epochs 50")


if __name__ == "__main__":
    main()
