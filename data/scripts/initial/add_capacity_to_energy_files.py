"""
Script to add installed capacity column to energy CSV files.

This script:
1. Reads the capacity data from data/external/italy_pv_capacity.csv
2. For each zone's energy file, merges the capacity data by date
3. Adds 'installed_capacity_mw' column to the existing CSV
4. Backs up original files before modifying

Location: scripts/add_capacity_to_energy_files.py
"""

import os
import sys
import pandas as pd
import shutil
from datetime import datetime

# Add root directory to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

# Italian bidding zones
ITALY_ZONES = {
    'IT-NORD': '10Y1001A1001A73I',
    'IT-CNOR': '10Y1001A1001A70O',
    'IT-CSUD': '10Y1001A1001A71M',
    'IT-SUD': '10Y1001A1001A788',
    'IT-SICI': '10Y1001A1001A74G',
    'IT-SARD': '10Y1001A1001A75E',
    'IT-CALA': '10Y1001C--00096J'
}


def load_capacity_data():
    """
    Load capacity data from CSV file.
    
    Returns:
        DataFrame with capacity data by zone and date
    """
    capacity_file = os.path.join(root_dir, 'data', 'external', 'italy_pv_capacity.csv')
    
    if not os.path.exists(capacity_file):
        print(f"✗ Capacity file not found: {capacity_file}")
        print("  Run scripts/prepare_capacity_data.py first")
        return None
    
    print(f"Loading capacity data from: {capacity_file}")
    df = pd.read_csv(capacity_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # Keep only necessary columns
    df = df[['date', 'zone', 'installed_capacity_mw']]
    
    print(f"  ✓ Loaded {len(df):,} capacity records for {df['zone'].nunique()} zones")
    return df


def backup_file(filepath):
    """
    Create a backup of the file before modifying.
    
    Args:
        filepath: Path to file to backup
    
    Returns:
        str: Path to backup file
    """
    backup_dir = os.path.join(root_dir, 'data', 'raw', 'energy', 'backup')
    os.makedirs(backup_dir, exist_ok=True)
    
    filename = os.path.basename(filepath)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = os.path.join(backup_dir, f"{filename}.{timestamp}.backup")
    
    shutil.copy2(filepath, backup_path)
    return backup_path


def add_capacity_to_energy_file(zone_name, capacity_df, create_backup=True):
    """
    Add installed capacity column to a zone's energy CSV file.
    
    Args:
        zone_name: Name of the zone (e.g., 'IT-NORD')
        capacity_df: DataFrame with capacity data
        create_backup: Whether to create backup before modifying
    
    Returns:
        bool: True if successful
    """
    # Get file path
    filename = f"{zone_name.lower().replace('-', '_')}_solar.csv"
    filepath = os.path.join(root_dir, 'data', 'raw', 'energy', filename)
    
    if not os.path.exists(filepath):
        print(f"\n✗ File not found: {filename}")
        return False
    
    print(f"\nProcessing {zone_name}...")
    
    # Create backup
    if create_backup:
        backup_path = backup_file(filepath)
        print(f"  ✓ Backup created: {os.path.basename(backup_path)}")
    
    # Read energy data
    print(f"  Reading energy data...")
    energy_df = pd.read_csv(filepath)
    original_rows = len(energy_df)
    
    # Parse dates
    energy_df['date'] = pd.to_datetime(energy_df['date'], utc=True)
    
    # Get capacity data for this zone
    zone_capacity = capacity_df[capacity_df['zone'] == zone_name].copy()
    
    # For merging, we need to match on date only (not time)
    # Create a date column without time
    energy_df['date_only'] = energy_df['date'].dt.date
    zone_capacity['date_only'] = zone_capacity['date'].dt.date
    
    # Merge capacity data
    print(f"  Merging capacity data...")
    merged_df = energy_df.merge(
        zone_capacity[['date_only', 'installed_capacity_mw']],
        on='date_only',
        how='left'
    )
    
    # Drop the temporary date_only column
    merged_df = merged_df.drop('date_only', axis=1)
    
    # Check if capacity column already exists
    if 'installed_capacity_mw' in energy_df.columns:
        print(f"  ⚠ Capacity column already exists, replacing...")
        # Replace the old column
        energy_df['installed_capacity_mw'] = merged_df['installed_capacity_mw']
        final_df = energy_df
    else:
        final_df = merged_df
    
    # Reorder columns: date, actual, day-ahead, intraday, installed_capacity_mw
    cols = ['date', 'actual', 'day-ahead', 'intraday', 'installed_capacity_mw']
    final_df = final_df[cols]
    
    # Check merge success
    capacity_count = final_df['installed_capacity_mw'].notna().sum()
    coverage = (capacity_count / len(final_df)) * 100
    
    print(f"  Rows: {original_rows:,}")
    print(f"  Capacity values: {capacity_count:,} ({coverage:.1f}% coverage)")
    
    if coverage < 50:
        print(f"  ⚠ Warning: Low capacity coverage - check date alignment")
    
    # Save modified file
    print(f"  Saving updated file...")
    final_df.to_csv(filepath, index=False)
    
    print(f"  ✓ {zone_name} updated successfully!")
    return True


def verify_updates():
    """
    Verify that all energy files have been updated with capacity data.
    
    Returns:
        dict: Summary of verification results
    """
    print("\n" + "="*70)
    print("VERIFYING UPDATES")
    print("="*70)
    
    results = {}
    
    for zone_name in ITALY_ZONES.keys():
        filename = f"{zone_name.lower().replace('-', '_')}_solar.csv"
        filepath = os.path.join(root_dir, 'data', 'raw', 'energy', filename)
        
        if not os.path.exists(filepath):
            results[zone_name] = {'status': 'missing', 'coverage': 0}
            continue
        
        df = pd.read_csv(filepath)
        
        if 'installed_capacity_mw' not in df.columns:
            results[zone_name] = {'status': 'no_capacity_column', 'coverage': 0}
        else:
            capacity_count = df['installed_capacity_mw'].notna().sum()
            coverage = (capacity_count / len(df)) * 100
            results[zone_name] = {
                'status': 'success',
                'coverage': coverage,
                'rows': len(df),
                'capacity_rows': capacity_count
            }
    
    # Print summary
    print("\nZone-by-Zone Summary:")
    for zone_name, result in sorted(results.items()):
        if result['status'] == 'success':
            print(f"  ✓ {zone_name}: {result['capacity_rows']:,}/{result['rows']:,} rows ({result['coverage']:.1f}%)")
        elif result['status'] == 'no_capacity_column':
            print(f"  ✗ {zone_name}: No capacity column found")
        else:
            print(f"  ✗ {zone_name}: File not found")
    
    return results


def show_sample_data():
    """Display sample data from updated files."""
    print("\n" + "="*70)
    print("SAMPLE DATA (IT-NORD)")
    print("="*70)
    
    filepath = os.path.join(root_dir, 'data', 'raw', 'energy', 'it_nord_solar.csv')
    
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print("\nFirst 5 rows:")
        print(df.head(5).to_string(index=False))
        
        # Show a row with capacity data
        if 'installed_capacity_mw' in df.columns:
            with_capacity = df[df['installed_capacity_mw'].notna()]
            if not with_capacity.empty:
                print("\nSample row with capacity (summer 2023):")
                summer_2023 = with_capacity[with_capacity['date'].str.contains('2023-06')]
                if not summer_2023.empty:
                    print(summer_2023.head(3).to_string(index=False))


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("ADD INSTALLED CAPACITY TO ENERGY FILES")
    print("="*70)
    print("\nThis will add 'installed_capacity_mw' column to each zone's CSV file")
    print("Original files will be backed up to data/raw/energy/backup/\n")
    
    # Step 1: Load capacity data
    capacity_df = load_capacity_data()
    
    if capacity_df is None:
        print("\n✗ Failed to load capacity data")
        return
    
    # Step 2: Process each zone
    print("\n" + "="*70)
    print("PROCESSING ZONES")
    print("="*70)
    
    success_count = 0
    for zone_name in sorted(ITALY_ZONES.keys()):
        if add_capacity_to_energy_file(zone_name, capacity_df, create_backup=True):
            success_count += 1
    
    # Step 3: Verify updates
    results = verify_updates()
    
    # Step 4: Show sample
    show_sample_data()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nProcessed {success_count}/{len(ITALY_ZONES)} zones successfully")
    print(f"Backups stored in: data/raw/energy/backup/")
    
    print("\n✓ CAPACITY DATA ADDED TO ENERGY FILES!")
    print("="*70)
    print("\nYour energy CSV files now have these columns:")
    print("  1. date - timestamp")
    print("  2. actual - actual generation (MW)")
    print("  3. day-ahead - day-ahead forecast (MW)")
    print("  4. intraday - intraday forecast (MW)")
    print("  5. installed_capacity_mw - installed capacity (MW) ← NEW!")
    print("\nYou can now calculate capacity factors directly:")
    print("  capacity_factor = actual / installed_capacity_mw")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
