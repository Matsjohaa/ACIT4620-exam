"""
Incremental update script for Italian solar power data.
This script only fetches new data since the last update, making it much faster.

The script:
1. Checks the latest date in existing CSV files
2. Fetches data from the day AFTER the latest date to avoid duplicates
3. Merges new data with existing data
4. Ensures no gaps or duplicates in the timeline

Location: data/scripts/update_data/update_power.py
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import pytz
from entsoe.entsoe import EntsoePandasClient
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

# Add the root directory to Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, root_dir)

# Load environment variables
load_dotenv()

# Italian bidding zones with their ENTSO-E codes (active zones only)
ITALY_ZONES = {
    'IT-NORD': '10Y1001A1001A73I',
    'IT-CNOR': '10Y1001A1001A70O', 
    'IT-CSUD': '10Y1001A1001A71M',
    'IT-SUD': '10Y1001A1001A788',   # Available from 2021+
    'IT-SICI': '10Y1001A1001A74G',
    'IT-SARD': '10Y1001A1001A75E',
    'IT-CALA': '10Y1001C--00096J'   # Available from 2021+ (corrected code)
}

# Date restrictions for newer zones
ZONE_START_DATES = {
    'IT-SUD': datetime(2021, 1, 1, tzinfo=pytz.UTC),
    'IT-CALA': datetime(2021, 1, 1, tzinfo=pytz.UTC)
}

def get_entsoe_client():
    """Initialize ENTSO-E client with API key from environment."""
    api_key = os.getenv('ENTSOE')
    if not api_key:
        raise ValueError("ENTSO-E API key not found. Please set ENTSOE in your .env file")
    return EntsoePandasClient(api_key=api_key)

def fetch_solar_data(client, zone_code, start_date, end_date, data_type):
    """
    Fetch solar generation data from ENTSO-E API.
    
    Args:
        client: ENTSO-E client
        zone_code: Bidding zone code
        start_date: Start date (timezone-aware)
        end_date: End date (timezone-aware)
        data_type: 'actual', 'day_ahead', or 'intraday'
    
    Returns:
        pandas.Series: Time series data
    """
    try:
        # Limit the date range to avoid API timeouts (max 1 year per request)
        max_days = 365
        date_range = (end_date - start_date).days
        
        if date_range > max_days:
            print(f"    Date range too large ({date_range} days), fetching last {max_days} days only")
            start_date = end_date - pd.Timedelta(days=max_days)
        
        if data_type == 'actual':
            data = client.query_generation(
                zone_code, 
                start=start_date, 
                end=end_date, 
                psr_type='B16'  # Solar
            )
            # Handle both Series and DataFrame cases
            if isinstance(data, pd.DataFrame):
                if 'Solar' in data.columns:
                    return data['Solar']
                elif len(data.columns) > 1:
                    # For actual generation, we might get multiple columns (generation vs consumption)
                    # We ONLY want generation data, not consumption
                    print(f"    Multiple columns found: {list(data.columns)}")
                    col_sums = data.sum()
                    print(f"    Column sums: {dict(col_sums)}")
                    
                    # Look for generation-related column names first
                    generation_cols = [col for col in data.columns if 
                                     'generation' in str(col).lower() or 
                                     'produced' in str(col).lower() or
                                     'inBiddingZone' in str(col)]
                    
                    if generation_cols:
                        selected_col = generation_cols[0]
                        print(f"    Found generation column: {selected_col}")
                    else:
                        # Fallback: assume the column with positive values is generation
                        max_col = col_sums.idxmax()
                        selected_col = max_col
                        print(f"    No clear generation column found, using column with highest values: {selected_col}")
                    
                    return data[selected_col]
                elif len(data.columns) == 1:
                    return data.iloc[:, 0]
                else:
                    # Fallback: sum all columns
                    return data.sum(axis=1)
            return data
            
        elif data_type == 'day_ahead':
            data = client.query_generation_forecast(
                zone_code, 
                start=start_date, 
                end=end_date,
                process_type='A01'  # Day-ahead forecast
            )
            # Filter for solar if we get multiple generation types
            if isinstance(data, pd.DataFrame) and 'Solar' in data.columns:
                return data['Solar']
            return data
            
        elif data_type == 'intraday':
            data = client.query_generation_forecast(
                zone_code, 
                start=start_date, 
                end=end_date,
                process_type='A31'  # Intraday forecast
            )
            # Filter for solar if we get multiple generation types
            if isinstance(data, pd.DataFrame) and 'Solar' in data.columns:
                return data['Solar']
            return data
    except Exception as e:
        print(f"    Error fetching {data_type} data: {e}")
        return pd.Series(dtype=float)

def get_last_update_date(zone_name, data_dir=None):
    """
    Get the last date we have data for in the existing CSV file.
    
    Args:
        zone_name: Name of the zone (e.g., 'IT-NORD')
        data_dir: Directory containing the CSV files (relative to root)
    
    Returns:
        pandas.Timestamp or None: Last date in the file, or None if file doesn't exist
    """
    if data_dir is None:
        # Default path relative to root directory
        data_dir = os.path.join(root_dir, 'data', 'raw', 'energy')
    
    filename = f"{zone_name.lower().replace('-', '_')}_solar.csv"
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"  No existing file found for {zone_name}: {filename}")
        return None
    
    try:
        # Read the CSV file - it should have 'date' as first column
        df = pd.read_csv(filepath)
        if len(df) == 0:
            print(f"  Empty file for {zone_name}")
            return None
        
        # Convert date column to datetime and find the max
        df['date'] = pd.to_datetime(df['date'])
        last_date = df['date'].max()
        
        # Return as naive datetime (no timezone) for consistency
        if last_date.tz is not None:
            last_date = last_date.tz_localize(None)
        
        print(f"  Last data date for {zone_name}: {last_date.date()}")
        return last_date
    
    except Exception as e:
        print(f"  Error reading file for {zone_name}: {e}")
        return None

def update_zone_data(zone_name, zone_code, start_date, end_date, forecast_types):
    """
    Update data for a specific zone from start_date to end_date.
    
    Args:
        zone_name: Name of the zone
        zone_code: ENTSO-E zone code  
        start_date: Start date for update
        end_date: End date for update
        forecast_types: List of forecast types to update
    
    Returns:
        dict: Dictionary with new data for each forecast type
    """
    client = get_entsoe_client()
    zone_data = {}
    
    for forecast_type in forecast_types:
        print(f"    Fetching {forecast_type} data...")
        
        data = fetch_solar_data(client, zone_code, start_date, end_date, forecast_type)
        
        if not data.empty:
            # Keep the original resolution (15-minute intervals)
            # Do NOT aggregate to daily - we want to preserve the time series detail
            zone_data[forecast_type] = data
        else:
            zone_data[forecast_type] = pd.Series(dtype=float)
        
        # Small delay to avoid hitting API rate limits
        import time
        time.sleep(0.5)
    
    return zone_data

def merge_with_existing_data(zone_name, new_zone_data, data_dir=None):
    """
    Merge new data with existing CSV file.
    
    Args:
        zone_name: Name of the zone
        new_zone_data: Dictionary with new data for each forecast type
        data_dir: Directory containing the CSV files
    """
    if data_dir is None:
        # Default path relative to root directory
        data_dir = os.path.join(root_dir, 'data', 'raw', 'energy')
    
    # Ensure output directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    filename = f"{zone_name.lower().replace('-', '_')}_solar.csv"
    filepath = os.path.join(data_dir, filename)
    
    # Create combined DataFrame from new data
    combined_data = pd.DataFrame()
    
    # Get all timestamps from new data
    all_timestamps = set()
    for forecast_type, data in new_zone_data.items():
        if isinstance(data, pd.Series) and not data.empty:
            all_timestamps.update(data.index)
    
    if all_timestamps:
        # Create DataFrame with all timestamps
        combined_data = pd.DataFrame(index=sorted(all_timestamps))
        combined_data.index.name = 'date'
        
        # Initialize columns
        combined_data['actual'] = ''
        combined_data['day-ahead'] = ''
        combined_data['intraday'] = ''
        
        # Fill with new data
        for forecast_type, data in new_zone_data.items():
            if isinstance(data, pd.Series) and not data.empty:
                col_name = forecast_type.replace('_', '-')
                if col_name in combined_data.columns:
                    combined_data.loc[data.index, col_name] = data.values
    
    if os.path.exists(filepath):
        # Read existing data
        existing_df = pd.read_csv(filepath)
        existing_df['date'] = pd.to_datetime(existing_df['date'])
        existing_df.set_index('date', inplace=True)
        
        if not combined_data.empty:
            # Merge: remove any overlapping dates from existing data to avoid duplicates
            non_overlapping_existing = existing_df[~existing_df.index.isin(combined_data.index)]
            
            # Combine existing non-overlapping data with new data
            final_df = pd.concat([non_overlapping_existing, combined_data]).sort_index()
        else:
            final_df = existing_df
        
        # Save updated file
        final_df.reset_index(inplace=True)
        final_df.to_csv(filepath, index=False)
        
        new_records = len(combined_data) if not combined_data.empty else 0
        print(f"  Updated {filename}: added {new_records} new records (total: {len(final_df)})")
        
    elif not combined_data.empty:
        # No existing file, create new one
        combined_data.reset_index(inplace=True)
        combined_data.to_csv(filepath, index=False)
        print(f"  Created {filename}: {len(combined_data)} records")
    else:
        print(f"  No data to save for {zone_name}")

def incremental_update(forecast_types=['day_ahead', 'intraday', 'actual'], lookback_days=0):
    """
    Perform incremental update for all Italian solar zones.
    
    Args:
        forecast_types: List of forecast types to update
        lookback_days: Number of days to look back from last data (to catch any revisions)
                      Set to 0 to start exactly from the day after the last data date
    """
    print("Starting incremental Italian solar power data update...")
    print(f"Forecast types: {forecast_types}")
    print(f"Lookback days: {lookback_days}")
    print("="*60)
    
    # Use UTC timezone for API calls
    utc = pytz.UTC
    current_date = datetime.now(utc).replace(hour=0, minute=0, second=0, microsecond=0)
    
    for zone_name, zone_code in ITALY_ZONES.items():
        print(f"\n--- Processing {zone_name} ({zone_code}) ---")
        
        # Get the last date we have data for
        last_date = get_last_update_date(zone_name)
        
        if last_date is None:
            # No existing data, start from zone restriction date or 2015
            if zone_name in ZONE_START_DATES:
                start_date = ZONE_START_DATES[zone_name]
            else:
                start_date = datetime(2015, 1, 1, tzinfo=utc)
            print(f"  No existing data. Starting from {start_date.date()}")
        else:
            # Start from the day after the last date, minus lookback_days to catch revisions
            # If lookback_days=0, start exactly from the day after last date (no duplicates)
            if lookback_days == 0:
                start_date = datetime.combine(
                    (last_date + timedelta(days=1)).date(),
                    datetime.min.time(),
                    tzinfo=utc
                )
                print(f"  Starting from day after last data: {start_date.date()} (no overlap)")
            else:
                start_date = datetime.combine(
                    (last_date - timedelta(days=lookback_days - 1)).date(),
                    datetime.min.time(),
                    tzinfo=utc
                )
                print(f"  Starting from {start_date.date()} (last date - {lookback_days - 1} days for revisions)")
        
        # Convert to pandas Timestamp with UTC timezone
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(current_date)
        
        # Skip if we're already up to date (start date is after current date)
        if start_date >= current_date:
            print(f"  {zone_name} is already up to date")
            continue
        
        print(f"  Fetching data from {start_date.date()} to {end_date.date()}")
        
        # Get new data
        new_zone_data = update_zone_data(zone_name, zone_code, start_date, end_date, forecast_types)
        
        # Merge with existing data
        merge_with_existing_data(zone_name, new_zone_data)
    
    print("\n" + "="*60)
    print("INCREMENTAL UPDATE COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    # Run incremental update
    # IMPORTANT: lookback_days controls overlap behavior:
    # - lookback_days=0: No overlap, starts day after last data (prevents duplicates)
    # - lookback_days=3: Overlaps 3 days to catch data revisions (may update existing data)
    
    # Note: intraday forecasts are often not available for historical dates
    # Only use 'intraday' if you need very recent/current day data
    incremental_update(
        forecast_types=['day_ahead', 'actual'],  # Removed 'intraday' - causes 400 errors for historical data
        lookback_days=1  # 1 day lookback to catch any data revisions
    )
    
    print("\nIncremental update completed. Check data/energy/ for updated CSV files.")
