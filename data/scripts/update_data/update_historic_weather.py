#!/usr/bin/env python3
"""

THIS HAS NOT YET BEEN TESTED IF IT WORKS
Incremental Weather Data Update Script

This script updates existing hourly weather CSV files with new data since the last update.
It reads the last date from each weather file and fetches only new data from OpenMeteo API.

Features:
- Detects existing weather files automatically
- Finds the last datetime in each file
- Fetches only new hourly weather data since last update
- Merges new data with existing data
- Prevents duplicates and overwrites
- Uses same weather parameters as initial weather script

Requirements:
- requests library
- pandas library

Usage:
    python data/scripts/update_data/update_historic_weather.py

Output:
    Updates existing CSV files in data/weather/ directory
"""

import os
import sys
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings("ignore")

# Add the root directory to Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, root_dir)

# Import weather parameters and locations from initial script
try:
    from data.scripts.initial.initial_weather_history import SOLAR_WEATHER_PARAMS, ITALY_ZONE_LOCATIONS
except ImportError:
    # Fallback definitions if import fails
    SOLAR_WEATHER_PARAMS = [
        'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
        'temperature_2m', 'apparent_temperature', 'cloudcover', 'dewpoint_2m',
        'precipitation', 'windspeed_10m', 'windgusts_10m', 'winddirection_10m',
        'surface_pressure', 'relativehumidity_2m'
    ]
    
    ITALY_ZONE_LOCATIONS = {
        'IT-NORD': (45.4642, 9.1900),
        'IT-CNOR': (43.7696, 11.2558),
        'IT-CSUD': (41.9028, 12.4964),
        'IT-SUD': (40.8518, 14.2681),
        'IT-SICI': (37.5079, 15.0830),
        'IT-SARD': (39.2238, 9.1217),
        'IT-CALA': (38.9072, 16.5947)
    }

def get_weather_files(weather_dir):
    """
    Get list of existing hourly weather CSV files.
    
    Returns:
        dict: Mapping of zone names to file paths
    """
    weather_files = {}
    
    if not os.path.exists(weather_dir):
        print(f"Weather directory {weather_dir} not found.")
        return weather_files
    
    # Look for hourly weather files
    for filename in os.listdir(weather_dir):
        if filename.endswith('_weather_hourly.csv'):
            # Extract zone name from filename
            zone_name = filename.replace('_weather_hourly.csv', '').replace('_', '-').upper()
            if zone_name in ITALY_ZONE_LOCATIONS:
                weather_files[zone_name] = os.path.join(weather_dir, filename)
    
    return weather_files

def get_last_update_datetime(filepath):
    """
    Get the last datetime from an existing weather CSV file.
    
    Args:
        filepath: Path to the weather CSV file
        
    Returns:
        datetime: Last datetime in the file, or None if file doesn't exist/is empty
    """
    try:
        if not os.path.exists(filepath):
            return None
            
        # Read just the last few lines to find the last datetime
        df = pd.read_csv(filepath, usecols=['datetime'])
        
        if df.empty:
            return None
            
        # Convert to datetime and get the last one
        df['datetime'] = pd.to_datetime(df['datetime'])
        last_datetime = df['datetime'].max()
        
        return last_datetime
        
    except Exception as e:
        print(f"    Error reading last datetime from {filepath}: {e}")
        return None

def fetch_weather_data(lat, lon, start_date, end_date, params):
    """
    Fetch hourly weather data from OpenMeteo API for a date range.
    
    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        params: List of weather parameters to fetch
        
    Returns:
        pandas.DataFrame: Weather data with datetime, date, hour columns
    """
    api_params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': ','.join(params),
        'timezone': 'Europe/Rome'
    }
    
    try:
        print(f"    Fetching weather data from {start_date} to {end_date}...")
        response = requests.get('https://archive-api.open-meteo.com/v1/archive', params=api_params)
        response.raise_for_status()
        
        data = response.json()
        
        if 'hourly' not in data:
            print(f"    No hourly data found in response")
            return pd.DataFrame()
        
        # Process hourly data
        hourly_data = data['hourly']
        df = pd.DataFrame(hourly_data)
        
        # Convert datetime column and create separate date and hour columns
        df['datetime'] = pd.to_datetime(df['time'])
        df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')
        df['hour'] = df['datetime'].dt.hour
        df = df.drop('time', axis=1)
        
        # Reorder columns: datetime, date, hour, then weather parameters
        weather_cols = [col for col in df.columns if col not in ['datetime', 'date', 'hour']]
        df = df[['datetime', 'date', 'hour'] + weather_cols]
        
        print(f"    Successfully fetched {len(df)} hours of weather data")
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"    Error fetching weather data: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"    Error processing weather data: {e}")
        return pd.DataFrame()

def fetch_weather_data_chunked(lat, lon, start_date, end_date, params, chunk_days=30):
    """
    Fetch hourly weather data in smaller chunks to avoid API limits.
    
    Args:
        lat: Latitude
        lon: Longitude  
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        params: List of weather parameters to fetch
        chunk_days: Size of each chunk in days (default 30 for hourly data)
    
    Returns:
        pandas.DataFrame: Combined weather data for the entire date range
    """
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    all_data = []
    current_start = start_dt
    
    while current_start < end_dt:
        # Calculate chunk end date (30 days for hourly data)
        chunk_end = min(current_start + timedelta(days=chunk_days), end_dt)
        
        chunk_start_str = current_start.strftime('%Y-%m-%d')
        chunk_end_str = chunk_end.strftime('%Y-%m-%d')
        
        # Fetch data for this chunk
        chunk_data = fetch_weather_data(lat, lon, chunk_start_str, chunk_end_str, params)
        
        if not chunk_data.empty:
            all_data.append(chunk_data)
        
        # Move to next chunk
        current_start = chunk_end + timedelta(days=1)
        
        # Longer delay for hourly data to respect API rate limits
        time.sleep(5)
    
    # Combine all chunks
    if all_data:
        combined_df = pd.concat(all_data, axis=0, ignore_index=True)
        # Sort by datetime and remove duplicates
        combined_df = combined_df.sort_values('datetime')
        combined_df = combined_df.drop_duplicates(subset=['datetime'], keep='first')
        return combined_df
    else:
        return pd.DataFrame()

def merge_with_existing_data(new_data, existing_filepath):
    """
    Merge new weather data with existing CSV file.
    
    Args:
        new_data: DataFrame with new weather data
        existing_filepath: Path to existing CSV file
        
    Returns:
        pandas.DataFrame: Combined data with duplicates removed
    """
    if new_data.empty:
        print("    No new data to merge")
        return pd.DataFrame()
    
    # Read existing data if file exists
    if os.path.exists(existing_filepath):
        try:
            existing_data = pd.read_csv(existing_filepath)
            existing_data['datetime'] = pd.to_datetime(existing_data['datetime'])
            
            # Combine and remove duplicates
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            combined_data = combined_data.drop_duplicates(subset=['datetime'], keep='last')
            combined_data = combined_data.sort_values('datetime')
            
            print(f"    Merged {len(new_data)} new records with {len(existing_data)} existing records")
            print(f"    Total records after merge: {len(combined_data)}")
            
            return combined_data
            
        except Exception as e:
            print(f"    Error reading existing file: {e}")
            return new_data
    else:
        print("    No existing file found, using new data only")
        return new_data

def update_weather_data():
    """
    Main function to update weather data for all zones.
    """
    print("="*70)
    print("Hourly Weather Data Update Script")
    print("="*70)
    
    weather_dir = os.path.join(root_dir, 'data', 'weather')
    
    # Get existing weather files
    weather_files = get_weather_files(weather_dir)
    
    if not weather_files:
        print("No existing hourly weather files found.")
        print("Please run the initial weather history script first:")
        print("python data/scripts/initial/initial_weather_history.py")
        return
    
    print(f"Found {len(weather_files)} existing weather files to update:")
    for zone, filepath in weather_files.items():
        print(f"  - {zone}: {os.path.basename(filepath)}")
    print()
    
    # Current date minus 5 days (OpenMeteo Archive API has ~5 day delay)
    end_date = datetime.now() - timedelta(days=5)
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    updated_files = 0
    
    for zone_name, filepath in weather_files.items():
        print(f"--- Updating {zone_name} ---")
        
        # Get last update datetime
        last_datetime = get_last_update_datetime(filepath)
        
        if last_datetime is None:
            print(f"  Could not determine last update time for {zone_name}")
            continue
            
        # Calculate start date for new data (next hour after last update)
        start_datetime = last_datetime + timedelta(hours=1)
        start_date_str = start_datetime.strftime('%Y-%m-%d')
        
        print(f"  Last update: {last_datetime}")
        print(f"  Fetching new data from: {start_date_str} to {end_date_str}")
        
        # Check if we need to update
        if start_datetime.date() > end_date.date():
            print(f"  ✅ Weather data is already up to date!")
            continue
        
        # Get coordinates for this zone
        if zone_name not in ITALY_ZONE_LOCATIONS:
            print(f"  ❌ Unknown zone: {zone_name}")
            continue
            
        lat, lon = ITALY_ZONE_LOCATIONS[zone_name]
        
        try:
            # Fetch new weather data
            new_weather_data = fetch_weather_data_chunked(
                lat, lon, start_date_str, end_date_str, SOLAR_WEATHER_PARAMS
            )
            
            if new_weather_data.empty:
                print(f"  No new weather data available for {zone_name}")
                continue
            
            # Merge with existing data
            combined_data = merge_with_existing_data(new_weather_data, filepath)
            
            if combined_data.empty:
                print(f"  No data to save for {zone_name}")
                continue
            
            # Save updated data
            combined_data.to_csv(filepath, index=False)
            
            print(f"  ✅ Updated {os.path.basename(filepath)}")
            print(f"  New date range: {combined_data['date'].min()} to {combined_data['date'].max()}")
            print(f"  Total records: {len(combined_data)} hours")
            
            updated_files += 1
            
        except Exception as e:
            print(f"  ❌ Error updating {zone_name}: {e}")
            continue
        
        print()
    
    print("="*70)
    print(f"Weather update completed! Updated {updated_files} files.")
    print("="*70)

if __name__ == "__main__":
    update_weather_data()
