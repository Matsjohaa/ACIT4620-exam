#!/usr/bin/env python3
"""
OpenMeteo Historical Weather Data Retrieval Script for Italian Solar Zones

This script retrieves historical weather data for Italian bidding zones from OpenMeteo API.
It fetches weather par    if weather_data.em    filepath = os.path.join(data_dir, filename)
    weather_data.to_csv(filepath, index=False)
    
    print(f"  Saved hourly weather data to {filename}")
    print(f"  Date range: {weather_data['date'].min()} to {weather_data['date'].max()}")
    print(f"  Total records: {len(weather_data)} hours")
    print(f"  Weather parameters: {list(weather_data.columns[3:])}")  # Skip datetime, date, hour columns      print(f"  No weather data to save for {zone_name}")
        return
    
    # Data already has datetime, date, and hour columns from processing
    # No need to reset index or convert
    
    # Create filename 
    filename = f'{zone_name.lower().replace("-", "_")}_weather_hourly.csv'nt to solar energy production from 2015 to present.

Weather parameters relevant to solar energy:
- Solar radiation (shortwave_radiation_sum, direct_radiation_sum, diffuse_radiation_sum)
- Direct Normal Irradiance (direct_normal_irradiance_sum)
- Global Tilted Irradiance (global_tilted_irradiance_sum) - for tilted panels
- Temperature (temperature_2m_max, temperature_2m_min, temperature_2m_mean)
- Cloud cover (cloudcover_mean, cloud_cover_low_mean, cloud_cover_mid_mean, cloud_cover_high_mean)
- Sunshine duration (sunshine_duration)
- Relative humidity (relativehumidity_2m_mean)
- Wind speed & direction (windspeed_10m_max, windspeed_10m_mean, winddirection_10m_dominant, windgusts_10m_max)
- Precipitation & snow (precipitation_sum, snowfall_sum, snow_depth_mean)
- Atmospheric conditions (surface_pressure_mean, visibility_mean)
- Solar-specific metrics (uv_index_max, daylight_duration)
- Weather classification (weather_code_dominant)

Requirements:
- requests library
- pandas library

Usage:
    python data/scripts/initial/initial_weather_history.py

Output:
    CSV files in data/weather/ directory with format: date,param1,param2,param3,...
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

# Central locations for each Italian bidding zone (lat, lon)
ITALY_ZONE_LOCATIONS = {
    'IT-NORD': (45.4642, 9.1900),    # Milan area (Northern Italy)
    'IT-CNOR': (43.7696, 11.2558),  # Florence area (Central-North Italy)
    'IT-CSUD': (41.9028, 12.4964),  # Rome area (Central-South Italy)
    'IT-SUD': (40.8518, 14.2681),   # Naples area (South Italy)
    'IT-SICI': (37.5079, 15.0830),  # Catania area (Sicily)
    'IT-SARD': (39.2238, 9.1217),   # Cagliari area (Sardinia)
    'IT-CALA': (38.9072, 16.5947)   # Cosenza area (Calabria)
}

# Weather parameters relevant to solar energy production (hourly data, confirmed working with Archive API)
SOLAR_WEATHER_PARAMS = [
    # Solar radiation and light (key for solar power analysis)
    'shortwave_radiation',         # Hourly shortwave solar radiation (W/m²)
    'direct_radiation',            # Hourly direct solar radiation (W/m²)
    'diffuse_radiation',           # Hourly diffuse solar radiation (W/m²)
    
    # Temperature parameters (affect solar panel efficiency)
    'temperature_2m',              # Hourly temperature at 2m (°C)
    'apparent_temperature',        # Hourly apparent temperature (°C)
    
    # Cloud cover and atmospheric conditions (directly impact solar irradiance)
    'cloudcover',                  # Hourly total cloud cover (%)
    'dewpoint_2m',                 # Hourly dew point temperature (°C)
    
    # Precipitation (affects panel cleanliness and atmospheric transmission)
    'precipitation',               # Hourly precipitation (mm)
    
    # Wind parameters (affect panel cooling and dust removal)
    'windspeed_10m',               # Hourly wind speed at 10m (km/h)
    'windgusts_10m',               # Hourly wind gusts at 10m (km/h)
    'winddirection_10m',           # Hourly wind direction at 10m (°)
    
    # Additional atmospheric parameters
    'surface_pressure',            # Hourly surface pressure (hPa)
    'relativehumidity_2m',         # Hourly relative humidity (%)
]

# Note: Parameters like direct_radiation_sum, diffuse_radiation_sum, 
# cloud_cover_low/mid/high_mean, visibility_mean, surface_pressure_mean
# are not available in the OpenMeteo Archive API despite being in current API

# Date restrictions for newer zones (same as power data)
ZONE_START_DATES = {
    'IT-SUD': '2021-01-01',
    'IT-CALA': '2021-01-01'
}

def get_zone_start_date(zone_name):
    """Get the earliest date for data availability for a zone."""
    if zone_name in ZONE_START_DATES:
        return ZONE_START_DATES[zone_name]
    return '2015-01-01'  # Default start date

def fetch_weather_data(lat, lon, start_date, end_date, params):
    """
    Fetch historical weather data from OpenMeteo API.
    
    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        params: List of weather parameters to fetch
    
    Returns:
        pandas.DataFrame: Weather data
    """
    # OpenMeteo Historical Weather API endpoint
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # Parameters for the API request
    api_params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': ','.join(params),  # Changed from 'daily' to 'hourly'
        'timezone': 'Europe/Rome'
    }
    
    try:
        print(f"    Fetching weather data from {start_date} to {end_date}...")
        response = requests.get(url, params=api_params)
        response.raise_for_status()
        
        data = response.json()
        
        if 'hourly' not in data:
            print(f"    No hourly data found in response")
            return pd.DataFrame()
        
        # Convert to DataFrame
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
    Using 30-day chunks for hourly data (~720 data points per chunk).
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

def save_weather_data(zone_name, weather_data, data_dir=None):
    """
    Save weather data to CSV file.
    
    Args:
        zone_name: Name of the zone
        weather_data: DataFrame with weather data
        data_dir: Directory to save data (relative to root)
    """
    if data_dir is None:
        data_dir = os.path.join(root_dir, 'data', 'raw', 'weather', 'historic')
    
    # Ensure output directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    if weather_data.empty:
        print(f"  No weather data to save for {zone_name}")
        return
    
    # Data already has datetime, date, and hour columns from processing
    # No need to reset index or convert dates
    
    # Create filename for hourly data
    filename = f'{zone_name.lower().replace("-", "_")}_weather_hourly.csv'
    filepath = os.path.join(data_dir, filename)
    weather_data.to_csv(filepath, index=False)
    
    print(f"  Saved hourly weather data to {filename}")
    print(f"  Date range: {weather_data['date'].min()} to {weather_data['date'].max()}")
    print(f"  Total records: {len(weather_data)} hours")
    print(f"  Weather parameters: {list(weather_data.columns[3:])}")  # Skip datetime, date, hour columns

def retrieve_italy_weather_data():
    """Main function to retrieve historical weather data for all Italian zones."""
    print("Starting Italian Weather Data Retrieval...")
    print("Weather parameters for solar energy analysis:")
    for param in SOLAR_WEATHER_PARAMS:
        print(f"  - {param}")
    print("=" * 70)
    
    # End date is yesterday (weather data usually has 1-day lag)
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    for zone_name, (lat, lon) in ITALY_ZONE_LOCATIONS.items():
        try:
            # Get appropriate start date for this zone
            start_date = get_zone_start_date(zone_name)
            
            print(f"\n--- Processing {zone_name} ---")
            print(f"Location: {lat:.4f}°N, {lon:.4f}°E")
            print(f"Date range: {start_date} to {end_date}")
            
            # Fetch weather data in chunks
            weather_data = fetch_weather_data_chunked(
                lat, lon, start_date, end_date, SOLAR_WEATHER_PARAMS
            )
            
            # Save the data
            save_weather_data(zone_name, weather_data)
            
        except Exception as e:
            print(f"Error processing {zone_name}: {e}")
            continue
    
    print("\n" + "=" * 70)
    print("Weather data retrieval completed!")
    print(f"Check data/weather/ directory for CSV files")

def print_api_info():
    """Print information about OpenMeteo API and data availability."""
    print("\n" + "=" * 70)
    print("OpenMeteo Historical Weather API Information:")
    print("=" * 70)
    print("• API: Open-Meteo Archive API (free, no registration required)")
    print("• Data source: ERA5 reanalysis from ECMWF")
    print("• Temporal resolution: Hourly data")
    print("• Spatial resolution: ~11 km")
    print("• Historical coverage: 1940 to present (with 5-day delay)")
    print("• Data relevant to solar energy production:")
    print("  - Solar Radiation: Direct, diffuse, DNI, and tilted irradiance")
    print("  - Cloud Cover: Total and by altitude (low/mid/high clouds)")
    print("  - Temperature: Max/min/mean affecting panel efficiency")
    print("  - Sunshine & Daylight: Duration of direct sun and total daylight")
    print("  - Atmospheric: Pressure, visibility, humidity affecting irradiance")
    print("  - Wind: Speed, direction, gusts for panel cooling and dust patterns")
    print("  - Snow/Precipitation: Blocking effects on panel performance")
    print("  - UV Index: Additional solar radiation indicator")
    print("  - Weather Codes: Categorical weather conditions")
    print("• Rate limits: Reasonable for non-commercial use")
    print("=" * 70)

if __name__ == "__main__":
    print_api_info()
    
    # Ask for confirmation since this fetches a lot of data
    response = input("\nThis will fetch 10+ years of HOURLY weather data for 7 zones (~600,000+ data points).\nContinue? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        retrieve_italy_weather_data()
        
        print("\n" + "=" * 70)
        print("WEATHER DATA SUMMARY:")
        print("=" * 70)
        
        # Show summary of created files
        weather_dir = os.path.join(root_dir, 'data', 'weather')
        if os.path.exists(weather_dir):
            weather_files = [f for f in os.listdir(weather_dir) if f.endswith('_weather.csv')]
            print(f"Created {len(weather_files)} weather data files:")
            for file in sorted(weather_files):
                filepath = os.path.join(weather_dir, file)
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath)
                    print(f"  {file}: {len(df)} records ({df['date'].min()} to {df['date'].max()})")
        
        print("\nWeather data can now be used for:")
        print("• Solar energy production correlation analysis") 
        print("• Weather impact on solar generation forecasting")
        print("• Seasonal and climate pattern analysis")
        print("• Machine learning models for solar prediction")
        
    else:
        print("Weather data retrieval cancelled.")
