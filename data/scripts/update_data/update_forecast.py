#!/usr/bin/env python3
"""
Weather Forecast Update Script

This script fetches 2-week hourly weather forecasts for all Italian bidding zones
using the same weather parameters as the historical weather data.

Features:
- Fetches 14-day hourly weather forecasts from OpenMeteo API
- Uses same weather parameters as historical data for consistency
- Covers all 7 Italian bidding zones
- Saves forecast data to data/weather/forecast/ directory
- Overwrites existing forecast files (forecast data is always fresh)

Requirements:
- requests library
- pandas library

Usage:
    python data/scripts/update_data/update_forecast.py

Output:
    CSV files in data/weather/forecast/ directory with format: 
    datetime,date,hour,shortwave_radiation,direct_radiation,...
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
        'IT-NORD': (45.4642, 9.1900),    # Milan area (Northern Italy)
        'IT-CNOR': (43.7696, 11.2558),  # Florence area (Central-North Italy)
        'IT-CSUD': (41.9028, 12.4964),  # Rome area (Central-South Italy)
        'IT-SUD': (40.8518, 14.2681),   # Naples area (South Italy)
        'IT-SICI': (37.5079, 15.0830),  # Catania area (Sicily)
        'IT-SARD': (39.2238, 9.1217),   # Cagliari area (Sardinia)
        'IT-CALA': (38.9072, 16.5947)   # Cosenza area (Calabria)
    }

def fetch_weather_forecast(lat, lon, start_date, end_date, params):
    """
    Fetch hourly weather forecast from OpenMeteo API for a date range.
    
    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        params: List of weather parameters to fetch
        
    Returns:
        pandas.DataFrame: Weather forecast data with datetime, date, hour columns
    """
    # Use the forecast API endpoint
    api_params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': ','.join(params),
        'timezone': 'Europe/Rome'
    }
    
    try:
        print(f"    Fetching weather forecast from {start_date} to {end_date}...")
        response = requests.get('https://api.open-meteo.com/v1/forecast', params=api_params)
        response.raise_for_status()
        
        data = response.json()
        
        if 'hourly' not in data:
            print(f"    No hourly forecast data found in response")
            return pd.DataFrame()
        
        # Process hourly forecast data
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
        
        print(f"    Successfully fetched {len(df)} hours of weather forecast data")
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"    Error fetching weather forecast: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"    Error processing weather forecast: {e}")
        return pd.DataFrame()

def save_forecast_data(zone_name, forecast_data, forecast_dir):
    """
    Save weather forecast data to CSV file in the forecast directory.
    
    Args:
        zone_name: Name of the zone
        forecast_data: DataFrame with forecast data
        forecast_dir: Directory to save forecast data
    """
    if forecast_data.empty:
        print(f"  No forecast data to save for {zone_name}")
        return
    
    # Ensure forecast directory exists
    os.makedirs(forecast_dir, exist_ok=True)
    
    # Create filename for forecast data
    filename = f'{zone_name.lower().replace("-", "_")}_weather_forecast.csv'
    filepath = os.path.join(forecast_dir, filename)
    
    # Save to CSV (overwrite if exists - forecast data is always fresh)
    forecast_data.to_csv(filepath, index=False)
    
    print(f"  ✅ Saved weather forecast to {filename}")
    print(f"  Date range: {forecast_data['date'].min()} to {forecast_data['date'].max()}")
    print(f"  Total records: {len(forecast_data)} hours")
    print(f"  Weather parameters: {list(forecast_data.columns[3:])}")  # Skip datetime, date, hour columns

def update_weather_forecasts():
    """
    Main function to fetch 2-week weather forecasts for all Italian zones.
    """
    print("="*70)
    print("2-Week Weather Forecast Update Script")
    print("="*70)
    print("• Forecast data source: OpenMeteo Forecast API (free)")
    print("• Forecast period: 14 days ahead")
    print("• Temporal resolution: Hourly data")
    print("• Spatial coverage: 7 Italian bidding zones")
    print("• Weather parameters: Same as historical data for consistency")
    print("  - Solar Radiation: shortwave, direct, diffuse radiation (W/m²)")
    print("  - Temperature: temperature_2m, apparent_temperature (°C)")
    print("  - Cloud cover: cloudcover (%)")
    print("  - Atmospheric: dewpoint_2m, surface_pressure, relativehumidity_2m")
    print("  - Wind: windspeed_10m, windgusts_10m, winddirection_10m")
    print("  - Precipitation: precipitation (mm)")
    print("="*70)
    
    # Calculate forecast date range (today + 14 days)
    start_date = datetime.now().date()
    end_date = start_date + timedelta(days=14)
    
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    print(f"Fetching weather forecasts for period: {start_date_str} to {end_date_str}")
    print(f"Expected total data points: ~{14 * 24 * len(ITALY_ZONE_LOCATIONS)} hours")
    print()
    
    # Setup forecast directory
    forecast_dir = os.path.join(root_dir, 'data', 'raw', 'weather', 'forecast')
    
    successful_updates = 0
    
    for zone_name, (lat, lon) in ITALY_ZONE_LOCATIONS.items():
        print(f"--- Processing {zone_name} ---")
        print(f"Location: {lat:.4f}°N, {lon:.4f}°E")
        
        try:
            # Fetch forecast data for this zone
            forecast_data = fetch_weather_forecast(
                lat, lon, start_date_str, end_date_str, SOLAR_WEATHER_PARAMS
            )
            
            if forecast_data.empty:
                print(f"  ❌ No forecast data retrieved for {zone_name}")
                continue
            
            # Save forecast data
            save_forecast_data(zone_name, forecast_data, forecast_dir)
            successful_updates += 1
            
        except Exception as e:
            print(f"  ❌ Error processing {zone_name}: {e}")
            continue
        
        # Small delay to be respectful to the API
        time.sleep(1)
        print()
    
    print("="*70)
    print(f"Weather forecast update completed!")
    print(f"Successfully updated {successful_updates} out of {len(ITALY_ZONE_LOCATIONS)} zones.")
    print(f"Forecast files saved to: {forecast_dir}")
    print("="*70)
    
    # Display summary of created files
    if successful_updates > 0:
        print("\nCreated forecast files:")
        try:
            for filename in sorted(os.listdir(forecast_dir)):
                if filename.endswith('_weather_forecast.csv'):
                    filepath = os.path.join(forecast_dir, filename)
                    file_size = os.path.getsize(filepath)
                    print(f"  - {filename} ({file_size:,} bytes)")
        except:
            pass

if __name__ == "__main__":
    update_weather_forecasts()
