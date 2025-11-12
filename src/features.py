"""
Feature engineering module for solar forecasting.

This module creates features for the CNN-LSTM model:
- Temporal features (hour, day, month, season)
- Solar geometry features (sun angle, day length)
- Lag features (historical production)
- Weather derivatives
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def add_temporal_features(df):
    """
    Add temporal features based on date/time.
    
    Args:
        df: DataFrame with 'date' column
    
    Returns:
        DataFrame with added temporal features
    """
    df = df.copy()
    
    # Extract basic time components
    df['hour'] = df['date'].dt.hour
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    
    # Cyclical encoding for hour (important for solar patterns)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Cyclical encoding for day of year (seasonal patterns)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Cyclical encoding for month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Season (meteorological seasons)
    def get_season(month):
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Autumn
    
    df['season'] = df['month'].apply(get_season)
    
    # Is weekend
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    return df


def add_solar_geometry_features(df, latitude, longitude):
    """
    Add solar geometry features (sun angle, day length).
    
    Args:
        df: DataFrame with 'date' column
        latitude: Latitude of the location
        longitude: Longitude of the location
    
    Returns:
        DataFrame with solar geometry features
    """
    df = df.copy()
    
    # Day length calculation (simplified)
    # Based on latitude and day of year
    day_of_year = df['day_of_year']
    lat_rad = np.radians(latitude)
    
    # Declination angle (simplified)
    declination = 23.45 * np.sin(np.radians(360 / 365 * (day_of_year + 284)))
    dec_rad = np.radians(declination)
    
    # Hour angle at sunrise/sunset
    cos_hour_angle = -np.tan(lat_rad) * np.tan(dec_rad)
    cos_hour_angle = np.clip(cos_hour_angle, -1, 1)  # Clip for polar regions
    hour_angle = np.degrees(np.arccos(cos_hour_angle))
    
    # Day length in hours
    df['day_length'] = 2 * hour_angle / 15  # Convert to hours
    
    # Solar altitude angle at noon (simplified)
    df['solar_altitude_max'] = 90 - latitude + declination
    
    # Estimated solar zenith angle for the current hour
    # Simplified calculation based on hour of day
    hour_from_noon = np.abs(df['hour'] - 12)
    df['hour_angle_current'] = 15 * hour_from_noon  # Degrees from solar noon
    
    # Approximate zenith angle
    zenith = np.degrees(np.arccos(
        np.sin(lat_rad) * np.sin(dec_rad) + 
        np.cos(lat_rad) * np.cos(dec_rad) * np.cos(np.radians(df['hour_angle_current']))
    ))
    df['solar_zenith'] = zenith
    df['solar_altitude'] = 90 - zenith
    
    # Is daytime (sun above horizon)
    df['is_daytime'] = (df['solar_altitude'] > 0).astype(int)
    
    return df


def add_lag_features(df, target_col='capacity_factor', lag_hours=[1, 3, 6, 12, 24, 48, 168]):
    """
    Add lag features for historical production.
    
    Args:
        df: DataFrame with target column
        target_col: Name of target column
        lag_hours: List of lag hours to create
    
    Returns:
        DataFrame with lag features
    """
    df = df.copy()
    
    # Sort by date to ensure correct lagging
    df = df.sort_values('date').reset_index(drop=True)
    
    # Create lag features
    for lag in lag_hours:
        df[f'{target_col}_lag_{lag}h'] = df[target_col].shift(lag)
    
    # Rolling statistics
    for window in [24, 48, 168]:  # 1 day, 2 days, 1 week
        df[f'{target_col}_rolling_mean_{window}h'] = df[target_col].shift(1).rolling(window=window).mean()
        df[f'{target_col}_rolling_std_{window}h'] = df[target_col].shift(1).rolling(window=window).std()
        df[f'{target_col}_rolling_max_{window}h'] = df[target_col].shift(1).rolling(window=window).max()
        df[f'{target_col}_rolling_min_{window}h'] = df[target_col].shift(1).rolling(window=window).min()
    
    return df


def add_forecast_error_features(df):
    """
    Add features based on forecast errors.
    
    Args:
        df: DataFrame with 'actual', 'day-ahead', and 'capacity_factor' columns
    
    Returns:
        DataFrame with forecast error features
    """
    df = df.copy()
    
    # Day-ahead forecast error
    if 'day-ahead' in df.columns and 'actual' in df.columns:
        df['dayahead_error'] = df['actual'] - df['day-ahead']
        df['dayahead_abs_error'] = np.abs(df['dayahead_error'])
        df['dayahead_pct_error'] = df['dayahead_error'] / (df['actual'] + 1e-6)  # Avoid division by zero
        
        # Rolling forecast error statistics
        for window in [24, 168]:  # 1 day, 1 week
            df[f'dayahead_error_rolling_mean_{window}h'] = df['dayahead_error'].shift(1).rolling(window=window).mean()
            df[f'dayahead_abs_error_rolling_mean_{window}h'] = df['dayahead_abs_error'].shift(1).rolling(window=window).mean()
    
    return df


def add_weather_derivatives(df):
    """
    Add derived weather features.
    
    Args:
        df: DataFrame with weather columns
    
    Returns:
        DataFrame with weather derivative features
    """
    df = df.copy()
    
    weather_cols = [
        'temperature_2m', 'shortwave_radiation', 'direct_radiation', 
        'diffuse_radiation', 'cloud_cover', 'wind_speed_10m'
    ]
    
    # Add rate of change for key weather variables
    for col in weather_cols:
        if col in df.columns:
            # Hourly change
            df[f'{col}_diff_1h'] = df[col].diff(1)
            
            # 3-hour and 6-hour change
            df[f'{col}_diff_3h'] = df[col].diff(3)
            df[f'{col}_diff_6h'] = df[col].diff(6)
            
            # Rolling statistics
            df[f'{col}_rolling_mean_24h'] = df[col].rolling(window=24).mean()
            df[f'{col}_rolling_std_24h'] = df[col].rolling(window=24).std()
    
    # Clear sky index (if we have radiation data)
    if 'shortwave_radiation' in df.columns and 'solar_altitude' in df.columns:
        # Simplified clear sky radiation estimate
        # This is a very rough approximation
        clear_sky_radiation = 1000 * np.sin(np.radians(df['solar_altitude'].clip(0, 90)))
        df['clear_sky_index'] = df['shortwave_radiation'] / (clear_sky_radiation + 1e-6)
        df['clear_sky_index'] = df['clear_sky_index'].clip(0, 1)
    
    return df


def engineer_features(df, zone_info=None):
    """
    Apply all feature engineering steps.
    
    Args:
        df: DataFrame with preprocessed data
        zone_info: Dictionary with zone information (latitude, longitude)
    
    Returns:
        DataFrame with all engineered features
    """
    print("Engineering features...")
    
    # Add temporal features
    print("  Adding temporal features...")
    df = add_temporal_features(df)
    
    # Add solar geometry features (if zone info provided)
    if zone_info and 'latitude' in zone_info and 'longitude' in zone_info:
        print("  Adding solar geometry features...")
        df = add_solar_geometry_features(df, zone_info['latitude'], zone_info['longitude'])
    
    # Add lag features
    print("  Adding lag features...")
    df = add_lag_features(df, target_col='capacity_factor')
    
    # Add forecast error features
    print("  Adding forecast error features...")
    df = add_forecast_error_features(df)
    
    # Add weather derivatives
    print("  Adding weather derivatives...")
    df = add_weather_derivatives(df)
    
    print("  ✓ Feature engineering complete!")
    
    return df


# Italian zone coordinates (approximate center of each zone)
ZONE_COORDINATES = {
    'IT-NORD': {'latitude': 45.5, 'longitude': 9.2},      # Northern Italy (Milan area)
    'IT-CNOR': {'latitude': 43.8, 'longitude': 11.2},     # Central North (Florence area)
    'IT-CSUD': {'latitude': 41.9, 'longitude': 12.5},     # Central South (Rome area)
    'IT-SUD': {'latitude': 40.8, 'longitude': 14.3},      # South (Naples area)
    'IT-SICI': {'latitude': 37.5, 'longitude': 14.0},     # Sicily (Catania area)
    'IT-SARD': {'latitude': 40.0, 'longitude': 9.0},      # Sardinia (Cagliari area)
    'IT-CALA': {'latitude': 38.9, 'longitude': 16.6}      # Calabria (Catanzaro area)
}


if __name__ == "__main__":
    print("Feature engineering module")
    print("Import this module to use feature engineering functions")
