"""
Data loading utilities for solar forecasting model.

This module handles loading and pdef prepare_sequences(df, 
                     sequence_length=168,  # 7 days input
                     forecast_horizon=336):  # 14 days outputaring train/test data for the CNN-LSTM model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional


# Define Italian zones
ZONES = ['IT-NORD', 'IT-CNOR', 'IT-CSUD', 'IT-SUD', 'IT-SICI', 'IT-SARD', 'IT-CALA']

# Weather feature columns
WEATHER_FEATURES = [
    'shortwave_radiation',
    'direct_radiation',
    'diffuse_radiation',
    'temperature_2m',
    'apparent_temperature',
    'cloudcover',
    'dewpoint_2m',
    'precipitation',
    'windspeed_10m',
    'windgusts_10m',
    'winddirection_10m',
    'surface_pressure',
    'relativehumidity_2m'
]

# Engineered features to help model distinguish weather conditions
ENGINEERED_FEATURES = [
    'solar_potential',      # radiation * (1 - cloudcover/100)
    'clear_sky_factor'      # 1 - cloudcover/100
]


def load_zone_data(zone: str, split: str = 'train') -> pd.DataFrame:
    """
    Load processed data for a specific zone.
    
    Args:
        zone: Zone name (e.g., 'IT-NORD')
        split: 'train' or 'test'
    
    Returns:
        DataFrame with zone data
    """
    zone_lower = zone.lower().replace('-', '_')
    filepath = Path(f'data/processed/{split}/{zone_lower}.csv')
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Add engineered features to help model learn weather patterns
    df['solar_potential'] = df['shortwave_radiation'] * (1 - df['cloudcover'] / 100)
    df['clear_sky_factor'] = 1 - df['cloudcover'] / 100
    
    print(f"Loaded {zone} {split}: {len(df)} records ({df['date'].min()} to {df['date'].max()})")
    
    return df


def load_all_zones(split: str = 'train') -> dict:
    """
    Load data for all zones.
    
    Args:
        split: 'train' or 'test'
    
    Returns:
        Dictionary mapping zone names to DataFrames
    """
    data = {}
    for zone in ZONES:
        try:
            data[zone] = load_zone_data(zone, split)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue
    
    return data


def prepare_sequences(df: pd.DataFrame, 
                     sequence_length: int = 168,  # 7 days
                     forecast_horizon: int = 336,  # 14 days
                     features: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for training the CNN-LSTM model.
    
    Args:
        df: DataFrame with processed data
        sequence_length: Number of hours to look back (default: 168 = 7 days)
        forecast_horizon: Number of hours to predict (default: 336 = 14 days)
        features: List of feature columns to use
    
    Returns:
        X: Input sequences (n_samples, sequence_length, n_features)
        y: Target sequences (n_samples, forecast_horizon)
    """
    if features is None:
        # Use ONLY weather + engineered features (NO hour!)
        # Force model to learn from weather conditions, not time patterns
        features = WEATHER_FEATURES + ENGINEERED_FEATURES
    
    # Filter available features
    available_features = [f for f in features if f in df.columns]
    
    if 'capacity_factor' not in df.columns:
        raise ValueError("capacity_factor column not found in data")
    
    # Extract features and target
    # X: Weather features + hour (no actual production data!)
    # y: Target capacity factor (what we want to predict)
    X_data = df[available_features].values
    y_data = df['capacity_factor'].values
    
    X_sequences = []
    y_sequences = []
    
    # Create sliding windows
    for i in range(len(df) - sequence_length - forecast_horizon + 1):
        X_seq = X_data[i:i + sequence_length]
        y_seq = y_data[i + sequence_length:i + sequence_length + forecast_horizon]
        
        # Check for NaN values
        if not np.isnan(X_seq).any() and not np.isnan(y_seq).any():
            X_sequences.append(X_seq)
            y_sequences.append(y_seq)
    
    X = np.array(X_sequences)
    y = np.array(y_sequences)
    
    print(f"Created {len(X)} sequences:")
    print(f"  Input shape: {X.shape} (samples, timesteps={sequence_length}, features={len(available_features)})")
    print(f"  Target shape: {y.shape} (samples, forecast_horizon={forecast_horizon})")
    
    return X, y


def prepare_test_sequences(df: pd.DataFrame,
                           features: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare test data for prediction.
    
    For the test set, we have a 14-day forecast period (Oct 27 - Nov 10).
    We'll use the entire test period as one sequence.
    
    Args:
        df: Test DataFrame with weather forecast data
        features: List of feature columns to use
    
    Returns:
        X: Input features (1, n_timesteps, n_features)
        y: Actual capacity factors for validation (1, n_timesteps)
    """
    if features is None:
        # For test data, we only have weather forecast + capacity_factor
        features = WEATHER_FEATURES + ['hour']
    
    # Filter available features
    available_features = [f for f in features if f in df.columns]
    
    X = df[available_features].values
    y = df['capacity_factor'].values if 'capacity_factor' in df.columns else None
    
    # Reshape to (1, timesteps, features) for single prediction
    X = X.reshape(1, X.shape[0], X.shape[1])
    if y is not None:
        y = y.reshape(1, -1)
    
    print(f"Test sequence prepared:")
    print(f"  Input shape: {X.shape} (1 sample, timesteps={df.shape[0]}, features={len(available_features)})")
    if y is not None:
        print(f"  Target shape: {y.shape} (1 sample, timesteps={df.shape[0]})")
    
    return X, y


def normalize_data(X_train: np.ndarray, 
                   X_test: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Normalize features using training set statistics.
    
    Args:
        X_train: Training data
        X_test: Test data (optional)
    
    Returns:
        X_train_normalized, X_test_normalized (or None), normalization_params
    """
    # Calculate mean and std for each feature across all samples and timesteps
    mean = X_train.mean(axis=(0, 1))
    std = X_train.std(axis=(0, 1))
    
    # Avoid division by zero
    std[std == 0] = 1.0
    
    # Normalize
    X_train_norm = (X_train - mean) / std
    
    X_test_norm = None
    if X_test is not None:
        X_test_norm = (X_test - mean) / std
    
    normalization_params = {
        'mean': mean,
        'std': std
    }
    
    print("Data normalized:")
    print(f"  Train shape: {X_train_norm.shape}")
    if X_test_norm is not None:
        print(f"  Test shape: {X_test_norm.shape}")
    
    return X_train_norm, X_test_norm, normalization_params


def get_data_summary(data_dict: dict) -> pd.DataFrame:
    """
    Get summary statistics for loaded data.
    
    Args:
        data_dict: Dictionary of zone DataFrames
    
    Returns:
        Summary DataFrame
    """
    summary = []
    for zone, df in data_dict.items():
        summary.append({
            'Zone': zone,
            'Records': len(df),
            'Start': df['date'].min(),
            'End': df['date'].max(),
            'Capacity Factor Mean': df['capacity_factor'].mean(),
            'Capacity Factor Std': df['capacity_factor'].std(),
            'Missing Values': df.isnull().sum().sum()
        })
    
    return pd.DataFrame(summary)


if __name__ == "__main__":
    print("="*70)
    print("DATA LOADER TEST")
    print("="*70)
    
    # Test loading training data
    print("\n1. Loading training data...")
    train_data = load_all_zones('train')
    print(f"\nLoaded {len(train_data)} zones")
    
    print("\n2. Training data summary:")
    summary = get_data_summary(train_data)
    print(summary.to_string())
    
    # Test loading test data
    print("\n3. Loading test data...")
    test_data = load_all_zones('test')
    print(f"\nLoaded {len(test_data)} zones")
    
    print("\n4. Test data summary:")
    summary_test = get_data_summary(test_data)
    print(summary_test.to_string())
    
    # Test sequence creation for one zone
    print("\n5. Creating training sequences for IT-NORD...")
    zone_df = train_data['IT-NORD']
    X, y = prepare_sequences(zone_df, sequence_length=168, forecast_horizon=336)
    
    print("\n6. Creating test sequences for IT-NORD...")
    test_df = test_data['IT-NORD']
    X_test, y_test = prepare_test_sequences(test_df)
    
    print("\n" + "="*70)
    print("✓ Data loader test complete!")
    print("="*70)
