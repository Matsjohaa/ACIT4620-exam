"""
Data loading utilities for the solar forecasting models.

This module:
- loads processed train/test data for Italian bidding zones,
- adds engineered weather features,
- prepares sliding-window sequences for both single-input and encoder–decoder models,
- supports residual targets: capacity_factor - day-ahead capacity_factor.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict


# Define Italian zones
ZONES: List[str] = [
    "IT-NORD",
    "IT-CNOR",
    "IT-CSUD",
    "IT-SUD",
    "IT-SICI",
    "IT-SARD",
    "IT-CALA",
]

# Weather feature columns
WEATHER_FEATURES: List[str] = [
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "temperature_2m",
    "apparent_temperature",
    "cloudcover",
    "dewpoint_2m",
    "precipitation",
    "windspeed_10m",
    "windgusts_10m",
    "winddirection_10m",
    "surface_pressure",
    "relativehumidity_2m",
]

# Engineered features to help model distinguish weather conditions
ENGINEERED_FEATURES: List[str] = [
    "solar_potential",   # radiation * (1 - cloudcover/100)
    "clear_sky_factor",  # 1 - cloudcover/100
]


def compute_day_ahead_capacity_factor(df: pd.DataFrame) -> np.ndarray:
    """
    Compute day-ahead capacity factor = day-ahead MW / installed_capacity_mw.

    Args:
        df: DataFrame containing 'day-ahead' and 'installed_capacity_mw' columns.

    Returns:
        1D numpy array of day-ahead capacity factor values.
    """
    if "day-ahead" not in df.columns:
        raise ValueError("day-ahead column not found in data")
    if "installed_capacity_mw" not in df.columns:
        raise ValueError("installed_capacity_mw column not found in data")

    day_ahead_cf = (
        df["day-ahead"] / df["installed_capacity_mw"].replace(0, np.nan)
    ).fillna(0.0)

    return day_ahead_cf.values


def prepare_sequences_with_future(
    df: pd.DataFrame,
    sequence_length: int = 168,
    forecast_horizon: int = 336,
    features: Optional[List[str]] = None,
    filter_nighttime: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build encoder–decoder sequences for the CNN–LSTM model (direct prediction).

    Encoder input  X_enc: past `sequence_length` hours of weather/features.
    Decoder input  X_dec: next `forecast_horizon` hours of weather/features.
    Target         y:     next `forecast_horizon` hours of capacity_factor.

    Args:
        df: Processed zone DataFrame sorted by date.
        sequence_length: Number of past hours in encoder input.
        forecast_horizon: Number of future hours to forecast.
        features: Optional list of feature columns to include. If None,
                  WEATHER_FEATURES + ENGINEERED_FEATURES are used.
        filter_nighttime: If True, only train on sequences with significant daytime hours.
                         This prevents model from wasting capacity learning night→zero.

    Returns:
        X_enc: np.ndarray of shape (n_samples, sequence_length, n_features)
        X_dec: np.ndarray of shape (n_samples, forecast_horizon, n_features)
        y:     np.ndarray of shape (n_samples, forecast_horizon)
    """
    if features is None:
        features = WEATHER_FEATURES + ENGINEERED_FEATURES

    available_features = [f for f in features if f in df.columns]

    if "capacity_factor" not in df.columns:
        raise ValueError("capacity_factor column not found in data")
    
    # Get radiation values for nighttime filtering
    if filter_nighttime and "shortwave_radiation" in df.columns:
        radiation = df["shortwave_radiation"].values
        RADIATION_THRESHOLD = 1e-3  # Same as evaluation
    else:
        radiation = None

    X_data = df[available_features].values
    y_base = df["capacity_factor"].values

    X_enc_list: List[np.ndarray] = []
    X_dec_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    
    skipped_nighttime = 0

    for i in range(len(df) - sequence_length - forecast_horizon + 1):
        enc_start = i
        enc_end = i + sequence_length
        dec_end = enc_end + forecast_horizon

        X_enc = X_data[enc_start:enc_end]
        X_dec = X_data[enc_end:dec_end]
        y_seq = y_base[enc_end:dec_end]

        # Skip sequences with NaNs
        if (
            np.isnan(X_enc).any()
            or np.isnan(X_dec).any()
            or np.isnan(y_seq).any()
        ):
            continue
        
        # Filter nighttime-only sequences (optimization for daytime-focused training)
        if filter_nighttime and radiation is not None:
            # Check radiation in forecast horizon
            rad_forecast = radiation[enc_end:dec_end]
            daytime_hours = np.sum(rad_forecast >= RADIATION_THRESHOLD)
            daytime_fraction = daytime_hours / forecast_horizon
            
            # Skip if less than 20% daytime hours (mostly nighttime)
            # This prevents wasting model capacity on trivial night→zero patterns
            if daytime_fraction < 0.2:
                skipped_nighttime += 1
                continue

        X_enc_list.append(X_enc)
        X_dec_list.append(X_dec)
        y_list.append(y_seq)

    X_enc = np.array(X_enc_list)
    X_dec = np.array(X_dec_list)
    y = np.array(y_list)

    print(f"Created {len(X_enc)} encoder–decoder sequences:")
    if filter_nighttime and skipped_nighttime > 0:
        total_possible = len(X_enc) + skipped_nighttime
        print(f"  → Filtered out {skipped_nighttime}/{total_possible} nighttime-heavy sequences ({100*skipped_nighttime/total_possible:.1f}%)")
        print(f"  → Kept {len(X_enc)} sequences with meaningful daytime hours")
    print(
        f"  Encoder input shape: {X_enc.shape} "
        f"(samples, enc_timesteps={sequence_length}, features={len(available_features)})"
    )
    print(
        f"  Decoder input shape: {X_dec.shape} "
        f"(samples, dec_timesteps={forecast_horizon}, features={len(available_features)})"
    )
    print(f"  Target shape: {y.shape} (samples, forecast_horizon={forecast_horizon})")

    return X_enc, X_dec, y


def load_zone_data(zone: str, split: str = "train", use_actual_weather: bool = False) -> pd.DataFrame:
    """
    Load processed data for a specific bidding zone.

    Args:
        zone: Zone name (e.g., 'IT-NORD').
        split: 'train' or 'test'.
        use_actual_weather: If True and split='test', load from test_actual_weather directory.
                           This uses observed weather instead of forecasts for testing.

    Returns:
        DataFrame with zone data, sorted by date, with engineered features added.
    """
    zone_lower = zone.lower().replace("-", "_")
    
    # Use actual weather test directory if requested
    if use_actual_weather and split == "test":
        filepath = Path(f"data/processed/test_actual_weather/{zone_lower}.csv")
    else:
        filepath = Path(f"data/processed/{split}/{zone_lower}.csv")

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Add engineered features to help model learn weather patterns
    # Handle both 'cloudcover' and 'cloud_cover' column names
    cloud_col = 'cloudcover' if 'cloudcover' in df.columns else 'cloud_cover'
    if cloud_col in df.columns:
        df["solar_potential"] = df["shortwave_radiation"] * (1 - df[cloud_col] / 100)
        df["clear_sky_factor"] = 1 - df[cloud_col] / 100

    weather_type = " (ACTUAL WEATHER)" if use_actual_weather and split == "test" else ""
    print(
        f"Loaded {zone} {split}{weather_type}: {len(df)} records "
        f"({df['date'].min()} to {df['date'].max()})"
    )

    return df


def load_all_zones(split: str = "train") -> Dict[str, pd.DataFrame]:
    """
    Load data for all predefined zones.

    Args:
        split: 'train' or 'test'.

    Returns:
        Dictionary mapping zone name -> DataFrame.
    """
    data: Dict[str, pd.DataFrame] = {}
    for zone in ZONES:
        try:
            data[zone] = load_zone_data(zone, split)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue

    return data


def prepare_sequences(
    df: pd.DataFrame,
    sequence_length: int = 168,  # 7 days
    forecast_horizon: int = 336,  # 14 days
    features: Optional[List[str]] = None,
    use_residual: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding-window sequences for the single-input CNN–LSTM.

    If use_residual=True, the target y is:
        residual = capacity_factor - day_ahead_capacity_factor
    otherwise y is raw capacity_factor.

    Args:
        df: Processed zone DataFrame sorted by date.
        sequence_length: Number of past hours in the input window.
        forecast_horizon: Number of future hours to predict.
        features: Optional list of feature column names.
        use_residual: If True, produce residuals instead of raw CF.

    Returns:
        X: np.ndarray of shape (n_samples, sequence_length, n_features)
        y: np.ndarray of shape (n_samples, forecast_horizon)
    """
    if features is None:
        # Use ONLY weather + engineered features (NO hour!)
        features = WEATHER_FEATURES + ENGINEERED_FEATURES

    available_features = [f for f in features if f in df.columns]

    print("FEATURE COLS USED IN SEQUENCES:")
    print(available_features)
    print("Number of feature columns:", len(available_features))

    if "capacity_factor" not in df.columns:
        raise ValueError("capacity_factor column not found in data")

    # Input features
    X_data = df[available_features].values

    # Target: either capacity_factor or residual
    if use_residual:
        day_ahead_cf = compute_day_ahead_capacity_factor(df)
        y_base = df["capacity_factor"].values - day_ahead_cf
    else:
        y_base = df["capacity_factor"].values

    X_sequences: List[np.ndarray] = []
    y_sequences: List[np.ndarray] = []

    for i in range(len(df) - sequence_length - forecast_horizon + 1):
        X_seq = X_data[i : i + sequence_length]
        y_seq = y_base[i + sequence_length : i + sequence_length + forecast_horizon]

        if not np.isnan(X_seq).any() and not np.isnan(y_seq).any():
            X_sequences.append(X_seq)
            y_sequences.append(y_seq)

    X = np.array(X_sequences)
    y = np.array(y_sequences)

    print(f"Created {len(X)} sequences:")
    print(
        f"  Input shape: {X.shape} "
        f"(samples, timesteps={sequence_length}, features={len(available_features)})"
    )
    print(f"  Target shape: {y.shape} (samples, forecast_horizon={forecast_horizon})")

    return X, y


def prepare_test_sequences(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Prepare test data for prediction on a contiguous horizon.

    For the test set, we typically have a 14-day forecast period.
    We use the entire test period as a single sequence.

    Args:
        df: Test DataFrame with weather forecast data (and possibly capacity_factor).
        features: Optional list of feature columns to use.

    Returns:
        X: np.ndarray of shape (1, n_timesteps, n_features)
        y: Optional[np.ndarray] of shape (1, n_timesteps) if capacity_factor present,
           otherwise None.
    """
    if features is None:
        # For test data, we only have weather forecast + capacity_factor
        features = WEATHER_FEATURES + ["hour"]

    available_features = [f for f in features if f in df.columns]

    X = df[available_features].values
    y: Optional[np.ndarray]
    y = df["capacity_factor"].values if "capacity_factor" in df.columns else None

    # Reshape to (1, timesteps, features) for single prediction
    X = X.reshape(1, X.shape[0], X.shape[1])
    if y is not None:
        y = y.reshape(1, -1)

    print("Test sequence prepared:")
    print(
        f"  Input shape: {X.shape} "
        f"(1 sample, timesteps={df.shape[0]}, features={len(available_features)})"
    )
    if y is not None:
        print(f"  Target shape: {y.shape} (1 sample, timesteps={df.shape[0]})")

    return X, y


def normalize_data(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]]:
    """
    Normalize features using training set statistics.

    Args:
        X_train: Training data of shape (n_samples, T, F).
        X_test: Optional test data of shape (m_samples, T, F).

    Returns:
        X_train_normalized: np.ndarray with same shape as X_train.
        X_test_normalized: Optional[np.ndarray] with same shape as X_test.
        normalization_params: dict with keys 'mean' and 'std' (1D arrays of length F).
    """
    # Calculate mean and std for each feature across all samples and timesteps
    mean = X_train.mean(axis=(0, 1))
    std = X_train.std(axis=(0, 1))

    # Avoid division by zero
    std[std == 0] = 1.0

    X_train_norm = (X_train - mean) / std

    X_test_norm: Optional[np.ndarray] = None
    if X_test is not None:
        X_test_norm = (X_test - mean) / std

    normalization_params: Dict[str, np.ndarray] = {"mean": mean, "std": std}

    print("Data normalized:")
    print(f"  Train shape: {X_train_norm.shape}")
    if X_test_norm is not None:
        print(f"  Test shape: {X_test_norm.shape}")

    return X_train_norm, X_test_norm, normalization_params


def get_data_summary(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Get summary statistics for a dict of zone DataFrames.

    Args:
        data_dict: Mapping from zone name -> DataFrame.

    Returns:
        Summary DataFrame with record counts, time span, CF stats and NaNs.
    """
    summary_rows: List[dict] = []
    for zone, df in data_dict.items():
        summary_rows.append(
            {
                "Zone": zone,
                "Records": len(df),
                "Start": df["date"].min(),
                "End": df["date"].max(),
                "Capacity Factor Mean": df["capacity_factor"].mean(),
                "Capacity Factor Std": df["capacity_factor"].std(),
                "Missing Values": df.isnull().sum().sum(),
            }
        )

    return pd.DataFrame(summary_rows)
