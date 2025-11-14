# Source Code - CNN-LSTM Solar Forecasting

This directory contains the implementation of the CNN-LSTM hybrid model for 14-day solar power forecasting in Italy.

## Project Structure

```
src/
├── data_loader.py      # Data loading and sequence preparation
├── model.py            # CNN-LSTM model architecture
├── train.py            # Training pipeline
├── predict.py          # Prediction and evaluation
└── features.py         # Feature engineering (optional)
```

## Architecture

The model follows the hybrid approach from literature (Salman et al., 2024):

1. **CNN Layers**: Extract local temporal patterns from weather data
   - 2 Conv1D layers (64, 128 filters)
   - Batch normalization and dropout for regularization

2. **LSTM Layers**: Capture long-term temporal dependencies
   - 2 LSTM layers (128, 64 units)
   - Recurrent dropout for sequence modeling

3. **Dense Layers**: Generate 14-day forecast
   - Dense layers (256, 128 units)
   - Output layer: 336 values (14 days × 24 hours)

## Training Configuration

- **Input**: 7 days (168 hours) of weather data + capacity factor
- **Output**: 14 days (336 hours) of capacity factor predictions
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Nadam (best per literature)
- **Features**: 13 weather parameters + capacity factor + hour (15 total)

## Weather Features Used

From OpenMeteo weather forecasts:
1. `shortwave_radiation` - Global horizontal irradiance
2. `direct_radiation` - Direct normal irradiance
3. `diffuse_radiation` - Diffuse horizontal irradiance
4. `temperature_2m` - Air temperature at 2m
5. `apparent_temperature` - Feels-like temperature
6. `cloudcover` - Cloud coverage percentage
7. `dewpoint_2m` - Dew point temperature
8. `precipitation` - Rainfall
9. `windspeed_10m` - Wind speed at 10m
10. `windgusts_10m` - Wind gusts
11. `winddirection_10m` - Wind direction
12. `surface_pressure` - Surface air pressure
13. `relativehumidity_2m` - Relative humidity

## Usage

### 1. Install Dependencies

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

### 2. Train Model

Train on all zones:
```bash
python src/train.py --epochs 100 --batch-size 32
```

Train on specific zones:
```bash
python src/train.py --zones IT-NORD IT-CNOR --epochs 50
```

Options:
- `--zones`: Zones to train on (default: all)
- `--sequence-length`: Input window in hours (default: 168 = 7 days)
- `--forecast-horizon`: Output window in hours (default: 336 = 14 days)
- `--batch-size`: Batch size (default: 32)
- `--epochs`: Maximum epochs (default: 100)
- `--model-type`: 'full' or 'simple' (default: full)
- `--model-path`: Model save directory (default: models/)

### 3. Evaluate Model

```bash
python src/predict.py --model-path models/best_model_full.keras
```

This will:
- Load the trained model
- Make predictions on test data (Oct 27 - Nov 10, 2025)
- Calculate metrics (MAE, RMSE, MAPE, R²)
- Generate visualization plots
- Save results to `results/` directory

## Output Files

### Training
- `models/best_model_full.keras` - Trained model
- `models/normalization_params.npz` - Feature normalization parameters
- `models/training_history_full.json` - Training metrics per epoch
- `models/training_history_full.png` - Training curves visualization

### Evaluation
- `results/evaluation_summary.csv` - Performance metrics by zone
- `results/{zone}_forecast.png` - Individual zone predictions
- `results/all_zones_forecast.png` - Overview of all zones
- `results/evaluation_metrics.png` - Error metrics comparison

## Data Flow

```
1. Load processed data
   ├── Train: 2015 - Oct 26, 2025 (historic weather)
   └── Test: Oct 27 - Nov 10, 2025 (weather forecast)

2. Create sequences
   ├── Training: Sliding windows (7 days → 14 days prediction)
   └── Test: Single 14-day forecast period

3. Normalize features
   └── Using training set statistics (mean, std)

4. Train model
   ├── CNN extracts weather patterns
   ├── LSTM captures temporal dependencies
   └── Dense layer outputs 14-day forecast

5. Evaluate on test set
   ├── Compare predictions vs actual production
   └── Calculate MAE, RMSE, MAPE, R²
```

## Training Data

- **Zones**: 7 Italian bidding zones
- **Period**: 2015 - Oct 26, 2025 (up to 10.7 years)
- **Samples**: ~94,000 sequences per zone (with sliding window)
- **Features**: Weather forecast + capacity factor + hour
- **Target**: Capacity factor (actual generation / installed capacity)

## Test Data

- **Period**: Oct 27 - Nov 10, 2025 (14 days)
- **Source**: Weather FORECAST made on Oct 27
- **Purpose**: Simulate real operational forecasting
- **Validation**: Actual production values available for evaluation

## Key Design Decisions

1. **Capacity Factor Normalization**: Dividing by installed capacity accounts for 62% capacity growth over 11 years

2. **Weather Forecast for Testing**: Test set uses forecast weather (not historic) to simulate real forecasting scenario

3. **No Day-Ahead in Test**: Test set excludes day-ahead forecasts to evaluate model's ability to predict from weather alone

4. **7-Day Input Window**: Provides sufficient context for weekly patterns and trends

5. **14-Day Output Horizon**: Extends traditional day-ahead to two-week planning window

## Expected Performance

Based on literature:
- **Target MAE**: < 5% (Laaroussi et al., 2022)
- **Target MAPE**: < 5% (Salman et al., 2024 achieved < 1% on French data)
- **Validation**: Oct 27 - Nov 10 forecast period

Note: This is a "proof of concept" demonstrating hybrid CNN-LSTM approach rather than operational system.

## References

1. Salman et al. (2024): Hybrid deep learning models for time series forecasting of solar power
2. Laaroussi et al. (2022): Solar power forecasting using CNN-LSTM hybrid model
