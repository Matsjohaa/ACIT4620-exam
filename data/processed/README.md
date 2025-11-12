# Processed Data - Train/Test Split

## Directory Structure

```
data/processed/
├── train/              # Training data (2015-2025 up to Oct 27)
│   ├── it_nord.csv
│   ├── it_cnor.csv
│   ├── it_csud.csv
│   ├── it_sud.csv
│   ├── it_sici.csv
│   ├── it_sard.csv
│   ├── it_cala.csv
│   └── italy_all_zones.csv
│
├── test/               # Test data (Oct 28 - Nov 10, 2025)
│   ├── it_nord.csv
│   ├── it_cnor.csv
│   ├── it_csud.csv
│   ├── it_sud.csv
│   ├── it_sici.csv
│   ├── it_sard.csv
│   ├── it_cala.csv
│   └── italy_all_zones.csv
│
└── [zone]_processed.csv  # Full merged datasets (before split)
```

---

## Training Data (`train/`)

**Purpose:** Historical data for model training

**Date Range:** 
- IT-NORD, IT-CNOR, IT-SICI, IT-SARD: 2015-01-01 to 2025-10-27
- IT-CSUD: 2017-01-01 to 2025-10-27
- IT-SUD, IT-CALA: 2021-01-01 to 2025-10-27

**Columns (21):**
- `date` - Timestamp (hourly, UTC)
- `zone` - Bidding zone identifier
- `actual` - Actual solar generation (MW)
- `day-ahead` - Day-ahead forecast (MW)
- `intraday` - Intraday forecast (MW)
- `installed_capacity_mw` - Installed PV capacity (MW)
- `capacity_factor` - Actual / Capacity (0-1)
- `hour` - Hour of day (0-23)
- **Weather parameters (13):**
  - `shortwave_radiation` (W/m²)
  - `direct_radiation` (W/m²)
  - `diffuse_radiation` (W/m²)
  - `temperature_2m` (°C)
  - `apparent_temperature` (°C)
  - `cloudcover` (%)
  - `dewpoint_2m` (°C)
  - `precipitation` (mm)
  - `windspeed_10m` (m/s)
  - `windgusts_10m` (m/s)
  - `winddirection_10m` (degrees)
  - `surface_pressure` (hPa)
  - `relativehumidity_2m` (%)

**Record Counts:**
- IT-NORD: 94,520 records
- IT-CNOR: 94,520 records
- IT-CSUD: 77,049 records
- IT-SUD: 42,080 records
- IT-SICI: 85,809 records
- IT-SARD: 85,809 records
- IT-CALA: 42,080 records
- **Total: 521,867 records**

---

## Test Data (`test/`)

**Purpose:** Validation data simulating real forecasting scenario

**Date Range:** 2025-10-28 to 2025-11-06 (10 days, 240 hours)
- IT-CALA: Only to 2025-10-29 (70 hours) due to weather data availability

**Columns (19):**
- `date` - Timestamp (hourly, UTC)
- `zone` - Bidding zone identifier
- `installed_capacity_mw` - Installed PV capacity (MW)
- `capacity_factor` - **Actual values for validation only** (0-1)
- `actual` - **Actual generation for validation only** (MW)
- `hour` - Hour of day (0-23)
- **Weather forecast parameters (13):** Same as training

**Columns REMOVED from test (to simulate real forecasting):**
- ❌ `day-ahead` - Not available in real forecasting
- ❌ `intraday` - Not available in real forecasting

**Record Counts:**
- IT-NORD: 263 records (10 days + 23 hours)
- IT-CNOR: 263 records
- IT-CSUD: 263 records
- IT-SUD: 263 records
- IT-SICI: 263 records
- IT-SARD: 263 records
- IT-CALA: 70 records (3 days - 2 hours)
- **Total: 1,648 records**

---

## Key Differences: Train vs Test

| Feature | Training Data | Test Data |
|---------|--------------|-----------|
| **Purpose** | Learn patterns | Validate predictions |
| **Date Range** | 2015-2025 (up to Oct 27) | Oct 28 - Nov 10, 2025 |
| **Historical Production** | ✅ Included | ❌ Excluded |
| **Day-Ahead Forecast** | ✅ Included | ❌ Excluded |
| **Intraday Forecast** | ✅ Included | ❌ Excluded |
| **Weather Data** | ✅ Historical | ✅ **Forecast** |
| **Capacity** | ✅ Included | ✅ Included |
| **Actual Values** | ✅ Included | ✅ **For validation only** |

---

## Usage in Model Training

### Training Phase:
```python
import pandas as pd

# Load training data
train = pd.read_csv('data/processed/train/it_nord.csv')

# Use ALL features for training:
# - Historical production patterns
# - Weather correlations
# - Day-ahead forecast biases
# - Temporal patterns
```

### Prediction Phase:
```python
# Load test data
test = pd.read_csv('data/processed/test/it_nord.csv')

# Model inputs:
# - Weather forecast (shortwave_radiation, temperature, etc.)
# - Installed capacity (for denormalization)
# - Temporal features (hour, day of year)

# Model outputs:
# - Predicted capacity_factor for next 14 days

# Validation:
# - Compare predictions vs test['actual']
# - Calculate MAE, RMSE, MAPE
```

---

## Realistic Forecasting Simulation

This train/test split simulates a **realistic operational scenario**:

1. **On October 27, 2025 at midnight**, we have:
   - ✅ All historical data up to Oct 27
   - ✅ 14-day weather forecast (Oct 28 - Nov 10)
   - ❌ No future production data
   - ❌ No day-ahead or intraday forecasts for future period

2. **Model task**: Predict solar production for Oct 28 - Nov 10 using ONLY:
   - Historical patterns learned during training
   - Weather forecast for the target period
   - Solar geometry (sun angle, day length)

3. **Validation**: On Nov 12, 2025, we compare predictions against actual observed values

This setup **prevents data leakage** and tests the model's ability to generalize to truly unseen future data.

---

## Data Quality Notes

### Hourly Resolution
- All data aggregated to hourly resolution
- Post-Jan 1, 2025 data (originally 15-minute) averaged to hourly
- Consistent with weather data resolution

### Missing Values
- Training data filtered to include only records with actual generation
- Some zones have shorter history (IT-SUD, IT-CALA from 2021)
- IT-CALA test data ends Oct 29 (weather forecast limitation)

### Capacity Factor Range
- Training mean: 17.48% (realistic for Italian solar)
- Range: 0% (night) to 100% (peak production)
- Clipped to [0, 1] to handle any numerical issues

---

## Next Steps

1. **Feature Engineering**: Add temporal, solar geometry, and lag features
2. **Sequence Generation**: Create sliding windows for CNN-LSTM
3. **Model Training**: Train on historical patterns
4. **Prediction**: Generate 14-day forecasts
5. **Validation**: Compare predictions vs actual values in test set
