# Codebase Simplification - Residual Learning Removed

**Date:** November 20, 2025  
**Purpose:** Remove all residual learning code to avoid confusion. The system now ONLY supports direct prediction.

## Summary

Successfully removed all residual learning functionality from the codebase. The system now exclusively performs **direct prediction** - predicting solar production directly from weather data, without using day-ahead forecasts as input.

## Changes Made

### 1. Deleted Files
- **`src/forecast_production.py`** - Entire file removed (was residual-only)

### 2. Modified Files

#### `src/evaluate_forecast.py`
- **Removed:** `compute_day_ahead_capacity_factor` import
- **Removed:** Day-ahead capacity factor computation in `build_test_sample()`
- **Removed:** Residual correlation calculation
- **Changed:** Prediction logic from adding residual to day-ahead → direct capacity factor prediction
- **Updated:** Function signatures to not return `day_ahead_cf`
- **Updated:** Documentation to clarify "direct prediction mode"

Key changes:
```python
# OLD (residual learning):
resid_cf = model(Xe, Xd)
cf_pred = np.clip(day_ahead_cf + resid_cf, 0.0, 1.2)
corr = np.corrcoef(y_true_cf - day_ahead_cf, resid_cf)[0, 1]

# NEW (direct prediction):
cf_pred = model(Xe, Xd)
cf_pred = np.clip(cf_pred, 0.0, 1.2)
```

#### `src/data_loader.py`
- **Removed:** `use_residual` parameter from `prepare_sequences_with_future()`
- **Removed:** Residual target calculation logic
- **Updated:** Documentation to remove residual learning references
- **Simplified:** Target is now always `capacity_factor` (never `capacity_factor - day_ahead_cf`)

Key changes:
```python
# OLD:
def prepare_sequences_with_future(..., use_residual: bool = False):
    if use_residual:
        day_ahead_cf = compute_day_ahead_capacity_factor(df)
        y_base = df["capacity_factor"].values - day_ahead_cf
    else:
        y_base = df["capacity_factor"].values

# NEW:
def prepare_sequences_with_future(...):
    y_base = df["capacity_factor"].values
```

#### `src/train.py`
- **Removed:** `use_residual` parameter from `train_model()` function
- **Removed:** `--residual` command line argument
- **Removed:** `use_residual` argument in `prepare_sequences_with_future()` call
- **Removed:** `args.residual` in `train_model()` call
- **Updated:** Documentation to clarify "direct prediction mode"
- **Updated:** Help text for `--large-model` to show correct parameter count (2.5M not 1M)

Key changes:
```python
# OLD:
def train_model(..., use_residual: bool = False, ...):
    """Train model. Can do residual or direct prediction."""
    ...
    X_enc, X_dec, y = prepare_sequences_with_future(..., use_residual=use_residual)

# NEW:
def train_model(...):
    """Train model (direct prediction mode)."""
    ...
    X_enc, X_dec, y = prepare_sequences_with_future(...)
```

## What Was Kept

### Architectural Residual Connections
**IMPORTANT:** `src/model_large.py` still contains "residual" in variable names and comments. These refer to **architectural residual connections** (skip connections in the neural network), NOT residual learning mode. These are a standard deep learning technique and should be kept.

Example (kept as-is):
```python
# This is ARCHITECTURE, not learning mode - perfectly fine!
z = self.decoder_fc4(z)
z = z + residual1  # Skip connection
```

### Utility Functions
- `compute_day_ahead_capacity_factor()` in `data_loader.py` - kept but unused
  - Still exists for potential future analysis
  - Not imported or called by any active code

## Testing Status

✅ **Code compiles** - No syntax errors  
✅ **No import errors** - All undefined references fixed  
✅ **Logical consistency** - All code paths use direct prediction  

⚠️ **Model checkpoint compatibility**: Current saved models (IT-NORD) are large models (29MB, 2.5M params). Use `--large-model` flag when evaluating.

## Current Model Status

| Zone | Model Type | Parameters | Status | Location |
|------|------------|------------|--------|----------|
| IT-NORD | Large V2 | 2.5M | ✅ Trained | `models/it-nord/model.pt` |
| Others | - | - | ❌ Not trained | - |

**Note:** All trained models are direct prediction (never used residual learning in practice, since `use_residual` defaulted to `False`).

## Command Examples

### Training (Direct Prediction Only)
```bash
# Standard model (242K params)
python src/train.py --zones IT-NORD

# Large model (2.5M params)
python src/train.py --zones IT-NORD --large-model
```

### Evaluation (Direct Prediction Only)
```bash
# Using large model with weather forecasts
python src/evaluate_forecast.py --zones IT-NORD --large-model

# Using large model with actual observed weather
python src/evaluate_forecast.py --zones IT-NORD --large-model --actual-weather
```

## Benefits of Simplification

1. **✅ Clearer Intent** - Code explicitly shows direct prediction
2. **✅ Less Confusion** - No mixing of residual and direct approaches
3. **✅ Simpler API** - Fewer parameters to worry about
4. **✅ Easier Maintenance** - One prediction mode to support
5. **✅ Matches Usage** - Reflects actual model usage (never used residual mode)

## What This Means

**The system predicts solar production directly from weather data.**

- **Input:** Historic weather (168h) + Future weather forecast (336h)
- **Output:** Predicted solar capacity factor (336h)
- **No day-ahead forecast needed at inference time**

This is suitable for:
- ✅ Long-term forecasts (2+ weeks)
- ✅ Scenarios without day-ahead forecasts
- ✅ Standalone weather-to-solar prediction

This is NOT suitable for:
- ❌ Short-term correction of day-ahead forecasts (would need residual learning)
- ❌ Scenarios where day-ahead is very accurate (direct prediction ceiling ~R²=0.47)

## Future Work

If you ever need residual learning again:
1. Restore `use_residual` parameter in `data_loader.py`
2. Restore `--residual` flag in `train.py`
3. Restore residual addition logic in `evaluate_forecast.py`
4. See git history for exact code to restore

But for now: **Direct prediction only!** 🎯
