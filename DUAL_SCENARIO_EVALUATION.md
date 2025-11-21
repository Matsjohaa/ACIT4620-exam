# Dual-Scenario Evaluation System

## Overview
The evaluation script (`src/evaluate_forecast.py`) has been upgraded to automatically test models with **both weather forecast and actual weather data** in a single run, providing comprehensive comparison visualizations with distinct color coding.

## Key Features

### Automatic Dual Evaluation
- **No flags needed**: Simply run `python src/evaluate_forecast.py --zones IT-NORD`
- Automatically evaluates both scenarios:
  1. **Weather Forecast** (realistic 2-week forecast scenario)
  2. **Actual Weather** (upper bound performance benchmark)

### Color-Coded Visualizations
- **Black (#000000)**: Actual production (ground truth)
- **Red (#FF6B6B)**: Predictions using weather forecasts
- **Teal (#4ECDC4)**: Predictions using actual weather data

This color scheme makes it easy to visually distinguish:
- How well the model performs in realistic conditions (red)
- The model's true learning capability when weather is perfect (teal)
- The gap between them shows weather forecast error impact

## Output Files

### 1. Time Series Comparison (`forecast_comparison.png`)
- Overlay plot showing actual production vs both prediction types
- Color-coded with red (forecast) and teal (actual weather)
- Includes metric boxes for each scenario
- Shows temporal patterns and differences

### 2. Scatter Plot Comparison (`scatter_comparison.png`)
- Side-by-side scatter plots for both scenarios
- Perfect prediction line for reference
- Metrics boxes showing MAE, RMSE, R² for each
- Easy visual comparison of model fit quality

### 3. Error Analysis Comparison (`error_comparison.png`)
- 4-panel comprehensive error analysis:
  1. **Error Distribution**: Overlaid histograms showing forecast vs actual weather errors
  2. **Absolute Errors Over Time**: Time series of error magnitudes
  3. **Boxplot Comparison**: Statistical distribution of errors
  4. **Metrics Bar Chart**: Visual comparison of MAE and RMSE

### 4. Predictions CSV (`predictions_comparison.csv`)
Columns:
- `datetime`: Timestamp
- `actual_mw`: Ground truth production
- `predicted_forecast_mw`: Predictions with weather forecasts
- `predicted_actual_mw`: Predictions with actual weather
- `error_forecast_mw`: Forecast weather prediction errors
- `error_actual_mw`: Actual weather prediction errors
- `abs_error_forecast_mw`: Absolute errors (forecast)
- `abs_error_actual_mw`: Absolute errors (actual)

### 5. Metrics Comparison Table (`metrics_comparison.csv`)
Side-by-side metrics:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R² Score
- Mean Error (Bias)
- Median Absolute Error
- Standard Deviation of Errors

### 6. All Zones Summary (`results/all_zones_summary.csv`)
Multi-zone evaluation results with both scenarios per zone.

## Usage

### Basic Usage
```bash
# Evaluate single zone with both scenarios
python src/evaluate_forecast.py --zones IT-NORD

# Evaluate all zones with both scenarios
python src/evaluate_forecast.py

# Evaluate specific zones
python src/evaluate_forecast.py --zones IT-NORD IT-SUD
```

### Output Example
```
================================================================================
RESULTS COMPARISON
================================================================================
Scenario             MAE (MW)     RMSE (MW)    R²        
--------------------------------------------------------------------------------
Forecast Weather     510.60       1008.54      0.487     
Actual Weather       414.29       822.62       0.659     
================================================================================
```

## Interpretation Guide

### Understanding the Dual Results

**Forecast Weather Performance** (Red):
- Real-world performance metric
- Includes weather forecast errors
- What you'd get in production deployment
- IT-NORD: MAE = 511 MW, R² = 0.487

**Actual Weather Performance** (Teal):
- Upper bound of model capability
- Shows what the model learned (no weather error)
- Diagnostic tool to separate model vs data issues
- IT-NORD: MAE = 414 MW, R² = 0.659

### Key Insights from Comparison

1. **Model Quality Check**:
   - If R² with actual weather is high (>0.6): Model learned correctly ✓
   - If R² with actual weather is low (<0.3): Model has issues ✗

2. **Weather Forecast Impact**:
   - Difference between scenarios shows weather error contribution
   - IT-NORD: ~19% of error comes from weather forecasts
   - Formula: `(MAE_forecast - MAE_actual) / MAE_forecast * 100`

3. **Bias Analysis**:
   - Both scenarios show consistent underprediction bias (~-45%)
   - This indicates systematic model behavior, not random
   - Can be corrected with bias adjustment: `predicted * 1.45`

## Technical Details

### New Functions Added

1. **`save_comparison_csv()`**: Creates comprehensive predictions CSV
2. **`plot_forecast_comparison()`**: Time series with color coding
3. **`plot_scatter_comparison()`**: Side-by-side scatter analysis
4. **`plot_error_comparison()`**: 4-panel error diagnostics
5. **`create_comparison_metrics_table()`**: Metrics comparison

### Code Structure
```python
def evaluate_zone(zone):
    # Load both test datasets
    test_df_forecast = load_test_data(zone, forecast=True)
    test_df_actual = load_test_data(zone, forecast=False)
    
    # Evaluate both scenarios
    results_forecast = evaluate_single_scenario(..., test_df_forecast, "Forecast Weather")
    results_actual = evaluate_single_scenario(..., test_df_actual, "Actual Weather")
    
    # Generate comparison visualizations
    save_comparison_csv(...)
    plot_forecast_comparison(...)  # Red vs Teal
    plot_scatter_comparison(...)
    plot_error_comparison(...)
    create_comparison_metrics_table(...)
```

## Benefits

1. **Diagnostic Power**: Separate model issues from data quality issues
2. **Single Command**: No need to run evaluation twice manually
3. **Clear Visualization**: Color-coded plots make differences obvious
4. **Comprehensive Reports**: All metrics and plots in one run
5. **Production Ready**: Forecast weather results show real-world performance
6. **Research Insight**: Actual weather results show model learning quality

## Results Summary - IT-NORD Small Model (242K params)

| Scenario | MAE (MW) | RMSE (MW) | R² | Bias |
|----------|----------|-----------|-----|------|
| **Forecast Weather** | 511 | 1,009 | 0.487 | -48% |
| **Actual Weather** | 414 | 823 | 0.659 | -44% |
| **Improvement** | 19% | 18% | 35% | - |

**Key Finding**: The model learned correctly (R² = 0.659 with perfect weather). About 19% of the prediction error comes from weather forecast inaccuracy, not model issues.

## Legacy Mode Removed

The old `--actual-weather` flag has been removed. The script now **always evaluates both scenarios automatically**. This ensures:
- Consistent evaluation methodology
- Complete performance picture
- No manual flag management
- Reduced user error

## Files Modified

- `src/evaluate_forecast.py`: Complete refactoring for dual-scenario evaluation
  - Added 5 new comparison functions (~200 lines)
  - Refactored `evaluate_zone()` for automatic dual testing
  - Updated summary output format
  - Enhanced color scheme throughout

## Next Steps

To evaluate other zones with the new system:
```bash
# Evaluate all 6 zones with both scenarios
python src/evaluate_forecast.py
```

This will generate comprehensive comparison reports for all zones, making it easy to:
- Compare model performance across zones
- Identify zones where weather forecasts are most limiting
- Validate model learning quality across different regions
- Spot systematic biases or issues
