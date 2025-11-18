# Solar Forecasting - Usage Guide

## 🚀 Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Evaluate trained model (generates all reports)
python src/evaluate_forecast.py

# Train new model
python src/train.py --zones IT-NORD --epochs 25 --model-type encoder --residual
```

## 📊 Evaluation Outputs

When you run `python src/evaluate_forecast.py`, it generates comprehensive reports:

### Console Output
- Summary metrics (MAE, RMSE, R²)
- Detailed metrics table (15+ metrics)
- File paths for all generated reports

### Generated Files (in `results/<zone>/`)

1. **`predictions.csv`** - Hour-by-hour predictions
   - Columns: datetime, actual_mw, predicted_mw, error_mw, absolute_error_mw, percent_error
   - 336 rows (14 days × 24 hours)
   - Perfect for detailed analysis in Excel/Python

2. **`forecast.png`** - Time series visualization
   - Actual vs predicted power over 14 days
   - Metrics box (MAE, RMSE, R²)
   - Date formatting for easy interpretation

3. **`scatter_plot.png`** - Predicted vs actual
   - Each point = one hour
   - Perfect prediction line (red dashed)
   - Helps identify systematic bias
   - Equal aspect ratio for fair comparison

4. **`error_analysis.png`** - 4-subplot error analysis
   - **Top-left**: Error histogram (distribution)
   - **Top-right**: Absolute error histogram
   - **Bottom-left**: Error over time (shows patterns)
   - **Bottom-right**: Percentage error histogram

5. **`hourly_performance.png`** - Performance by hour of day
   - **Left**: MAE by hour (bars + error bars)
   - **Right**: Average actual vs predicted by hour
   - Identifies which hours are hardest to predict

6. **`metrics.csv`** - Comprehensive metrics table
   - 15 different metrics
   - Easy to import into reports
   - Includes bias, median error, MAPE, etc.

## 📈 Training Outputs

When you run training, files are saved to `models/<zone>/`:

1. **`model.pt`** - Trained model checkpoint (~2.8 MB)
   - Contains model weights
   - Can be loaded for prediction

2. **`norm.npz`** - Normalization parameters (~740 B)
   - Mean and std for each feature
   - Required for prediction

3. **`training_history.json`** - Epoch-by-epoch metrics
   - Train/val loss and MAE for each epoch
   - Learning rate history
   - JSON format for easy parsing

4. **`training_curves.png`** - Training visualization
   - Loss curves (train vs validation)
   - MAE curves (train vs validation)
   - **Overfitting indicator** (red warning if val >> train)
   - **Best epoch marker** (vertical line on MAE plot)
   - Helps diagnose training issues

## 🎯 Understanding the Plots

### Forecast Plot (`forecast.png`)
- **Purpose**: See how well model tracks actual production
- **Look for**: 
  - Do curves follow similar patterns?
  - Are peaks captured?
  - Nighttime correctly at zero?

### Scatter Plot (`scatter_plot.png`)
- **Purpose**: Identify systematic bias
- **Look for**:
  - Points close to red line = good predictions
  - Points above line = over-prediction
  - Points below line = under-prediction
  - Tight cluster = consistent performance

### Error Analysis (`error_analysis.png`)
- **Top-left (Error histogram)**:
  - Centered at 0 = no bias
  - Narrow = consistent predictions
- **Top-right (Absolute error)**:
  - Most errors should be small
  - Long tail = occasional large errors
- **Bottom-left (Error over time)**:
  - Patterns indicate systematic issues
  - Random scatter = good model
- **Bottom-right (Percentage error)**:
  - Centered at 0 = no proportional bias
  - Wide = struggles with different magnitudes

### Hourly Performance (`hourly_performance.png`)
- **Left (MAE by hour)**:
  - Which hours are hardest?
  - Morning/evening transitions typically hardest
  - Nighttime should be perfect (zero error)
- **Right (Average by hour)**:
  - Does model capture daily pattern?
  - Similar bar heights = good fit

### Training Curves (`training_curves.png`)
- **Overfitting**: Validation loss >> training loss
  - Gap keeps growing
  - Red warning shown
- **Underfitting**: Both losses high and decreasing
  - Train longer
  - Or use more complex model
- **Good fit**: Validation loss close to training loss
  - Green checkmark shown
  - Gap stable or shrinking
- **Best epoch**: Vertical line shows lowest validation MAE

## 📁 File Organization

```
models/
└── it-nord/
    ├── model.pt                 # 2.8 MB - model weights
    ├── norm.npz                 # 740 B - normalization
    ├── training_history.json    # epoch metrics
    └── training_curves.png      # training visualization

results/
└── it-nord/
    ├── forecast.png             # 184 KB - time series
    ├── predictions.csv          # 22 KB - hour-by-hour data
    ├── scatter_plot.png         # 92 KB - pred vs actual
    ├── error_analysis.png       # 161 KB - error breakdown
    ├── hourly_performance.png   # 67 KB - by hour analysis
    └── metrics.csv              # 457 B - all metrics
```

## 🔧 Training Parameters

```bash
python src/train.py \
  --zones IT-NORD \              # Which zone(s) to train on
  --epochs 25 \                  # Number of training epochs
  --model-type encoder \         # Use encoder-decoder architecture
  --residual \                   # Predict residuals (recommended)
  --batch-size 32 \              # Batch size (default: 32)
  --lr 0.001                     # Learning rate (default: 0.001)
```

**Recommended settings:**
- Single zone: `--zones IT-NORD` (faster, zone-specific)
- Multi-zone: `--zones IT-NORD IT-CSUD` (slower, generalizes better)
- Epochs: 25-50 (25 usually sufficient)
- Always use `--residual` flag (better performance)

## 📊 Metrics Explained

| Metric | What it means | Good value |
|--------|---------------|------------|
| **MAE** | Average absolute error | < 150 MW |
| **RMSE** | Penalizes large errors more | < 300 MW |
| **R²** | Variance explained (0-1) | > 0.95 |
| **Bias** | Systematic over/under prediction | Close to 0 |
| **Median AE** | Middle value of errors | < MAE |
| **MAPE** | Percentage error | < 20% |

## 🚦 Troubleshooting

### High errors during specific hours
- Check `hourly_performance.png`
- Morning/evening transitions are naturally harder
- Consider adding more time-based features

### Overfitting (val loss >> train loss)
- Reduce epochs
- Increase dropout
- Add more training data
- Use regularization

### Underfitting (both losses high)
- Train longer (more epochs)
- Use larger model
- Check if data quality is good
- Ensure normalization is working

### Systematic bias (errors not centered at 0)
- Check `error_analysis.png` histogram
- May need to adjust residual learning
- Check if nighttime zeroing is working

## 🎓 Best Practices

1. **Always check training curves** after training
   - Look for overfitting warning
   - Note the best epoch
   
2. **Review all evaluation plots** before reporting
   - Forecast plot: overall fit
   - Scatter plot: systematic bias
   - Error analysis: distribution
   - Hourly: time-of-day patterns

3. **Use CSV files for detailed analysis**
   - `predictions.csv`: identify problematic hours
   - `metrics.csv`: copy metrics to reports

4. **Keep zone folders organized**
   - One folder per zone
   - All related files together
   - Easy to compare zones

5. **Document your experiments**
   - Note which parameters worked best
   - Keep best model files
   - Save training curves for comparison

---

**Last Updated:** November 18, 2025  
**Version:** 2.0.0 (Zone-based with comprehensive reporting)
