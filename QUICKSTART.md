# CNN-LSTM Solar Forecasting - Quick Start Guide

## 🎯 Project Overview

14-day solar production forecasting for Italy's 7 bidding zones using CNN-LSTM hybrid deep learning model with **PyTorch**.

**Training Period**: 2015 - Oct 26, 2025 (with historic weather)  
**Test Period**: Oct 27 - Nov 10, 2025 (with weather forecast)  
**Architecture**: CNN → LSTM → Dense (87,824 parameters)

## ✅ Current Status

- [x] Data collection (ENTSO-E + OpenMeteo)
- [x] Preprocessing and train/test split
- [x] PyTorch model architecture implemented
- [x] Training pipeline ready (Apple Silicon optimized)
- [x] Evaluation pipeline ready
- [x] Data leakage fixed (14 features: weather + hour only)
- [ ] Full model training (next step)
- [ ] Evaluation and analysis

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Verify PyTorch with MPS (Apple Silicon GPU):
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, Device: {torch.device(\"mps\" if torch.backends.mps.is_available() else \"cpu\")}')"
```

### 2. Train Model

**Quick test** (10% sample, 1 zone, ~1 minute):
```bash
python src/train.py --zones IT-NORD --epochs 5 --batch-size 16 --sample 0.1
```

**Medium test** (30% sample, 1 zone, ~10 minutes):
```bash
python src/train.py --zones IT-NORD --epochs 20 --batch-size 32 --sample 0.3
```

**Full training** (all 7 zones, 2-4 hours):
```bash
python src/train.py --epochs 50 --batch-size 32
```

Train specific zones only:
```bash
python src/train.py --zones IT-NORD IT-CNOR --epochs 50
```

### 3. Evaluate Model

After training completes:
```bash
python src/predict.py
```

Evaluate specific zones:
```bash
python src/predict.py --zones IT-NORD IT-CNOR
```

This will:
- Load the best trained model (`models/best_model_pytorch.pt`)
- Make predictions on test data (Oct 27 - Nov 10)
- Calculate metrics (MAE, RMSE, MAPE, R²)
- Generate visualization plots
- Save results to `results/` folder

### 4. View Results

Check the results:
```bash
ls results/
cat results/evaluation_metrics.json
open results/evaluation_summary.png  # macOS
```

## 📊 Expected Outputs

### Training Outputs (`models/`)
- `best_model_pytorch.pt` - Trained model checkpoint
- `normalization_params_pytorch.npz` - Feature normalization stats
- `training_history_pytorch.json` - Training metrics by epoch
- `training_curves_pytorch.png` - Loss and MAE plots

### Evaluation Outputs (`results/`)
- `evaluation_metrics.json` - MAE, RMSE, MAPE, R² by zone
- `evaluation_summary.png` - Bar charts comparing zones
- `IT-NORD_forecast.png` - Individual zone time series
- `IT-CNOR_forecast.png` - ...

## 📈 Performance Targets

Based on literature (Salman et al., 2024; Laaroussi et al., 2022):

- **Target MAE**: < 5% (0.05 capacity factor)
- **Target MAPE**: < 5%
- **Target RMSE**: < 10% (0.10 capacity factor)
- **Target R²**: > 0.90

## 🔧 Troubleshooting

### Out of Memory?
Reduce batch size:
```bash
python src/train.py --batch-size 16
```

### Training too slow?
- Increase batch size: `--batch-size 64`
- Train on fewer zones: `--zones IT-NORD`
- Use sample for testing: `--sample 0.1`

### Import Errors?
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
ACIT4620-exam/
├── data/
│   ├── raw/              # Original data
│   ├── processed/
│   │   ├── train/        # Training data (2015 - Oct 26, historic weather)
│   │   └── test/         # Test data (Oct 27 - Nov 10, weather forecast)
│   └── scripts/          # Data collection scripts
├── src/
│   ├── data_loader.py    # Data loading utilities
│   ├── model.py          # PyTorch CNN-LSTM architecture
│   ├── train.py          # PyTorch training pipeline
│   ├── predict.py        # Prediction & evaluation
│   └── features.py       # Feature definitions
├── models/               # Saved PyTorch models
├── results/              # Evaluation results
├── PYTORCH_GUIDE.md      # Detailed PyTorch guide
├── QUICKSTART.md         # This file
└── requirements.txt      # PyTorch dependencies
```

## 📝 Training Configuration

**Model Architecture (SimpleCNNLSTM):**
- CNN: 1 layer (64 filters, kernel size 3)
- LSTM: 1 layer (64 units)
- Dense: 2 layers (128 units) + output (336 units)
- Dropout: 0.2 for regularization
- Batch normalization after CNN
- **Total parameters**: 87,824

**Training Setup:**
- Optimizer: NAdam (learning rate 0.001)
- Loss: Mean Squared Error (MSE)
- Metrics: MAE (Mean Absolute Error)
- Validation split: 20%
- Early stopping: Patience 10 epochs
- Device: MPS (Apple Silicon GPU) or CPU

**Data:**
- Input: 7 days (168 hours) × 14 features
- Output: 14 days (336 hours) capacity factor predictions
- Training samples: ~89,000 sequences per zone
- **Features (14)**: 13 weather params + hour (NO capacity_factor!)
  - Weather: radiation, temperature, clouds, wind, humidity, pressure
  - Time: hour of day

## 🎓 Literature References

1. **Salman et al. (2024)**: Hybrid deep learning models for time series forecasting of solar power
   - CNN-LSTM-Transformer achieved MAE < 1%
   - NAdam optimizer best performance

2. **Laaroussi et al. (2022)**: Solar power forecasting using CNN-LSTM hybrid model
   - CNN for local patterns, LSTM for sequences
   - MAE < 5% on real PV plant data

## 🔬 Key Design Decisions

1. **PyTorch with MPS**: Apple Silicon optimized, stable, fast
2. **Data Leakage Fixed**: Removed capacity_factor from input (only weather forecast)
3. **Capacity Factor Normalization**: Accounts for 62% growth in Italian solar capacity
4. **Weather Forecast Testing**: Test set uses forecast (not historic) to simulate reality
5. **7-Day Input Window**: Captures weekly patterns and recent trends
6. **14-Day Output Horizon**: Extends traditional day-ahead to two-week planning
7. **Multi-Zone Training**: Learns patterns across all 7 Italian zones

## 🚦 Next Steps

1. **Run Full Training**: `python src/train.py --epochs 50 --batch-size 32`
2. **Monitor Progress**: Check `models/training_curves_pytorch.png`
3. **Evaluate Model**: `python src/predict.py`
4. **Analyze Results**: Review plots in `results/` folder
5. **Compare with Literature**: MAE < 5%, MAPE < 5%, R² > 0.90
6. **Document Findings**: Write project report

## 💡 Tips

- **Apple Silicon**: PyTorch uses MPS for GPU acceleration automatically
- **Start small**: Use `--sample 0.1` for quick testing
- **Monitor training**: Loss should decrease smoothly
- **Batch size**: 32 is good balance of speed and stability
- **Early stopping**: Training stops if no improvement for 10 epochs

## 📊 Expected Training Results

### Quick Test (10% sample, IT-NORD, 3 epochs)
- Validation MAE: ~0.187 (18.7%)
- Training time: ~1 minute
- Device: MPS (Apple Silicon)

### Full Training (all zones, 50 epochs)
- Expected MAE: 0.03-0.08 (3-8%)
- Expected MAPE: 3-8%
- Expected R²: 0.85-0.95
- Training time: 2-4 hours

## 📧 Documentation

For more details see:
- **[PYTORCH_GUIDE.md](PYTORCH_GUIDE.md)** - Complete PyTorch training guide
- **[README.md](README.md)** - Project overview

## 🚀 Quick Start

### 1. Verify Setup

Check that you have all processed data:
```bash
ls data/processed/train/
ls data/processed/test/
```

You should see 7 zone files in each directory (IT-NORD, IT-CNOR, etc.)

### 2. Train Model

Train on all zones (recommended):
```bash
python src/train.py --epochs 100 --batch-size 32
```

Quick test with fewer epochs:
```bash
python src/train.py --epochs 10 --batch-size 32
```

Train on specific zones only:
```bash
python src/train.py --zones IT-NORD IT-CNOR --epochs 50
```

**Expected training time**: ~30-60 minutes per 100 epochs (depending on hardware)

### 3. Evaluate Model

After training completes:
```bash
python src/predict.py
```

This will:
- Load the best trained model
- Make predictions on test data (Oct 27 - Nov 10)
- Calculate metrics (MAE, RMSE, MAPE, R²)
- Generate visualization plots
- Save results to `results/` folder

### 4. View Results

Check the results:
```bash
ls results/
cat results/evaluation_summary.csv
open results/all_zones_forecast.png  # macOS
```

## 📊 Expected Outputs

### Training Outputs (`models/`)
- `best_model_full.keras` - Trained model weights
- `normalization_params.npz` - Feature normalization stats
- `training_history_full.json` - Training metrics by epoch
- `training_history_full.png` - Training curves plot

### Evaluation Outputs (`results/`)
- `evaluation_summary.csv` - Performance metrics by zone
- `it_nord_forecast.png` - Individual zone predictions
- `it_cnor_forecast.png` - ...
- `all_zones_forecast.png` - Overview of all zones
- `evaluation_metrics.png` - Error metrics comparison

## 📈 Performance Targets

Based on literature (Salman et al., 2024; Laaroussi et al., 2022):

- **Target MAE**: < 5% (0.05 capacity factor)
- **Target MAPE**: < 5%
- **Target RMSE**: < 10% (0.10 capacity factor)

## 🔧 Troubleshooting

### Out of Memory?
Reduce batch size:
```bash
python src/train.py --batch-size 16
```

### Training too slow?
Train on fewer zones first:
```bash
python src/train.py --zones IT-NORD IT-CNOR
```

### Model not converging?
Check training curves in `models/training_history_full.png`

## 📁 Project Structure

```
ACIT4620-exam/
├── data/
│   ├── raw/              # Original data
│   ├── processed/
│   │   ├── train/        # Training data (2015 - Oct 26, historic weather)
│   │   └── test/         # Test data (Oct 27 - Nov 10, weather forecast)
│   └── scripts/          # Data collection scripts
├── src/
│   ├── data_loader.py    # Data loading utilities
│   ├── model.py          # CNN-LSTM architecture
│   ├── train.py          # Training pipeline
│   ├── predict.py        # Prediction & evaluation
│   └── features.py       # Feature engineering (optional)
├── models/               # Saved models (created after training)
├── results/              # Evaluation results (created after prediction)
└── requirements-model.txt
```

## 📝 Training Configuration

**Model Architecture:**
- CNN: 2 layers (64, 128 filters, kernel size 3)
- LSTM: 2 layers (128, 64 units)
- Dense: 2 layers (256, 128 units) + output (336 units)
- Dropout: 0.2 for regularization
- Batch normalization after each CNN/LSTM layer

**Training Setup:**
- Optimizer: Nadam (learning rate 0.001)
- Loss: Mean Squared Error (MSE)
- Metrics: MAE, RMSE
- Validation split: 20%
- Early stopping: Patience 15 epochs
- Learning rate reduction: Factor 0.5, patience 5

**Data:**
- Input: 7 days (168 hours) × 15 features
- Output: 14 days (336 hours) capacity factor predictions
- Training samples: ~94,000 sequences per zone
- Features: 13 weather params + capacity factor + hour

## 🎓 Literature References

1. **Salman et al. (2024)**: Hybrid deep learning models for time series forecasting of solar power
   - CNN-LSTM-Transformer achieved MAE < 1%
   - Nadam optimizer best performance

2. **Laaroussi et al. (2022)**: Solar power forecasting using CNN-LSTM hybrid model
   - CNN for local patterns, LSTM for sequences
   - MAE < 5% on real PV plant data

## 🔬 Key Design Decisions

1. **Capacity Factor Normalization**: Accounts for 62% growth in Italian solar capacity (2012-2023)
2. **Weather Forecast Testing**: Test set uses forecast (not historic) to simulate real forecasting
3. **7-Day Input Window**: Captures weekly patterns and recent trends
4. **14-Day Output Horizon**: Extends traditional day-ahead to two-week planning window
5. **Multi-Zone Training**: Learns patterns across all 7 Italian zones

## 🚦 Next Steps

1. **Run Training**: `python src/train.py --epochs 100`
2. **Monitor Progress**: Check `models/training_history_full.png`
3. **Evaluate Model**: `python src/predict.py`
4. **Analyze Results**: Review plots in `results/` folder
5. **Document Findings**: Compare with literature benchmarks

## 💡 Tips

- Start with simple model for faster testing: `--model-type simple`
- Use fewer epochs for initial testing: `--epochs 20`
- Monitor GPU/CPU usage during training
- Training loss should decrease smoothly (check plot)
- Validation loss should track training loss (no major divergence = good)

## 📧 Questions?

Check:
- `src/README.md` - Detailed module documentation
- `data/processed/README.md` - Data structure documentation
- Training output logs for error messages
