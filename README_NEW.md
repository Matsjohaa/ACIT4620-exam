# Solar Energy Forecasting with Encoder-Decoder CNN-LSTM

Deep learning model for 14-day solar power forecasting using encoder-decoder architecture with residual learning.

## 🚀 Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Evaluate trained model
python src/evaluate_forecast.py

# Train new model
python src/train.py --zones IT-NORD --epochs 25 --model-type encoder --residual
```

## 📁 Project Structure (Zone-Based)

```
├── models/
│   └── <zone>/              # Zone-specific models (e.g., it-nord)
│       ├── model.pt         # Trained model checkpoint
│       ├── norm.npz         # Normalization parameters
│       ├── training_history.json
│       └── training_curves.png
│
├── results/
│   └── <zone>/              # Zone-specific results (e.g., it-nord)
│       └── forecast.png     # 14-day forecast visualization
│
├── src/
│   ├── model.py             # Neural network architectures
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── train.py             # Training script
│   └── evaluate_forecast.py # Evaluation and prediction
│
└── data/
    ├── energy/              # Energy production data
    └── weather/             # Weather forecast data
```

## 📊 Model Performance (IT-NORD)

**Test Period:** October 27 - November 10, 2025 (336 hours)

| Metric | Value |
|--------|-------|
| MAE | 120.12 MW |
| RMSE | 243.73 MW |
| R² | 0.970 |

**Improvement:** 36.4% better than baseline model

## 🏗️ Architecture

**Encoder-Decoder CNN-LSTM with Residual Learning**

- **Input:** 168 hours (7 days) of weather history + 336 hours (14 days) future weather
- **Output:** 336-hour solar power forecast
- **Features:** 15 total (13 weather + 2 time-based)
- **Parameters:** 240,225

**Components:**
- Encoder: 2 CNN layers + LSTM (128 hidden units)
- Decoder: 3 dense layers (256→128→64→1)
- Batch normalization + LeakyReLU + Dropout (0.15)
- Residual learning (predicts deviation from day-ahead forecast)

## 🎯 Key Features

✅ **Zone-based organization** - Models and results organized by zone  
✅ **Residual learning** - Learns to correct day-ahead forecasts  
✅ **Nighttime zeroing** - Physically valid predictions (no solar at night)  
✅ **Apple Silicon optimized** - Fast training on M1/M2/M3 with MPS  
✅ **Clean file naming** - Simple, intuitive file structure  

## 🔧 Usage

### Training

```bash
# Train on specific zone
python src/train.py \
  --zones IT-NORD \
  --epochs 25 \
  --model-type encoder \
  --residual \
  --batch-size 32 \
  --lr 0.001
```

**Output:** `models/it-nord/model.pt` and `models/it-nord/norm.npz`

### Evaluation

```bash
python src/evaluate_forecast.py
```

**Output:**
- Console: MAE, RMSE, R² metrics
- File: `results/it-nord/forecast.png`

### Training on Multiple Zones

```bash
# Multi-zone training (future)
python src/train.py --zones IT-NORD IT-CSUD --model-type encoder --residual
# → Saves to: models/multi-zone/
```

## 📈 Training Details

- **Device:** Apple Silicon MPS (Metal Performance Shaders)
- **Training time:** ~45 minutes for 25 epochs on M1/M2
- **Data:** 94,004 sequences (75,204 train / 18,800 validation)
- **Loss:** Variation-Aware Loss (MSE + variance penalty)
- **Optimizer:** NAdam with learning rate scheduling

## 🌙 Post-Processing

Automatic nighttime zeroing ensures physically valid predictions:
- Zero when hour < 6 or > 18
- Zero when solar radiation ≤ 1e-3 W/m²
- Affects ~57% of predictions in test period

## 📚 Documentation

- **PRODUCTION_GUIDE.md** - Complete usage guide
- **MODEL_IMPROVEMENT_SUMMARY.md** - Technical details and improvement history
- **FINAL_SUMMARY.txt** - Quick reference

## 🔬 Input Features (15 total)

**Weather (13):**
- temperature_2m, relative_humidity_2m, dew_point_2m
- wind_speed_10m, wind_direction_10m
- cloud_cover, shortwave_radiation, direct_radiation, diffuse_radiation
- precipitation, snowfall, surface_pressure, vapour_pressure_deficit

**Engineered (2):**
- hour_sin, hour_cos (cyclical time encoding)

## 🎓 Model Evolution

| Version | Description | MAE | R² |
|---------|-------------|-----|-----|
| v1.0 | Simple CNN-LSTM | 188.84 MW | 0.929 |
| v1.1 | Old encoder (buggy) | 386.73 MW | 0.678 |
| **v2.0** | **New encoder (production)** | **120.12 MW** | **0.970** |

**Key Improvements:**
- Deeper decoder (3 layers vs 2)
- Batch normalization
- LeakyReLU activation
- Larger encoder hidden state (128 vs 64)

## ⚙️ Requirements

- Python 3.9+
- PyTorch 2.9.1
- Apple Silicon Mac (or CUDA GPU)
- See `.venv` for complete dependencies

## 🚦 Troubleshooting

**Model not found:**
```bash
ls -lh models/it-nord/model.pt  # Check model exists
```

**Poor performance:**
- Ensure `--residual` flag is used during training
- Check normalization parameters are loaded correctly
- Verify nighttime zeroing is applied

**Training slow:**
- Check MPS is available: `python -c "import torch; print(torch.backends.mps.is_available())"`
- Reduce batch size if memory issues

---

**Last Updated:** November 18, 2025  
**Model Version:** 2.0.0 (Zone-based)  
**Zone:** IT-NORD
