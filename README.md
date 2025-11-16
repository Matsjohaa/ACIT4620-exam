# ACIT4620 Exam - 14-Day Solar Forecasting

Deep learning project for forecasting solar production across 7 Italian bidding zones using CNN-LSTM hybrid models with PyTorch.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Quick test (10% sample, 1 zone, ~1 minute)
python src/train.py --zones IT-NORD --epochs 5 --batch-size 16 --sample 0.1

# Full training (all zones, 2-4 hours)
python src/train.py --epochs 50 --batch-size 32

# Model evaluation (simple vs encoder, residual mode)
python src/evaluate_compare_models.py

# Evaluate on test period (Oct 27 - Nov 10, 2025)
python src/predict.py
```

## Documentation

- 📘 **[PYTORCH_GUIDE.md](PYTORCH_GUIDE.md)** - Complete training guide
- 🚀 **[QUICKSTART.md](QUICKSTART.md)** - Quick start instructions

## Key Features

✅ **PyTorch with Apple Silicon (MPS) support**  
✅ **14-day solar production forecast** (Oct 27 - Nov 10, 2025)  
✅ **7 Italian bidding zones**  
✅ **Training: Historic weather** (ERA5, 5-day lag)  
✅ **Testing: Weather forecast** (14-day prediction)  
✅ **Data leakage fixed** (14 features: weather + hour only)
✅ **Optional residual learning** (capacity factor – day-ahead forecast)

## Project Structure

```
├── src/
│   ├── train.py                          # PyTorch training
│   ├── predict.py                        # Model evaluation
│   ├── model.py                          # CNN-LSTM architecture
    ├── evaluate_compare_models.py        # Compare simple vs encoder vs day-ahead
│   └── data_loader.py                    # Data utilities
├── data/
│   ├── processed/train/  # Training data (2015 - Oct 26)
│   └── processed/test/   # Test data (Oct 27 - Nov 10)
├── models/               # Saved models
└── results/              # Evaluation results
```

## Technology Stack

- **PyTorch 2.9.0** - Deep learning framework
- **ENTSO-E API** - Solar production data
- **Open-Meteo API** - Weather data (historic & forecast)
- **Python 3.13** - Programming language

## Model Architecture

**SimpleCNNLSTM** (87,824 parameters):
- Conv1D layers for spatial feature extraction
- LSTM layers for temporal dependencies
- Dense layers for 14-day forecast output

**EncoderDecoderCNNLSTM** (extended model):
- Encoder: CNN + LSTM over past 168 hours of weather
- Decoder: Forecasted weather for the next 336 hours
- Supports residual learning (target = CF − day-ahead CF)
- Allows weather-aware 14-day forecasting


## Expected Results

| Metric | Target | Expected |
|--------|--------|----------|
| MAE    | < 5%   | 3-8%     |
| MAPE   | < 5%   | 3-8%     |
| R²     | > 0.90 | 0.85-0.95|

## License

Academic project for ACIT4620 course.
