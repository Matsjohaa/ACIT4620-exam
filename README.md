# ACIT4620 Solar Forecasting

Deep learning pipeline for forecasting hourly solar power production across the Italian bidding zones. The repository contains training, evaluation, and visualization utilities implemented in PyTorch.

## Environment Setup
1. Use Python 3.10+ with virtual environments to avoid dependency conflicts:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```


## Train a Model
Run `src/train.py` to train a zone-specific attention-based CNN-LSTM model. Provide at least one zone in ENTSO-E format (defaults to all zones when omitted):

```bash
python src/train.py --zones IT-NORD --epochs 25 --batch-size 32 --dropout 0.25
```
**Note:** The training loop was tuned on Apple Silicon (MPS backend). On Windows/Linux machines use CUDA (when available) or CPU-only mode, and consider enabling `--sample` or reducing the forecast horizon if you run into memory issues.

Useful flags:

- `--zones IT-NORD IT-SUD` — train multiple zones sequentially.
- `--sample 0.2` — randomly subsample 20% of the training rows (quick smoke-test).
- `--lr 0.0005` — override learning rate.

Each run creates `models/<zone>/` containing `model.pt`, normalization stats (`norm.npz`), and `training_curves.png`.

## Evaluate Forecasts
Use `src/evaluate_forecast.py` after training to produce metrics, CSV exports, and plots:

```bash
python src/evaluate_forecast.py --zones IT-NORD IT-SICI
```

Options:
- No arguments evaluates all zones listed in `ZONES`.
- `--zones ...` restricts the evaluation set.
- `--actual-weather` compares model outputs when fed actual observed weather instead of forecasts.

Artifacts are written to `results/<zone>/`, including:
- `predictions_forecast.png` / `predictions_actual.png` — line plots for 14-day horizons.
- `predictions_comparison.csv` — hour-level actual vs. predicted MW with absolute/percent errors.
- `results/all_zones_summary.csv` — aggregate MAE/RMSE/R² for every zone/scenario.

## Visual Diagnostics (Optional)
Create a grid of scatter plots across all zones once evaluation CSV files exist:

```bash
python src/plot_scatter_grid.py
```

Scatter plots are saved to `results/scatter_grid.png`.

## Project Structure
```
ACIT4620-exam/
├── README.md
├── requirements.txt
├── src/
│   ├── train.py                # Training entry point
│   ├── evaluate_forecast.py    # Evaluation/plotting entry point
│   ├── data_loader.py          # Data ingestion and normalization helpers
│   ├── model_attention.py      # Encoder-decoder architecture
│   └── plot_scatter_grid.py    # Optional diagnostic plots
├── data/
│   ├── processed/
│   │   ├── train/
│   │   ├── test_forecast/
│   │   └── test_actual_weather/
│   └── raw/
├── models/                     # Saved weights + normalization per zone
└── results/                    # Evaluation CSVs/plots per zone
```
