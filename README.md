# ACIT4620 Solar Forecasting

This project provides a deep learning pipeline for forecasting hourly solar power production in the Italian bidding zones using weather and energy data.

## How to Train
Run the following command to train the model for a specific zone (e.g., IT-NORD):

```
python src/train.py --zones IT-NORD --dropout 0.25 --epochs 25
```

## How to Evaluate
To evaluate the trained model and generate plots and metrics:

```
python src/evaluate.py --zones IT-SICI
```

## Project Structure
```
ACIT4620-exam/
├── README.md                # Project overview and instructions
├── update_date.py           # Utility script
├── src/
│   ├── train.py             # Model training script
│   ├── evaluate.py          # Model evaluation script
│   ├── data_loader.py       # Data loading and preprocessing
│   └── ...
├── plot_scatter_grid.py     # Generates grid of error scatter plots
├── data/
│   └── processed/           # Processed weather and energy data
│       ├── train/           # Training data per zone
│       └── test_forecast/   # Test data per zone (forecast weather)
│       └── test_actual_weather/ # Test data per zone (actual weather)
├── results/
│   └── [zone]/              # Model outputs and plots per zone
└── models/
    └── [zone]/              # Trained model weights per zone
```
