"""
Example: How to use the production forecasting script.

This demonstrates forecasting WITHOUT day-ahead power forecast data.
"""
import pandas as pd
from pathlib import Path
from src.data_loader import load_zone_data

# Create example production forecast using real weather data
zone = "IT-NORD"

print("=" * 80)
print("PRODUCTION FORECASTING EXAMPLE (No Day-Ahead Power Forecast)")
print("=" * 80)
print()
print("Scenario: You want to forecast solar production using ONLY weather forecasts.")
print("You do NOT have access to grid operator day-ahead power forecasts.")
print()

# Load real data as example
print("Step 1: Preparing example data...")
train_df = load_zone_data(zone, split="train")
test_df = load_zone_data(zone, split="test")

# Create historical data (last 168 hours before forecast period)
historical = train_df.tail(168).copy()
historical_file = "example_data/historical_168h.csv"
Path("example_data").mkdir(exist_ok=True)
historical.to_csv(historical_file, index=False)
print(f"  ✅ Created {historical_file}")
print(f"     Contains: {len(historical)} hours of historical weather + production")

# Create forecast data (336 hours of weather forecast)
forecast = test_df.copy()

# IMPORTANT: Remove 'actual' and 'day-ahead' columns - these wouldn't be available in production!
columns_to_remove = ['actual', 'day-ahead', 'capacity_factor']
forecast_clean = forecast.drop(columns=[c for c in columns_to_remove if c in forecast.columns])

forecast_file = "example_data/weather_forecast_336h.csv"
forecast_clean.to_csv(forecast_file, index=False)
print(f"  ✅ Created {forecast_file}")
print(f"     Contains: {len(forecast_clean)} hours of WEATHER FORECAST ONLY")
print(f"     Features: {', '.join(forecast_clean.columns.tolist()[:5])}...")
print()

print("=" * 80)
print("Step 2: Run production forecast (3 different baseline methods)")
print("=" * 80)
print()

print("Method 1: Zero baseline (direct prediction)")
print("-" * 80)
print("Command:")
print(f"  python src/forecast_production.py \\")
print(f"    --zone {zone} \\")
print(f"    --historical {historical_file} \\")
print(f"    --forecast {forecast_file} \\")
print(f"    --baseline zero \\")
print(f"    --output results/{zone.lower()}/production_forecast_zero.csv")
print()
print("How it works:")
print("  - Model predicts residual")
print("  - Since we have no day-ahead, we use baseline = 0")
print("  - Final = 0 + residual = residual (direct prediction)")
print("  - Best for models trained heavily on residuals")
print()

print("Method 2: Simple solar potential baseline")
print("-" * 80)
print("Command:")
print(f"  python src/forecast_production.py \\")
print(f"    --zone {zone} \\")
print(f"    --historical {historical_file} \\")
print(f"    --forecast {forecast_file} \\")
print(f"    --baseline simple \\")
print(f"    --output results/{zone.lower()}/production_forecast_simple.csv")
print()
print("How it works:")
print("  - Creates simple baseline from radiation * (1 - cloudcover/100)")
print("  - Model adds residual correction on top")
print("  - Final = simple_baseline + residual")
print("  - More conservative, uses physical intuition")
print()

print("Method 3: Persistence baseline")
print("-" * 80)
print("Command:")
print(f"  python src/forecast_production.py \\")
print(f"    --zone {zone} \\")
print(f"    --historical {historical_file} \\")
print(f"    --forecast {forecast_file} \\")
print(f"    --baseline persistence \\")
print(f"    --output results/{zone.lower()}/production_forecast_persistence.csv")
print()
print("How it works:")
print("  - Uses last 336 hours production pattern as baseline")
print("  - Model adds residual to account for weather changes")
print("  - Final = last_week_pattern + residual")
print("  - Good for stable weather patterns")
print()

print("=" * 80)
print("Step 3: Compare methods and choose best for your use case")
print("=" * 80)
print()
print("Recommendation:")
print("  - For IT-NORD (good weather forecasts): Use 'zero' or 'simple'")
print("  - For IT-SARD (poor weather forecasts): Use 'simple' or 'persistence'")
print("  - Monitor accuracy and adjust based on real results")
print()

print("=" * 80)
print("READY TO TEST!")
print("=" * 80)
print()
print("Run this command to test:")
print(f"  python src/forecast_production.py \\")
print(f"    --zone {zone} \\")
print(f"    --historical {historical_file} \\")
print(f"    --forecast {forecast_file} \\")
print(f"    --baseline zero")
print()
