# Strategies to Improve Prediction Accuracy Across All Seven Zones

**Current Performance (IT-NORD, Small Model 242K):**
- MAE: 511 MW (forecast), 414 MW (actual weather)
- R²: 0.487 (forecast), 0.659 (actual weather)
- Bias: -48% (systematic underprediction)

## 🎯 Goal
Improve accuracy in a way that generalizes to all 7 Italian zones (IT-NORD, IT-CNOR, IT-CSUD, IT-SUD, IT-SICI, IT-SARD, IT-CALA).

---

## Strategy 1: Bias Correction (Quick Win) ⚡

### Problem
Model systematically underpredicts by 48%. This is consistent and correctable.

### Solution: Post-Processing Correction
```python
# Simple multiplicative bias correction
predicted_corrected = predicted * 1.48

# Or zone-specific correction factors
BIAS_FACTORS = {
    'IT-NORD': 1.48,
    'IT-CNOR': 1.45,  # Would need to measure
    'IT-CSUD': 1.52,
    # ... etc
}
predicted_corrected = predicted * BIAS_FACTORS[zone]
```

### Implementation Steps
1. **Evaluate all 7 zones** to measure bias per zone
2. **Calculate bias correction factor** for each: `factor = mean(actual) / mean(predicted)`
3. **Apply zone-specific correction** during prediction
4. **Re-evaluate** to confirm improvement

### Expected Impact
- ✅ Reduces MAE from ~511 → ~350 MW (30% improvement)
- ✅ Fixes systematic bias
- ✅ Zero training cost
- ✅ Works immediately

### Implementation
```python
def apply_bias_correction(predictions, actual_train, zone):
    """Calculate and apply bias correction based on training data."""
    # Calculate bias on training data (daytime hours only)
    daytime_mask = actual_train > 0
    if daytime_mask.sum() > 0:
        bias_factor = actual_train[daytime_mask].mean() / predictions[daytime_mask].mean()
        return predictions * bias_factor
    return predictions
```

### Pros
- ✅ Fast to implement
- ✅ No retraining needed
- ✅ Zone-specific tuning
- ✅ Interpretable

### Cons
- ⚠️ Doesn't fix underlying model issue
- ⚠️ May not work if bias changes over time

---

## Strategy 2: Improved Loss Function (Moderate Complexity) 🔧

### Problem
Current MSE loss treats underprediction and overprediction equally. But for solar forecasting:
- Underpredicting peak hours is costly (miss valuable production)
- Overpredicting nighttime is less critical (already zero)

### Solution: Custom Weighted Loss
```python
class AsymmetricLoss(nn.Module):
    """Penalize underprediction more than overprediction."""
    def __init__(self, alpha=2.0):
        super().__init__()
        self.alpha = alpha  # Weight for underprediction
    
    def forward(self, pred, target):
        error = target - pred
        # Underprediction (error > 0): weighted more
        # Overprediction (error < 0): normal weight
        loss = torch.where(
            error > 0,
            self.alpha * error**2,  # 2x penalty for underprediction
            error**2
        )
        return loss.mean()

# Or production-weighted loss
class ProductionWeightedLoss(nn.Module):
    """Weight errors by actual production level."""
    def forward(self, pred, target):
        # Normalize target to [0, 1]
        weights = torch.clamp(target, min=0.1)  # Avoid zero weights
        error = (pred - target) ** 2
        return (error * weights).mean()
```

### Implementation Steps
1. **Modify training script** to use new loss function
2. **Retrain model** for all zones
3. **Evaluate** on test set
4. **Compare** with baseline

### Expected Impact
- ✅ Reduces underprediction bias
- ✅ Better captures peak production
- ✅ More balanced errors
- 🎯 Potentially 10-20% MAE improvement

### Pros
- ✅ Addresses root cause of bias
- ✅ Generalizes across zones
- ✅ Physically motivated

### Cons
- ⚠️ Requires retraining all models
- ⚠️ Need to tune alpha parameter
- ⚠️ More complex

---

## Strategy 3: Better Features (High Impact) 🌟

### Problem
Current features may not capture all solar production factors:
- Cloud dynamics
- Aerosol effects
- Horizon effects
- Seasonal patterns

### Solution A: Enhanced Weather Features
```python
# Add derived features that capture solar physics
def add_enhanced_features(df):
    """Add physics-based solar features."""
    
    # 1. Clear sky index (actual radiation / theoretical max)
    df['clear_sky_index'] = df['shortwave_radiation'] / (df['theoretical_max_radiation'] + 1e-6)
    
    # 2. Cloud impact score
    df['cloud_impact'] = df['cloudcover'] * df['shortwave_radiation'] / 100
    
    # 3. Time-based features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # 4. Weather change rate (captures transitions)
    df['radiation_change'] = df['shortwave_radiation'].diff()
    df['cloud_change'] = df['cloudcover'].diff()
    
    # 5. Visibility/air quality proxy
    df['atmospheric_clarity'] = df['shortwave_radiation'] / (df['direct_radiation'] + 1e-6)
    
    # 6. Temperature effects (panel efficiency)
    df['temp_efficiency_factor'] = 1 - 0.004 * (df['temperature'] - 25)  # -0.4% per °C
    
    return df
```

### Solution B: Lag Features (Temporal Patterns)
```python
def add_lag_features(df, lags=[1, 3, 6, 12, 24]):
    """Add recent history features."""
    for lag in lags:
        df[f'radiation_lag_{lag}h'] = df['shortwave_radiation'].shift(lag)
        df[f'production_lag_{lag}h'] = df['actual'].shift(lag)
        df[f'cloud_lag_{lag}h'] = df['cloudcover'].shift(lag)
    return df
```

### Solution C: Rolling Statistics
```python
def add_rolling_features(df, windows=[3, 6, 12, 24]):
    """Add moving averages and trends."""
    for window in windows:
        df[f'radiation_ma_{window}h'] = df['shortwave_radiation'].rolling(window).mean()
        df[f'radiation_std_{window}h'] = df['shortwave_radiation'].rolling(window).std()
        df[f'cloud_ma_{window}h'] = df['cloudcover'].rolling(window).mean()
    return df
```

### Expected Impact
- ✅ Captures more solar physics
- ✅ Better temporal patterns
- ✅ Improved peak hour predictions
- 🎯 Potentially 15-25% MAE improvement

### Pros
- ✅ Generalizes to all zones
- ✅ Physically motivated
- ✅ No architecture changes needed

### Cons
- ⚠️ Need to regenerate all processed data
- ⚠️ Increases feature count (risk of overfitting)
- ⚠️ Requires domain knowledge

---

## Strategy 4: Separate Day/Night Models (Architecture Change) 🏗️

### Problem
Model wastes capacity learning "night → zero" which is trivial. This dilutes its ability to learn complex daytime patterns.

### Solution: Two-Model Approach
```python
class DayNightModel:
    def __init__(self, n_features):
        # Daytime model: learns solar production
        self.day_model = EncoderDecoderCNNLSTM(168, 336, n_features)
        # Nighttime model: simple rule (always zero)
        self.night_threshold = 1e-3  # radiation threshold
    
    def predict(self, X_enc, X_dec, radiation):
        """Predict with day/night logic."""
        # Get radiation values for test period
        is_daytime = radiation > self.night_threshold
        
        # Predict daytime hours only
        predictions = torch.zeros(len(radiation))
        if is_daytime.sum() > 0:
            day_pred = self.day_model(X_enc, X_dec)
            predictions[is_daytime] = day_pred[is_daytime]
        
        return predictions
```

### Implementation Steps
1. **Filter training data** to daytime hours only (radiation > threshold)
2. **Train model** on daytime-only data
3. **Predict** with simple rule: `if radiation < threshold: return 0, else: return model(X)`
4. **Evaluate** improvement

### Expected Impact
- ✅ Model focuses 100% capacity on hard problem (daytime)
- ✅ Reduces parameter waste on trivial patterns
- ✅ Better daytime predictions
- 🎯 Potentially 20-30% improvement on daytime hours

### Pros
- ✅ More efficient use of model capacity
- ✅ Cleaner training signal
- ✅ Generalizes across zones

### Cons
- ⚠️ Need to modify training pipeline
- ⚠️ More complex prediction logic
- ⚠️ Requires threshold tuning per zone

---

## Strategy 5: Ensemble Methods (Robust Improvement) 🎭

### Problem
Single model has high variance. Different random seeds give different results.

### Solution: Ensemble of Small Models
```python
class EnsembleModel:
    def __init__(self, n_models=5):
        """Train multiple models with different seeds."""
        self.models = []
        for seed in range(n_models):
            torch.manual_seed(seed)
            model = EncoderDecoderCNNLSTM(168, 336, n_features)
            # Train with different initialization
            self.models.append(train_model(model, seed=seed))
    
    def predict(self, X_enc, X_dec):
        """Average predictions from all models."""
        predictions = []
        for model in self.models:
            pred = model(X_enc, X_dec)
            predictions.append(pred)
        
        # Average (or median for robustness)
        return torch.stack(predictions).mean(dim=0)
        # return torch.stack(predictions).median(dim=0)[0]  # More robust
```

### Implementation Steps
1. **Train 5 models** with different random seeds
2. **Save all models** (5x storage but worth it)
3. **Ensemble prediction**: average all outputs
4. **Evaluate** ensemble vs single model

### Expected Impact
- ✅ Reduces prediction variance
- ✅ More stable/reliable predictions
- ✅ Better generalization
- 🎯 Typically 5-15% MAE improvement

### Pros
- ✅ Proven technique
- ✅ No architecture changes
- ✅ More robust predictions
- ✅ Uncertainty quantification (std of predictions)

### Cons
- ⚠️ 5x training time
- ⚠️ 5x inference time (can parallelize)
- ⚠️ 5x model storage

---

## Strategy 6: Multi-Task Learning (Advanced) 🧠

### Problem
Model only predicts production. Doesn't learn intermediate representations.

### Solution: Predict Multiple Targets
```python
class MultiTaskModel(nn.Module):
    """Predict production AND intermediate targets."""
    def __init__(self, enc_len, dec_len, n_features):
        super().__init__()
        self.encoder = Encoder(enc_len, n_features)
        self.decoder = Decoder(dec_len, n_features)
        
        # Multiple output heads
        self.production_head = nn.Linear(128, 1)
        self.radiation_head = nn.Linear(128, 1)  # Auxiliary task
        self.cloud_head = nn.Linear(128, 1)      # Auxiliary task
    
    def forward(self, X_enc, X_dec):
        enc_out = self.encoder(X_enc)
        dec_out = self.decoder(X_dec, enc_out)
        
        # Predict all targets
        production = self.production_head(dec_out)
        radiation = self.radiation_head(dec_out)
        clouds = self.cloud_head(dec_out)
        
        return production, radiation, clouds

# Multi-task loss
loss = mse(prod_pred, prod_true) + \
       0.5 * mse(rad_pred, rad_true) + \
       0.5 * mse(cloud_pred, cloud_true)
```

### Expected Impact
- ✅ Better learned representations
- ✅ More robust features
- ✅ Improved generalization
- 🎯 Potentially 10-20% improvement

### Pros
- ✅ Learns better internal representations
- ✅ Regularization effect
- ✅ Can predict uncertainty

### Cons
- ⚠️ More complex training
- ⚠️ Requires multiple labels
- ⚠️ Harder to tune

---

## Strategy 7: Transfer Learning Across Zones 🔄

### Problem
Training each zone independently. Not leveraging similarities between zones.

### Solution: Shared Base + Zone-Specific Heads
```python
class MultiZoneModel(nn.Module):
    """Shared encoder, zone-specific decoders."""
    def __init__(self, zones, n_features):
        super().__init__()
        # Shared encoder learns general solar patterns
        self.shared_encoder = Encoder(168, n_features)
        
        # Zone-specific decoders
        self.zone_decoders = nn.ModuleDict({
            zone: Decoder(336, 128) for zone in zones
        })
        
        # Zone-specific output heads
        self.zone_heads = nn.ModuleDict({
            zone: nn.Linear(128, 1) for zone in zones
        })
    
    def forward(self, X_enc, X_dec, zone):
        # Shared encoding
        enc_out = self.shared_encoder(X_enc)
        
        # Zone-specific decoding
        dec_out = self.zone_decoders[zone](X_dec, enc_out)
        production = self.zone_heads[zone](dec_out)
        
        return production
```

### Training Strategy
```python
# Phase 1: Pre-train shared encoder on all zones
for epoch in range(50):
    for zone in zones:
        X, y = load_zone_data(zone)
        loss = train_step(model, X, y, zone)

# Phase 2: Fine-tune zone-specific parts
for zone in zones:
    freeze_encoder(model)
    finetune_decoder(model, zone, epochs=20)
```

### Expected Impact
- ✅ Better generalization across zones
- ✅ Faster training for new zones
- ✅ Shared knowledge improves all zones
- 🎯 Potentially 15-25% improvement for smaller zones

### Pros
- ✅ Leverages all available data
- ✅ Better for zones with less data
- ✅ Single model for all zones

### Cons
- ⚠️ More complex training pipeline
- ⚠️ Harder to debug
- ⚠️ Risk of negative transfer

---

## Recommended Implementation Order 🎯

### Phase 1: Quick Wins (Week 1)
1. **Bias Correction** (Strategy 1)
   - Evaluate all 7 zones
   - Calculate bias factors
   - Apply correction
   - **Expected: 30% improvement, zero cost**

2. **Ensemble of Current Models** (Strategy 5)
   - Train 5 models with different seeds per zone
   - Average predictions
   - **Expected: +10% improvement**

### Phase 2: Feature Engineering (Week 2-3)
3. **Enhanced Features** (Strategy 3)
   - Add physics-based features
   - Add temporal features
   - Retrain all zones
   - **Expected: +15-20% improvement**

### Phase 3: Architecture Improvements (Week 4-5)
4. **Day/Night Separation** (Strategy 4)
   - Train daytime-only models
   - Apply simple nighttime rule
   - **Expected: +20% improvement on daytime**

5. **Improved Loss Function** (Strategy 2)
   - Implement asymmetric loss
   - Retrain with production weighting
   - **Expected: +10-15% improvement**

### Phase 4: Advanced Techniques (Week 6+)
6. **Multi-Zone Transfer Learning** (Strategy 7)
   - Implement shared architecture
   - Train on all zones simultaneously
   - **Expected: +15-20% improvement**

7. **Multi-Task Learning** (Strategy 6)
   - Add auxiliary prediction tasks
   - More robust representations
   - **Expected: +10-15% improvement**

---

## Expected Cumulative Impact

### Current (IT-NORD)
- MAE: 511 MW
- R²: 0.487
- Bias: -48%

### After Quick Wins (Phase 1)
- MAE: ~350 MW ✅ (31% better)
- R²: ~0.55
- Bias: ~0%

### After Feature Engineering (Phase 2)
- MAE: ~280 MW ✅ (45% better)
- R²: ~0.65
- Bias: <10%

### After Architecture Improvements (Phase 3)
- MAE: ~230 MW ✅ (55% better)
- R²: ~0.72
- Bias: <5%

### After Advanced Techniques (Phase 4)
- MAE: ~200 MW ✅ (60% better)
- R²: ~0.78
- Bias: <5%

---

## Code Example: Complete Improvement Pipeline

```python
# File: src/improved_pipeline.py

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

class ImprovedSolarPredictor:
    """Production-ready solar forecasting system."""
    
    def __init__(self, zone, use_ensemble=True, use_bias_correction=True):
        self.zone = zone
        self.use_ensemble = use_ensemble
        self.use_bias_correction = use_bias_correction
        
        # Load models
        if use_ensemble:
            self.models = self.load_ensemble(zone)
        else:
            self.models = [self.load_single_model(zone)]
        
        # Load bias correction factors
        if use_bias_correction:
            self.bias_factor = self.load_bias_factor(zone)
        else:
            self.bias_factor = 1.0
    
    def preprocess_features(self, df):
        """Add enhanced features."""
        # Physics-based features
        df['clear_sky_index'] = df['shortwave_radiation'] / (df['theoretical_max'] + 1e-6)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['temp_efficiency'] = 1 - 0.004 * (df['temperature'] - 25)
        
        # Temporal features
        for lag in [1, 3, 6, 12, 24]:
            df[f'radiation_lag_{lag}'] = df['shortwave_radiation'].shift(lag)
        
        return df
    
    def predict(self, X_enc, X_dec, radiation):
        """Generate predictions with all improvements."""
        # Ensemble prediction
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(X_enc, X_dec).cpu().numpy()
            predictions.append(pred)
        
        # Average ensemble
        pred = np.mean(predictions, axis=0)
        
        # Apply day/night logic
        is_nighttime = radiation < 1e-3
        pred[is_nighttime] = 0.0
        
        # Apply bias correction
        pred = pred * self.bias_factor
        
        # Clip to valid range
        pred = np.clip(pred, 0, self.installed_capacity)
        
        return pred
    
    def predict_with_uncertainty(self, X_enc, X_dec):
        """Get prediction + uncertainty estimate."""
        predictions = []
        for model in self.models:
            pred = model(X_enc, X_dec)
            predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        return mean, std  # Return prediction + confidence interval

# Usage
predictor = ImprovedSolarPredictor(
    zone='IT-NORD',
    use_ensemble=True,
    use_bias_correction=True
)

pred_mean, pred_std = predictor.predict_with_uncertainty(X_enc, X_dec)
print(f"Prediction: {pred_mean:.2f} ± {1.96*pred_std:.2f} MW (95% CI)")
```

---

## Evaluation Protocol

### Testing Improvements
For each improvement, evaluate on ALL 7 zones:

```bash
# Evaluate all zones
for zone in IT-NORD IT-CNOR IT-CSUD IT-SUD IT-SICI IT-SARD IT-CALA; do
    python src/evaluate_forecast.py --zones $zone
done

# Compare results
python src/compare_improvements.py --baseline vs --improved
```

### Success Criteria
- ✅ Average MAE across 7 zones improves by >20%
- ✅ R² across 7 zones improves by >0.1
- ✅ No zone gets worse (check for regressions)
- ✅ Bias reduced to <10% on average
- ✅ Improvement is consistent (not just lucky on IT-NORD)

---

## Conclusion

**Best Strategy for All 7 Zones:**

1. **Start with Bias Correction** (immediate 30% improvement)
2. **Add Ensemble** (another 10% improvement, no code changes)
3. **Improve Features** (15-20% improvement, generalizes well)
4. **Use Day/Night Split** (20% improvement on daytime)
5. **Consider Transfer Learning** (helps smaller zones)

**Expected Final Result:**
- MAE: 200-250 MW (vs current 511 MW)
- R²: 0.75-0.80 (vs current 0.487)
- Bias: <5% (vs current -48%)
- **Works consistently across all 7 Italian zones**

The key is that all these improvements are **zone-agnostic** - they improve the fundamental approach rather than overfitting to specific zones. This ensures they generalize to all regions.
