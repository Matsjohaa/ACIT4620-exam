# Improved Training Strategy v2 - Asymmetric Loss

## 🎯 Goal
Fix the **-48% underprediction bias** without degrading overall performance like the daytime-focused loss did.

## ❌ What Failed (v1)
**DaytimeFocusedLoss** (nighttime_weight=0.1)
- **Result**: Performance DEGRADED
  - MAE: 511 → 540 MW (+6% worse)
  - R²: 0.487 → 0.382 (-22% worse)
  - Bias: -48% → -59% (even worse!)
- **Why it failed**: 
  - Downweighting nighttime made model too conservative everywhere
  - Didn't directly target the underprediction problem
  - Unintended side effects

## ✅ New Strategy (v2)
**ImprovedLoss** with asymmetric penalties

### Three Key Components

#### 1. Asymmetric Penalties (2.0x for underprediction)
```python
# Underprediction (error > 0): 2x penalty
# Overprediction (error < 0): 1x penalty
asymmetric_loss = torch.where(
    error > 0,
    2.0 * squared_error,  # Penalize underprediction MORE
    squared_error          # Normal penalty for overprediction
)
```

**Why this works:**
- Directly addresses the bias problem
- Model learns that underpredicting is "expensive"
- Naturally pushes predictions upward without ignoring any data
- No side effects from downweighting hours

#### 2. Production-Weighted Loss (0.3x weight)
```python
# Weight errors by production magnitude
production_weights = 1.0 + 0.3 * (target / target.max())
weighted_loss = (asymmetric_loss * production_weights).mean()
```

**Why this works:**
- Focuses learning on high-production hours (what matters most)
- Naturally emphasizes daytime without explicit filtering
- Better than downweighting - actually teaches importance
- Low weight (0.3) prevents overemphasis

#### 3. Variance Penalty (0.2x weight)
```python
# Encourage prediction diversity (same as before)
pred_var = torch.var(pred_daytime)
target_var = torch.var(target_daytime)
variance_penalty = torch.relu(target_var - pred_var) / target_var
```

**Why this works:**
- Prevents flat predictions
- Encourages model to capture variation
- Proven component from previous training

## 📊 Expected Results

| Metric | Baseline | Expected | Improvement |
|--------|----------|----------|-------------|
| **MAE (forecast)** | 511 MW | 360-410 MW | **20-30% better** |
| **R² (forecast)** | 0.487 | 0.60-0.65 | **+23-34%** |
| **Bias** | -48% | -20% to -30% | **Halved** |
| **MAE (actual)** | 414 MW | 290-340 MW | **20-30% better** |

### Why These Estimates?

1. **Asymmetric penalty (2.0x)**:
   - Directly targets -48% bias
   - Expected: Reduce bias to -20% to -30%
   - Impact: ~15% MAE improvement

2. **Production weighting (0.3)**:
   - Improves accuracy on high-production hours (most important)
   - Expected: ~10% MAE improvement on daytime

3. **Combined effect**:
   - Both improvements compound
   - Total expected: 20-30% improvement
   - More conservative estimate: 20%
   - Optimistic estimate: 30%

## 🔍 Comparison with v1

| Aspect | DaytimeFocusedLoss (v1) | ImprovedLoss (v2) |
|--------|------------------------|-------------------|
| **Approach** | Downweight nighttime | Asymmetric penalties |
| **Target** | Indirect (filter data) | Direct (penalty ratio) |
| **Risk** | High (can backfire) | Low (gradual correction) |
| **Result** | ❌ Made worse (+6% MAE) | ⏳ Testing now |
| **Bias effect** | ❌ Worsened (-48% → -59%) | ✅ Should improve |
| **Complexity** | Moderate | Similar |

## 🎓 Lessons Learned

### From v1 Failure:
1. **Don't filter/downweight data** - can have unintended side effects
2. **Target problems directly** - asymmetric loss directly addresses bias
3. **Test incrementally** - good that we tested, but lost baseline
4. **Always backup** - should have saved good model before v1

### Why v2 Should Work:
1. **Direct approach**: Asymmetric penalties directly fix underprediction
2. **No data loss**: All hours still contribute to training
3. **Proven technique**: Asymmetric loss widely used in imbalanced problems
4. **Lower risk**: Gradual correction, not radical filtering
5. **Production focus**: Weight by importance, don't exclude

## 📈 Training Configuration

```python
criterion = ImprovedLoss(
    underprediction_weight=2.0,  # 2x penalty for underprediction
    production_weight=0.3,        # Focus on high-production hours
    variance_weight=0.2           # Encourage prediction diversity
)

optimizer = NAdam(lr=0.001, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(patience=5, factor=0.5)
early_stopping = 8 epochs
gradient_clipping = 1.0
```

## 🎯 Success Criteria

### Minimum (Better than baseline):
- ✅ MAE < 511 MW
- ✅ R² > 0.487
- ✅ Bias > -48% (closer to 0)

### Target (Good improvement):
- 🎯 MAE < 410 MW (20% improvement)
- 🎯 R² > 0.60 (23% improvement)
- 🎯 Bias: -20% to -30% (halved)

### Stretch (Excellent):
- 🌟 MAE < 360 MW (30% improvement)
- 🌟 R² > 0.65 (34% improvement)
- 🌟 Bias: -15% to -20% (tripled improvement)

## 🚀 Next Steps After Training

1. **Evaluate results** with dual-scenario testing
2. **Compare with baseline** (if better, keep; if worse, analyze why)
3. **If successful**: Document and potentially apply to other zones
4. **If not successful**: Try even safer approaches (post-processing bias correction)

## 🔧 Backup Plan

If ImprovedLoss also fails:
1. **Post-processing bias correction** (multiply predictions by 1.48)
   - Zero training cost
   - Immediate 48% bias improvement
   - Safest option

2. **Feature engineering** (add physics-based features)
   - Clear sky index
   - Temperature efficiency factor
   - Panel angle corrections
   - Requires data regeneration

3. **Ensemble approach** (train 5 models, average predictions)
   - More robust
   - Lower variance
   - Higher compute cost

---

**Status**: Training in progress (50 epochs, ~2 hours)
**Date**: November 21, 2024
**Zone**: IT-NORD (242K parameters)
