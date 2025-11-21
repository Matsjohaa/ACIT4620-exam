"""
Test the newly trained direct prediction model for IT-NORD
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from src.model import EncoderDecoderCNNLSTM
from src.data_loader import prepare_sequences_with_future

def load_model(zone: str):
    """Load the trained model"""
    model_path = Path(f'models/{zone}/model.pt')
    norm_path = Path(f'models/{zone}/norm.npz')
    
    # Load normalization params
    norm_data = np.load(norm_path)
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Get model info
    n_features = checkpoint['n_features']
    use_residual = checkpoint.get('use_residual', False)
    
    # Create model
    model = EncoderDecoderCNNLSTM(n_features=n_features, use_residual=use_residual)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model for {zone}")
    print(f"  Features: {n_features}")
    print(f"  Trained with residual: {use_residual}")
    
    return model, norm_data, use_residual

def evaluate_direct_model(zone='IT-NORD'):
    """Evaluate the direct prediction model on test data"""
    print(f"\n{'='*80}")
    print(f"TESTING DIRECT PREDICTION MODEL FOR {zone}")
    print(f"{'='*80}\n")
    
    # Load test data
    test_file = f'data/processed/test/{zone.lower().replace("-", "_")}.csv'
    df = pd.read_csv(test_file, parse_dates=['timestamp'])
    
    if len(df) == 0:
        print(f"No test data found for {zone}")
        return
    
    # Load model
    model, norm_data, use_residual = load_model(zone)
    
    if use_residual:
        print("⚠️  WARNING: This model was trained with residual mode!")
        print("   This is NOT the direct prediction model.\n")
        return
    
    print(f"✅ Confirmed: Model trained in DIRECT prediction mode\n")
    
    # Get installed capacity
    capacity = df['installed_capacity_mw'].iloc[0]
    
    # Prepare sequences
    encoder_data, decoder_data, target_data, valid_indices = prepare_sequences_with_future(
        df,
        zone=zone,
        encoder_hours=168,
        decoder_hours=336,
        use_residual=False  # Direct prediction
    )
    
    print(f"Test data prepared:")
    print(f"  Number of sequences: {len(encoder_data)}")
    print(f"  Encoder shape: {encoder_data.shape}")
    print(f"  Decoder shape: {decoder_data.shape}")
    print(f"  Target shape: {target_data.shape}\n")
    
    # Normalize data
    encoder_mean = norm_data['encoder_mean']
    encoder_std = norm_data['encoder_std']
    decoder_mean = norm_data['decoder_mean']
    decoder_std = norm_data['decoder_std']
    
    encoder_data = (encoder_data - encoder_mean) / (encoder_std + 1e-8)
    decoder_data = (decoder_data - decoder_mean) / (decoder_std + 1e-8)
    
    # Convert to tensors
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    
    encoder_tensor = torch.FloatTensor(encoder_data).to(device)
    decoder_tensor = torch.FloatTensor(decoder_data).to(device)
    
    # Make predictions
    print("Generating predictions...")
    with torch.no_grad():
        predictions = model(encoder_tensor, decoder_tensor)
        predictions = predictions.cpu().numpy().flatten()
    
    # Convert from capacity factor to MW
    predictions_mw = predictions * capacity
    actual_mw = target_data.flatten() * capacity
    
    # Zero out nighttime predictions
    solar_rad = decoder_data[:, :, 1]  # Solar radiation is feature 1
    is_night = solar_rad.flatten() < 0.01
    predictions_mw[is_night] = 0
    
    # Calculate metrics
    mae = np.mean(np.abs(predictions_mw - actual_mw))
    rmse = np.sqrt(np.mean((predictions_mw - actual_mw) ** 2))
    
    # R² score
    ss_res = np.sum((actual_mw - predictions_mw) ** 2)
    ss_tot = np.sum((actual_mw - np.mean(actual_mw)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    print(f"\n{'='*80}")
    print("DIRECT PREDICTION MODEL PERFORMANCE")
    print(f"{'='*80}")
    print(f"\nMetrics on test set:")
    print(f"  MAE:  {mae:.2f} MW")
    print(f"  RMSE: {rmse:.2f} MW")
    print(f"  R² Score: {r2:.3f}")
    print(f"\nPrediction statistics:")
    print(f"  Mean prediction: {np.mean(predictions_mw):.2f} MW")
    print(f"  Max prediction:  {np.max(predictions_mw):.2f} MW")
    print(f"  Peak capacity factor: {np.max(predictions_mw) / capacity:.3f}")
    print(f"  Nighttime predictions zeroed: {np.sum(is_night)}/{len(is_night)} ({100*np.sum(is_night)/len(is_night):.1f}%)")
    
    # Compare with day-ahead baseline
    print(f"\n{'='*80}")
    print("COMPARISON WITH DAY-AHEAD BASELINE")
    print(f"{'='*80}\n")
    
    # Load day-ahead data if available
    if 'day_ahead_power_mw' in df.columns:
        # Extract day-ahead for valid sequences
        day_ahead_baseline = []
        for idx in valid_indices:
            seq_start = idx + 168
            seq_end = idx + 168 + 336
            day_ahead_baseline.append(df.iloc[seq_start:seq_end]['day_ahead_power_mw'].values)
        day_ahead_baseline = np.concatenate(day_ahead_baseline)
        
        # Calculate baseline metrics
        mae_baseline = np.mean(np.abs(day_ahead_baseline - actual_mw))
        rmse_baseline = np.sqrt(np.mean((day_ahead_baseline - actual_mw) ** 2))
        ss_res_baseline = np.sum((actual_mw - day_ahead_baseline) ** 2)
        r2_baseline = 1 - (ss_res_baseline / ss_tot) if ss_tot > 0 else 0
        
        print(f"Day-ahead baseline:")
        print(f"  MAE:  {mae_baseline:.2f} MW")
        print(f"  RMSE: {rmse_baseline:.2f} MW")
        print(f"  R² Score: {r2_baseline:.3f}")
        
        print(f"\nDirect model performance vs day-ahead:")
        if mae < mae_baseline:
            improvement = ((mae_baseline - mae) / mae_baseline) * 100
            print(f"  ✅ MAE improved by {improvement:.1f}%")
        else:
            degradation = ((mae - mae_baseline) / mae_baseline) * 100
            print(f"  ❌ MAE worse by {degradation:.1f}%")
        
        if r2 > r2_baseline:
            improvement = ((r2 - r2_baseline) / abs(r2_baseline)) * 100 if r2_baseline != 0 else float('inf')
            print(f"  ✅ R² improved by {improvement:.1f}%")
        else:
            degradation = ((r2_baseline - r2) / abs(r2_baseline)) * 100 if r2_baseline != 0 else float('inf')
            print(f"  ❌ R² worse by {degradation:.1f}%")
    
    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    evaluate_direct_model('IT-NORD')
