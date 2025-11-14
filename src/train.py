"""
PyTorch training script for solar forecasting.
Stable on Apple Silicon with MPS acceleration!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from data_loader import (
    load_all_zones, 
    prepare_sequences, 
    normalize_data,
    WEATHER_FEATURES
)
from model import (
    SimpleCNNLSTM,
    CNNLSTM,
    get_device,
    print_model_summary,
    calculate_metrics
)


class VariationAwareLoss(nn.Module):
    """
    Custom loss that encourages predictions to vary more.
    Combines MSE with a penalty for low variance in predictions.
    """
    def __init__(self, variance_weight=0.2):
        super(VariationAwareLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.variance_weight = variance_weight
        
    def forward(self, pred, target):
        # Standard MSE loss
        mse_loss = self.mse(pred, target)
        
        # Variance penalty: penalize if predictions have low variance
        pred_var = torch.var(pred, dim=1).mean()
        target_var = torch.var(target, dim=1).mean()
        
        # Penalize when prediction variance is much lower than target variance
        variance_penalty = torch.relu(target_var - pred_var) / (target_var + 1e-8)
        
        # Combined loss
        total_loss = mse_loss + self.variance_weight * variance_penalty
        
        return total_loss


def train_model(zones=None,
                sequence_length=168,
                forecast_horizon=336,  # 14 days
                batch_size=32,
                epochs=50,
                validation_split=0.2,
                learning_rate=0.001,
                sample_frac=None,
                model_type='simple'):
    """
    Train CNN-LSTM model using PyTorch.
    
    Args:
        zones: List of zone names (None = all zones)
        sequence_length: Input sequence length in hours (default: 168 = 7 days)
        forecast_horizon: Output forecast length in hours (default: 336 = 14 days)
        batch_size: Batch size for training
        epochs: Number of training epochs
        validation_split: Fraction of data for validation
        learning_rate: Learning rate for optimizer
        sample_frac: Fraction of data to use (for testing, e.g., 0.1 = 10%)
        model_type: 'simple' or 'full'
    """
    
    device = get_device()
    print("\n" + "="*80)
    print("PYTORCH TRAINING - APPLE SILICON OPTIMIZED")
    print("="*80)
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Load data
    print("\n1. Loading training data...")
    train_data = load_all_zones(split='train')
    
    if zones:
        train_data = {k: v for k, v in train_data.items() if k in zones}
    
    print(f"   Loaded {len(train_data)} zones: {list(train_data.keys())}")
    
    # Prepare sequences
    print("\n2. Preparing sequences...")
    all_X, all_y = [], []
    
    for zone, df in train_data.items():
        print(f"\n   Processing {zone}:")
        print(f"   - Original size: {len(df)} records")
        
        if sample_frac:
            df = df.sample(frac=sample_frac, random_state=42)
            print(f"   - Sampled to: {len(df)} records ({sample_frac*100:.0f}%)")
        
        X_zone, y_zone = prepare_sequences(
            df, 
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon
        )
        
        print(f"   - Created {len(X_zone)} sequences")
        all_X.append(X_zone)
        all_y.append(y_zone)
    
    # Combine data
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    print(f"\n   Total sequences: {len(X)}")
    print(f"   Input shape: {X.shape} (samples, timesteps, features)")
    print(f"   Target shape: {y.shape} (samples, forecast_horizon)")
    
    # Normalize
    print("\n3. Normalizing data...")
    X_norm, _, norm_params = normalize_data(X)
    
    # Save normalization parameters
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    np.savez(models_dir / 'normalization_params_pytorch.npz', **norm_params)
    print(f"   Saved normalization parameters")
    
    # Train/validation split
    print("\n4. Splitting train/validation...")
    n_val = int(len(X_norm) * validation_split)
    n_train = len(X_norm) - n_val
    
    X_train, X_val = X_norm[:n_train], X_norm[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    print(f"   Train: {len(X_train)} sequences")
    print(f"   Validation: {len(X_val)} sequences")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Build model
    print("\n5. Building model...")
    n_features = X_train.shape[2]
    
    if model_type == 'simple':
        model = SimpleCNNLSTM(
            sequence_length=sequence_length,
            n_features=n_features,
            forecast_horizon=forecast_horizon
        ).to(device)
    else:
        model = CNNLSTM(
            sequence_length=sequence_length,
            n_features=n_features,
            forecast_horizon=forecast_horizon
        ).to(device)
    
    print_model_summary(model, sequence_length, n_features)
    
    # Custom loss function that encourages variation
    criterion = VariationAwareLoss(variance_weight=0.2)
    optimizer = optim.NAdam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler - reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print(f"\n6. Training (epochs={epochs}, batch_size={batch_size})...")
    print("="*80)
    
    history = {'train_loss': [], 'train_mae': [], 'val_loss': [], 'val_mae': [], 'lr': []}
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_mae = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for X_batch, y_batch in train_pbar:
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            train_mae += torch.mean(torch.abs(y_pred - y_batch)).item()
            
            train_pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_mae = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for X_batch, y_batch in val_pbar:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                
                val_loss += loss.item()
                val_mae += torch.mean(torch.abs(y_pred - y_batch)).item()
                
                val_pbar.set_postfix({'loss': loss.item()})
        
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train - Loss: {train_loss:.5f}, MAE: {train_mae:.5f}")
        print(f"  Val   - Loss: {val_loss:.5f}, MAE: {val_mae:.5f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
            }, models_dir / 'best_model_pytorch.pt')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.5f})")
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
    
    # Save training history
    print("\n7. Saving results...")
    
    history_file = models_dir / 'training_history_pytorch.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"   Saved training history")
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_mae'], label='Train')
    axes[1].plot(history['val_mae'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Mean Absolute Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = models_dir / 'training_curves_pytorch.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved training curves")
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.5f}")
    print(f"Best validation MAE: {min(history['val_mae']):.5f}")
    print(f"Model saved to: {models_dir / 'best_model_pytorch.pt'}")
    print("="*80 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CNN-LSTM model with PyTorch')
    parser.add_argument('--zones', type=str, nargs='+', default=None,
                       help='Zones to train on (default: all)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--sample', type=float, default=None,
                       help='Sample fraction (e.g., 0.1 for 10%%)')
    parser.add_argument('--model-type', type=str, default='simple',
                       choices=['simple', 'full'],
                       help='Model architecture (default: simple)')
    
    args = parser.parse_args()
    
    train_model(
        zones=args.zones,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        sample_frac=args.sample,
        model_type=args.model_type
    )
