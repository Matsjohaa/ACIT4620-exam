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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from typing import List, Optional, Dict

sys.path.append(str(Path(__file__).parent))

from data_loader import (  # type: ignore
    load_all_zones,
    prepare_sequences,
    prepare_sequences_with_future,
    normalize_data,
)

from model import (  # type: ignore
    EncoderDecoderCNNLSTM,
    get_device,
    print_model_summary,
)

from model_large import (  # type: ignore
    LargeEncoderDecoderCNNLSTM,
)

from model_attention import (  # type: ignore
    EncoderDecoderAttentionCNNLSTM,
)


class ImprovedLoss(nn.Module):
    """
    Improved loss function with asymmetric penalty for underprediction.
    
    Key improvements:
    1. Penalizes underprediction MORE than overprediction (addresses -48% bias)
    2. Weights errors by production magnitude (focuses on high-production periods)
    3. Variance penalty for prediction diversity
    """
    
    def __init__(self, 
                 underprediction_weight: float = 2.0,
                 production_weight: float = 0.3,
                 variance_weight: float = 0.2):
        super().__init__()
        self.underprediction_weight = underprediction_weight
        self.production_weight = production_weight
        self.variance_weight = variance_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute improved loss.
        
        Args:
            pred: Model predictions, shape (batch, horizon)
            target: Ground truth, same shape as pred
        
        Returns:
            Scalar loss with asymmetric penalties
        """
        # 1. Asymmetric loss: penalize underprediction more
        error = target - pred
        
        # Underprediction (error > 0): higher penalty
        # Overprediction (error < 0): standard penalty
        squared_error = error ** 2
        asymmetric_loss = torch.where(
            error > 0,
            self.underprediction_weight * squared_error,  # 2x penalty for underprediction
            squared_error  # Normal penalty for overprediction
        )
        
        # 2. Production-weighted loss: focus on high-production hours
        # Weight by target magnitude (normalized)
        production_weights = 1.0 + self.production_weight * (target / (target.max() + 1e-8))
        weighted_loss = (asymmetric_loss * production_weights).mean()
        
        # 3. Variance penalty: encourage prediction diversity
        # Only compute for non-zero targets (daytime)
        mask = target > 0.01
        if mask.sum() > 1:
            pred_daytime = pred[mask]
            target_daytime = target[mask]
            
            pred_var = torch.var(pred_daytime)
            target_var = torch.var(target_daytime)
            variance_penalty = torch.relu(target_var - pred_var) / (target_var + 1e-8)
        else:
            variance_penalty = torch.tensor(0.0, device=pred.device)
        
        # Combined loss
        total_loss = weighted_loss + self.variance_weight * variance_penalty
        
        return total_loss


class DaytimeFocusedLoss(nn.Module):
    """
    Loss function that focuses training on daytime hours (radiation > 0).
    This prevents the model from wasting capacity learning trivial night→zero patterns.
    
    Nighttime hours contribute minimally to the loss, allowing the model to 
    focus on learning the complex daytime solar production patterns.
    """
    
    def __init__(self, variance_weight: float = 0.2, nighttime_weight: float = 0.1):
        super().__init__()
        self.variance_weight = variance_weight
        self.nighttime_weight = nighttime_weight  # Reduced weight for nighttime
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute daytime-focused loss.
        
        Args:
            pred: Model predictions, shape (batch, horizon)
            target: Ground truth, same shape as pred
        
        Returns:
            Scalar loss with higher weight on daytime hours
        """
        # Identify daytime vs nighttime based on target values
        # If target > 0.01, consider it daytime (non-trivial production)
        is_daytime = (target > 0.01).float()
        
        # Create weight mask: 1.0 for daytime, nighttime_weight for nighttime
        weights = is_daytime + (1 - is_daytime) * self.nighttime_weight
        
        # Weighted MSE loss
        squared_errors = (pred - target) ** 2
        weighted_mse = (squared_errors * weights).mean()
        
        # Variance penalty (only on daytime hours)
        if is_daytime.sum() > 0:
            # Mask to get only daytime predictions/targets
            daytime_mask = is_daytime > 0.5
            pred_daytime = pred[daytime_mask]
            target_daytime = target[daytime_mask]
            
            if len(pred_daytime) > 1:  # Need at least 2 samples for variance
                pred_var = torch.var(pred_daytime)
                target_var = torch.var(target_daytime)
                variance_penalty = torch.relu(target_var - pred_var) / (target_var + 1e-8)
            else:
                variance_penalty = torch.tensor(0.0, device=pred.device)
        else:
            variance_penalty = torch.tensor(0.0, device=pred.device)
        
        # Combined loss
        total_loss = weighted_mse + self.variance_weight * variance_penalty
        
        return total_loss


def train_model(
    zones: Optional[List[str]] = None,
    sequence_length: int = 168,
    forecast_horizon: int = 336,  # 14 days
    batch_size: int = 32,
    epochs: int = 50,
    validation_split: float = 0.2,
    learning_rate: float = 0.001,
    sample_frac: Optional[float] = None,
    use_large_model: bool = False,
    use_attention: bool = False,
) -> str:
    """
    Train EncoderDecoderCNNLSTM model using PyTorch (direct prediction mode).

    Args:
        zones: List of zone names (None = all zones).
        sequence_length: Input sequence length in hours (default: 168 = 7 days).
        forecast_horizon: Output forecast length in hours (default: 336 = 14 days).
        batch_size: Batch size for training.
        epochs: Number of training epochs.
        validation_split: Fraction of data for validation.
        learning_rate: Learning rate for optimizer.
        sample_frac: Optional fraction of data to use (for testing).
        use_large_model: Whether to use large model (not recommended).
        use_attention: Whether to use attention mechanism (experimental).
        sample_frac: Fraction of data to use (for testing, e.g., 0.1 = 10%).
        use_large_model: If True, use large model with ~2.5M parameters.
    """

    device = get_device()
    print("\n" + "=" * 80)
    print("PYTORCH TRAINING - APPLE SILICON OPTIMIZED")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n1. Loading training data...")
    train_data: Dict[str, pd.DataFrame] = load_all_zones(split="train")

    if zones:
        train_data = {k: v for k, v in train_data.items() if k in zones}

    print(f"   Loaded {len(train_data)} zones: {list(train_data.keys())}")

    # ------------------------------------------------------------------
    # 2. Prepare sequences
    # ------------------------------------------------------------------
    print("\n2. Preparing sequences...")

    all_X_enc: List[np.ndarray] = []
    all_X_dec: List[np.ndarray] = []
    all_y_encdec: List[np.ndarray] = []

    for zone, df in train_data.items():
        print(f"\n   Processing {zone}:")
        print(f"   - Original size: {len(df)} records")

        if sample_frac is not None:
            n_samples = int(len(df) * sample_frac)
            df = df.iloc[:n_samples]
            print(f"   - Using sample fraction {sample_frac}, {len(df)} records")

        X_enc_zone, X_dec_zone, y_zone = prepare_sequences_with_future(
            df,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon,
        )
        print(f"   - Created {len(X_enc_zone)} encoder–decoder sequences")

        if X_enc_zone.size > 0:
            all_X_enc.append(X_enc_zone)
            all_X_dec.append(X_dec_zone)
            all_y_encdec.append(y_zone)

    # ------------------------------------------------------------------
    # 3. Combine + normalize
    # ------------------------------------------------------------------
    print("\n3. Normalizing data...")

    # Determine zone name for folder structure
    if zones and len(zones) == 1:
        zone_name = zones[0].lower()
    else:
        zone_name = "multi-zone"
    
    models_dir = Path("models") / zone_name
    models_dir.mkdir(exist_ok=True, parents=True)
    print(f"   Using models directory: {models_dir}")

    if len(all_X_enc) == 0:
        raise ValueError(
            "No encoder–decoder sequences were created.\n"
            "Possible causes:\n"
            "  - zones filter left you with 0 zones (check --zones argument)\n"
            "  - after residual calculation there are NaNs in 'capacity_factor' "
            "or 'day-ahead' / 'installed_capacity_mw', so all windows were filtered out.\n"
            "  - df is too short for sequence_length + forecast_horizon."
        )

    X_enc = np.concatenate(all_X_enc, axis=0)
    X_dec = np.concatenate(all_X_dec, axis=0)
    y = np.concatenate(all_y_encdec, axis=0)

    print(f"\n   Total sequences: {len(X_enc)}")
    print(f"   Encoder input shape: {X_enc.shape}")
    print(f"   Decoder input shape: {X_dec.shape}")
    print(f"   Target shape: {y.shape}")

    # Normalize encoder + decoder together along the time axis
    X_all = np.concatenate([X_enc, X_dec], axis=1)  # [N, 168+336, F]
    X_all_norm, _, norm_params = normalize_data(X_all)

    # Split back to enc/dec parts
    X_enc_norm = X_all_norm[:, :sequence_length, :]
    X_dec_norm = X_all_norm[:, sequence_length:, :]

    # Save normalization params
    norm_filename = "norm.npz"
    np.savez(models_dir / norm_filename, **norm_params)
    print(f"   Saved normalization parameters to {models_dir / norm_filename}")

    # Train/val split
    print("\n4. Splitting train/validation...")
    n_val = int(len(X_enc_norm) * validation_split)
    n_train = len(X_enc_norm) - n_val

    X_enc_train, X_enc_val = X_enc_norm[:n_train], X_enc_norm[n_train:]
    X_dec_train, X_dec_val = X_dec_norm[:n_train], X_dec_norm[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    print(f"   Train: {len(X_enc_train)} sequences")
    print(f"   Validation: {len(X_enc_val)} sequences")

    # Tensors (stay on CPU; moved to device in loop)
    X_enc_train_tensor = torch.FloatTensor(X_enc_train)
    X_dec_train_tensor = torch.FloatTensor(X_dec_train)
    y_train_tensor = torch.FloatTensor(y_train)

    X_enc_val_tensor = torch.FloatTensor(X_enc_val)
    X_dec_val_tensor = torch.FloatTensor(X_dec_val)
    y_val_tensor = torch.FloatTensor(y_val)

    # Datasets / loaders
    train_dataset = TensorDataset(
        X_enc_train_tensor, X_dec_train_tensor, y_train_tensor
    )
    val_dataset = TensorDataset(
        X_enc_val_tensor, X_dec_val_tensor, y_val_tensor
    )

    train_loader: DataLoader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader: DataLoader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    n_features = X_enc_train.shape[2]

    # ------------------------------------------------------------------
    # 5. Build model
    # ------------------------------------------------------------------
    print("\n5. Building model...")

    if use_attention:
        print("Using ATTENTION model with temporal awareness (~280k parameters)")
        model: nn.Module = EncoderDecoderAttentionCNNLSTM(
            enc_sequence_length=sequence_length,
            dec_sequence_length=forecast_horizon,
            n_features=n_features,
        ).to(device)
    elif use_large_model:
        print("Using LARGE model (~1M parameters)")
        model: nn.Module = LargeEncoderDecoderCNNLSTM(
            enc_sequence_length=sequence_length,
            dec_sequence_length=forecast_horizon,
            n_features=n_features,
        ).to(device)
    else:
        print("Using standard model (~242k parameters)")
        model: nn.Module = EncoderDecoderCNNLSTM(
            enc_sequence_length=sequence_length,
            dec_sequence_length=forecast_horizon,
            n_features=n_features,
        ).to(device)

    print_model_summary(model, sequence_length, n_features)

    # Using improved loss with asymmetric penalties to fix underprediction bias
    criterion: nn.Module = ImprovedLoss(
        underprediction_weight=2.0,  # 2x penalty for underprediction
        production_weight=0.3,        # Focus on high-production hours
        variance_weight=0.2           # Encourage prediction diversity
    )
    print("\n📊 Using ImprovedLoss:")
    print("   • Asymmetric penalties: 2.0x for underprediction (fixes bias)")
    print("   • Production weighting: 0.3 (focus on high-production hours)")
    print("   • Variance penalty: 0.2 (encourage prediction diversity)")
    
    optimizer: optim.Optimizer = optim.NAdam(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=1e-4  # L2 regularization to prevent overfitting
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    print(f"\n6. Training (epochs={epochs}, batch_size={batch_size})...")
    print("=" * 80)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_mae": [],
        "val_loss": [],
        "val_mae": [],
        "lr": [],
    }
    best_val_loss: float = float("inf")
    best_model_state = None
    patience = 8  # Early stopping patience
    patience_counter = 0
    
    print(f"Early stopping enabled: patience={patience} epochs")
    print(f"Weight decay (L2): 1e-4")
    print(f"Gradient clipping: max_norm=1.0")
    print("=" * 80)

    for epoch in range(epochs):
        # ------------------- TRAIN -------------------
        model.train()
        train_loss = 0.0
        train_mae = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in train_pbar:
            optimizer.zero_grad()

            X_enc_batch, X_dec_batch, y_batch = batch
            X_enc_batch = X_enc_batch.to(device)
            X_dec_batch = X_dec_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_enc_batch, X_dec_batch)

            loss = criterion(y_pred, y_batch)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            train_loss += loss.item()
            train_mae += torch.mean(torch.abs(y_pred - y_batch)).item()

            train_pbar.set_postfix({"loss": loss.item()})

        train_loss /= len(train_loader)
        train_mae /= len(train_loader)

        # ------------------- VALIDATION -------------------
        model.eval()
        val_loss = 0.0
        val_mae = 0.0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for batch in val_pbar:
                X_enc_batch, X_dec_batch, y_batch = batch
                X_enc_batch = X_enc_batch.to(device)
                X_dec_batch = X_dec_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred = model(X_enc_batch, X_dec_batch)

                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
                val_mae += torch.mean(torch.abs(y_pred - y_batch)).item()

                val_pbar.set_postfix({"loss": loss.item()})

        val_loss /= len(val_loader)
        val_mae /= len(val_loader)

        # Record history
        history["train_loss"].append(train_loss)
        history["train_mae"].append(train_mae)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # Epoch summary
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train - Loss: {train_loss:.5f}, MAE: {train_mae:.5f}")
        print(f"  Val   - Loss: {val_loss:.5f}, MAE: {val_mae:.5f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0  # Reset patience counter
            
            model_filename = "model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                },
                models_dir / model_filename,
            )
            print(f"  ✓ Saved best model to {models_dir / model_filename} (val_loss: {val_loss:.5f})")
        else:
            patience_counter += 1
            print(f"  ⚠ No improvement for {patience_counter}/{patience} epochs")

        # Early stopping check
        if patience_counter >= patience:
            print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            print(f"   Best validation loss: {best_val_loss:.5f} (from epoch {epoch+1-patience_counter})")
            print(f"   Restoring best model weights...")
            model.load_state_dict(best_model_state)
            break

        scheduler.step(val_loss)

    # ------------------------------------------------------------------
    # 7. Save results & plots
    # ------------------------------------------------------------------
    print("\n7. Saving results...")

    history_file = models_dir / "training_history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"   Saved training history to {history_file}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    axes[0].plot(history["train_loss"], label="Train", linewidth=2, marker='o', markersize=4)
    axes[0].plot(history["val_loss"], label="Validation", linewidth=2, marker='s', markersize=4)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss (MSE)", fontsize=11)
    axes[0].set_title("Training Loss", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Add overfitting indicator
    train_val_gap = history["val_loss"][-1] - history["train_loss"][-1]
    if train_val_gap > 0.1:
        axes[0].text(0.5, 0.95, "⚠️ Possible Overfitting", 
                    transform=axes[0].transAxes, fontsize=10, 
                    ha='center', va='top', color='red', fontweight='bold')
    elif abs(train_val_gap) < 0.01:
        axes[0].text(0.5, 0.95, "✓ Good Fit", 
                    transform=axes[0].transAxes, fontsize=10, 
                    ha='center', va='top', color='green', fontweight='bold')

    # MAE plot
    axes[1].plot(history["train_mae"], label="Train", linewidth=2, marker='o', markersize=4)
    axes[1].plot(history["val_mae"], label="Validation", linewidth=2, marker='s', markersize=4)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("MAE", fontsize=11)
    axes[1].set_title("Mean Absolute Error", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Add best epoch marker
    best_epoch = np.argmin(history["val_mae"])
    axes[1].axvline(best_epoch, color='red', linestyle='--', alpha=0.5, linewidth=1)
    axes[1].text(best_epoch, axes[1].get_ylim()[1]*0.95, f'Best: Epoch {best_epoch+1}',
                ha='center', fontsize=9, color='red')

    plt.tight_layout()
    plot_file = models_dir / "training_curves.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved training curves to {plot_file}")

    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Zone: {zone_name.upper()}")
    print(f"Best validation loss: {best_val_loss:.5f}")
    print(f"Best validation MAE: {min(history['val_mae']):.5f}")
    print(f"Model saved to: {models_dir / 'model.pt'}")
    print(f"Normalization saved to: {models_dir / 'norm.npz'}")
    print("=" * 80 + "\n")
    
    return zone_name


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train CNN-LSTM model with PyTorch")
    parser.add_argument(
        "--zones",
        type=str,
        nargs="+",
        default=None,
        help="Zones to train on (default: all)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs (default: 50)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        help="Sample fraction: e.g. 0.1 for 10%%",
    )
    parser.add_argument(
        "--large-model",
        action="store_true",
        help="Use large model with ~2.5M parameters (default: standard 242k model)"
    )
    parser.add_argument(
        "--attention",
        action="store_true",
        help="Use attention mechanism for temporal awareness (experimental, ~280k parameters)"
    )

    args = parser.parse_args()

    train_model(
        zones=args.zones,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        sample_frac=args.sample,
        use_large_model=args.large_model,
        use_attention=args.attention,
    )
