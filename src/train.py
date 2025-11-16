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
    CNNLSTM,
    SimpleCNNLSTM,
    EncoderDecoderCNNLSTM,
    get_device,
    print_model_summary,
)


class VariationAwareLoss(nn.Module):
    """
    Custom loss that encourages predictions to vary more.
    Combines MSE with a penalty for low variance in predictions.
    """

    def __init__(self, variance_weight: float = 0.2) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.variance_weight = variance_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute variation-aware loss.

        Args:
            pred: Model predictions, shape (batch, horizon).
            target: Ground truth, same shape as pred.

        Returns:
            Scalar loss tensor.
        """
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


def train_model(
    zones: Optional[List[str]] = None,
    sequence_length: int = 168,
    forecast_horizon: int = 336,  # 14 days
    batch_size: int = 32,
    epochs: int = 50,
    validation_split: float = 0.2,
    learning_rate: float = 0.001,
    sample_frac: Optional[float] = None,
    model_type: str = "simple",
    use_residual: bool = False,
) -> None:
    """
    Train CNN-LSTM model using PyTorch.

    Args:
        zones: List of zone names (None = all zones).
        sequence_length: Input sequence length in hours (default: 168 = 7 days).
        forecast_horizon: Output forecast length in hours (default: 336 = 14 days).
        batch_size: Batch size for training.
        epochs: Number of training epochs.
        validation_split: Fraction of data for validation.
        learning_rate: Learning rate for optimizer.
        sample_frac: Fraction of data to use (for testing, e.g., 0.1 = 10%).
        model_type: 'simple', 'full', or 'encoder'.
        use_residual: If True, train on residual (cf - day_ahead_cf).
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
    use_encoder_decoder: bool = model_type == "encoder"

    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
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

        if use_encoder_decoder:
            X_enc_zone, X_dec_zone, y_zone = prepare_sequences_with_future(
                df,
                sequence_length=sequence_length,
                forecast_horizon=forecast_horizon,
                use_residual=use_residual,
            )
            print(f"   - Created {len(X_enc_zone)} encoder–decoder sequences")

            if X_enc_zone.size > 0:
                all_X_enc.append(X_enc_zone)
                all_X_dec.append(X_dec_zone)
                all_y_encdec.append(y_zone)

        else:
            X_zone, y_zone = prepare_sequences(
                df,
                sequence_length=sequence_length,
                forecast_horizon=forecast_horizon,
                use_residual=use_residual,
            )
            print(f"   - Created {len(X_zone)} sequences")

            if X_zone.size > 0:
                all_X.append(X_zone)
                all_y.append(y_zone)

    # ------------------------------------------------------------------
    # 3. Combine + normalize
    # ------------------------------------------------------------------
    print("\n3. Normalizing data...")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    if use_encoder_decoder:
        if len(all_X_enc) == 0:
            raise ValueError(
                "No encoder–decoder sequences were created.\n"
                "Possible causes:\n"
                "  - zones filter left you with 0 soner (sjekk --zones argument)\n"
                "  - after residual calculation there are NaNs in 'capacity_factor' "
                "or 'day-ahead' / 'installed_capacity_mw', so all windows were filtered ut.\n"
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
        np.savez(models_dir / "normalization_params_pytorch.npz", **norm_params)
        print(f"   Saved normalization parameters")

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

    else:
        # Old single-input path
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)

        print(f"\n   Total sequences: {len(X)}")
        print(f"   Input shape: {X.shape} (samples, timesteps, features)")
        print(f"   Target shape: {y.shape} (samples, forecast_horizon)")

        X_norm, _, norm_params = normalize_data(X)
        np.savez(models_dir / "normalization_params_pytorch.npz", **norm_params)
        print(f"   Saved normalization parameters")

        print("\n4. Splitting train/validation...")
        n_val = int(len(X_norm) * validation_split)
        n_train = len(X_norm) - n_val

        X_train, X_val = X_norm[:n_train], X_norm[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]

        print(f"   Train: {len(X_train)} sequences")
        print(f"   Validation: {len(X_val)} sequences")

        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        n_features = X_train.shape[2]

    # ------------------------------------------------------------------
    # 5. Build model
    # ------------------------------------------------------------------
    print("\n5. Building model...")

    if model_type == "simple":
        model: nn.Module = SimpleCNNLSTM(
            sequence_length=sequence_length,
            n_features=n_features,
            forecast_horizon=forecast_horizon,
        ).to(device)
    elif model_type == "full":
        model = CNNLSTM(
            sequence_length=sequence_length,
            n_features=n_features,
            forecast_horizon=forecast_horizon,
        ).to(device)
    elif model_type == "encoder":
        model = EncoderDecoderCNNLSTM(
            enc_sequence_length=sequence_length,
            dec_sequence_length=forecast_horizon,
            n_features=n_features,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    print_model_summary(model, sequence_length, n_features)

    criterion: nn.Module = VariationAwareLoss(variance_weight=0.2)
    optimizer: optim.Optimizer = optim.NAdam(model.parameters(), lr=learning_rate)
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

    for epoch in range(epochs):
        # ------------------- TRAIN -------------------
        model.train()
        train_loss = 0.0
        train_mae = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in train_pbar:
            optimizer.zero_grad()

            if use_encoder_decoder:
                X_enc_batch, X_dec_batch, y_batch = batch
                X_enc_batch = X_enc_batch.to(device)
                X_dec_batch = X_dec_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred = model(X_enc_batch, X_dec_batch)
            else:
                X_batch, y_batch = batch
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch)
            loss.backward()
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
                if use_encoder_decoder:
                    X_enc_batch, X_dec_batch, y_batch = batch
                    X_enc_batch = X_enc_batch.to(device)
                    X_dec_batch = X_dec_batch.to(device)
                    y_batch = y_batch.to(device)

                    y_pred = model(X_enc_batch, X_dec_batch)
                else:
                    X_batch, y_batch = batch
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)

                    y_pred = model(X_batch)

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
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                },
                models_dir / "best_model_pytorch.pt",
            )
            print(f"  ✓ Saved best model (val_loss: {val_loss:.5f})")

        scheduler.step(val_loss)

    # ------------------------------------------------------------------
    # 7. Save results & plots
    # ------------------------------------------------------------------
    print("\n7. Saving results...")

    history_file = models_dir / "training_history_pytorch.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    print("   Saved training history")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_mae"], label="Train")
    axes[1].plot(history["val_mae"], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("Mean Absolute Error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = models_dir / "training_curves_pytorch.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved training curves")

    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.5f}")
    print(f"Best validation MAE: {min(history['val_mae']):.5f}")
    print(f"Model saved to: {models_dir / 'best_model_pytorch.pt'}")
    print("=" * 80 + "\n")


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
        help="Sample fraction (e.g., 0.1 for 10%)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="simple",
        choices=["simple", "full", "encoder"],
        help="Model architecture (default: simple)",
    )
    parser.add_argument(
        "--residual",
        action="store_true",
        help=(
            "Train on residual (capacity_factor - day_ahead_cf) "
            "instead of raw capacity_factor"
        ),
    )

    args = parser.parse_args()

    train_model(
        zones=args.zones,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        sample_frac=args.sample,
        model_type=args.model_type,
        use_residual=args.residual,
    )
