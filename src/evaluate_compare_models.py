import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Tuple, List

from data_loader import (
    load_zone_data,
    WEATHER_FEATURES,
    ENGINEERED_FEATURES,
    compute_day_ahead_capacity_factor,
)
from model import SimpleCNNLSTM, EncoderDecoderCNNLSTM, get_device

ZONE: str = "IT-NORD"
SEQ_LEN: int = 168
HORIZON: int = 336
MODELS_DIR: Path = Path("models")


def load_norm_params(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load normalization parameters (mean, std) from a .npz file.

    Args:
        path: Path to the saved normalization params.

    Returns:
        mean: 1D array of feature means.
        std:  1D array of feature stds.
    """
    data = np.load(path)
    mean = data["mean"]
    std = data["std"]
    return mean, std


def build_sample(df: pd.DataFrame, start_datetime: str) -> Tuple[
    np.ndarray,  # X_enc_raw
    np.ndarray,  # X_dec_raw
    np.ndarray,  # y_true
    np.ndarray,  # day_ahead_cf
    pd.DatetimeIndex,  # dates
    int,  # n_features
]:
    """
    Build a single encoder–decoder sample from a full time series.

    Args:
        df: Full zone DataFrame (train split).
        start_datetime: Timestamp where the forecast horizon starts.

    Returns:
        X_enc_raw: Past SEQ_LEN hours of features, shape (SEQ_LEN, F).
        X_dec_raw: Future HORIZON hours of features, shape (HORIZON, F).
        y_true:    True capacity_factor for the horizon, shape (HORIZON,).
        day_ahead_cf: Day-ahead capacity factor for the horizon, shape (HORIZON,).
        dates:     DatetimeIndex for the horizon.
        n_features: Number of feature columns used (F).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    features: List[str] = WEATHER_FEATURES + ENGINEERED_FEATURES
    features = [f for f in features if f in df.columns]

    start_ts = pd.to_datetime(start_datetime)
    idx_candidates = df.index[df["date"] == start_ts].tolist()
    if not idx_candidates:
        raise ValueError(f"No exact timestamp {start_ts} in data for {ZONE}")
    idx_start = idx_candidates[0]

    enc_start = idx_start - SEQ_LEN
    enc_end = idx_start
    dec_end = idx_start + HORIZON

    if enc_start < 0 or dec_end > len(df):
        raise ValueError("Not enough history/future around start_datetime")

    X_all = df[features].values  # [T, F]
    X_enc_raw = X_all[enc_start:enc_end]   # [SEQ_LEN, F]
    X_dec_raw = X_all[enc_end:dec_end]     # [HORIZON, F]

    # True CF and day-ahead CF for horizon
    y_true = df["capacity_factor"].values[enc_end:dec_end]  # [HORIZON]
    day_ahead_cf_full = compute_day_ahead_capacity_factor(df)
    day_ahead_cf = day_ahead_cf_full[enc_end:dec_end]

    dates = pd.to_datetime(df["date"].values[enc_end:dec_end])

    return X_enc_raw, X_dec_raw, y_true, day_ahead_cf, dates, len(features)


def load_simple_model(n_features: int, device: torch.device) -> SimpleCNNLSTM:
    """
    Load the trained simple residual model from disk.

    Args:
        n_features: Number of input features per timestep.
        device: Torch device to load the model to.

    Returns:
        Loaded SimpleCNNLSTM model in eval mode.
    """
    ckpt_path = MODELS_DIR / "best_model_simple_resid.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model = SimpleCNNLSTM(
        sequence_length=SEQ_LEN,
        n_features=n_features,
        forecast_horizon=HORIZON,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_encoder_model(n_features: int, device: torch.device) -> EncoderDecoderCNNLSTM:
    """
    Load the trained encoder–decoder residual model from disk.

    Args:
        n_features: Number of input features per timestep.
        device: Torch device to load the model to.

    Returns:
        Loaded EncoderDecoderCNNLSTM model in eval mode.
    """
    ckpt_path = MODELS_DIR / "best_model_encoder_resid.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model = EncoderDecoderCNNLSTM(
        enc_sequence_length=SEQ_LEN,
        dec_sequence_length=HORIZON,
        n_features=n_features,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def plot_comparison(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    day_ahead_cf: np.ndarray,
    y_pred_simple: np.ndarray,
    y_pred_encoder: np.ndarray,
    title: str,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot actual CF, day-ahead CF, and model predictions on the same horizon.

    Args:
        dates: DatetimeIndex for the forecast horizon.
        y_true: True capacity factor, shape (HORIZON,).
        day_ahead_cf: Day-ahead capacity factor, shape (HORIZON,).
        y_pred_simple: Simple model CF predictions, shape (HORIZON,).
        y_pred_encoder: Encoder model CF predictions, shape (HORIZON,).
        title: Plot title.
        save_path: Optional path to save figure; if None, plt.show() is used.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(dates, y_true, label="Actual CF", linewidth=1.5)
    plt.plot(dates, day_ahead_cf, label="Day-ahead CF", linestyle=":", linewidth=1.5)
    plt.plot(
        dates,
        y_pred_simple,
        label="Simple model CF",
        linestyle="--",
        linewidth=1.5,
    )
    plt.plot(
        dates,
        y_pred_encoder,
        label="Encoder model CF",
        linestyle="-.",
        linewidth=1.5,
    )

    plt.ylim(0, 1.05)
    plt.ylabel("Capacity factor")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {save_path}")
    else:
        plt.show()


def main() -> None:
    """
    Compare simple vs encoder models on two horizons (winter/summer) and
    generate MAE numbers + plots.
    """
    device = get_device()

    # 1) Load data
    df = load_zone_data(ZONE, split="train")

    # 2) Scenario start dates (winter vs summer)
    scenarios = [
        "2024-01-19 00:00:00+00:00",  # winter low CF day
        "2024-07-04 00:00:00+00:00",  # summer high CF day
    ]

    for start_dt in scenarios:
        print(f"\n=== Comparing models starting {start_dt} ===")

        X_enc_raw, X_dec_raw, y_true, day_ahead_cf, dates, n_features = build_sample(
            df, start_dt
        )

        # 3) Load normalization params
        mean_simple, std_simple = load_norm_params(MODELS_DIR / "norm_simple_resid.npz")
        mean_encoder, std_encoder = load_norm_params(
            MODELS_DIR / "norm_encoder_resid.npz"
        )

        # reshape means/std to [1, 1, F]
        mean_simple = mean_simple.reshape(1, 1, -1)
        std_simple = std_simple.reshape(1, 1, -1)

        mean_encoder = mean_encoder.reshape(1, 1, -1)
        std_encoder = std_encoder.reshape(1, 1, -1)

        # 4) Normalize features separately for each model
        # simple: encoder-only input [1, SEQ_LEN, F]
        X_enc_simple = (X_enc_raw[None, ...] - mean_simple) / std_simple

        # encoder: encoder + decoder input
        X_enc_encoder = (X_enc_raw[None, ...] - mean_encoder) / std_encoder
        X_dec_encoder = (X_dec_raw[None, ...] - mean_encoder) / std_encoder

        # 5) Load models
        simple_model = load_simple_model(n_features, device)
        encoder_model = load_encoder_model(n_features, device)

        # 6) Run inference (on residual targets)
        with torch.no_grad():
            # Simple model
            Xs = torch.tensor(X_enc_simple, dtype=torch.float32, device=device)
            resid_simple = simple_model(Xs).cpu().numpy().reshape(-1)  # [HORIZON]

            # Encoder model
            Xe = torch.tensor(X_enc_encoder, dtype=torch.float32, device=device)
            Xd = torch.tensor(X_dec_encoder, dtype=torch.float32, device=device)
            resid_encoder = encoder_model(Xe, Xd).cpu().numpy().reshape(-1)

        # 7) Add day-ahead back to get CF predictions
        cf_pred_simple = day_ahead_cf + resid_simple
        cf_pred_encoder = day_ahead_cf + resid_encoder

        # clip to [0,1] sane range
        cf_pred_simple = np.clip(cf_pred_simple, 0.0, 1.2)
        cf_pred_encoder = np.clip(cf_pred_encoder, 0.0, 1.2)

        # 8) Compute simple MAEs for this horizon
        mae_simple = float(np.mean(np.abs(cf_pred_simple - y_true)))
        mae_encoder = float(np.mean(np.abs(cf_pred_encoder - y_true)))
        mae_dayahead = float(np.mean(np.abs(day_ahead_cf - y_true)))

        print("  Horizon MAE:")
        print(f"    Day-ahead: {mae_dayahead:.4f}")
        print(f"    Simple model: {mae_simple:.4f}")
        print(f"    Encoder model: {mae_encoder:.4f}")

        title = (
            f"{ZONE} forecast starting {start_dt} "
            f"(MAE: DA={mae_dayahead:.3f}, simple={mae_simple:.3f}, enc={mae_encoder:.3f})"
        )
        out_name = f"results/{ZONE}_compare_{start_dt[:10]}.png"
        plot_comparison(
            dates,
            y_true,
            day_ahead_cf,
            cf_pred_simple,
            cf_pred_encoder,
            title,
            out_name,
        )


if __name__ == "__main__":
    main()
