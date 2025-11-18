import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from pathlib import Path

from data_loader import (
    load_zone_data,
    WEATHER_FEATURES,
    ENGINEERED_FEATURES,
    compute_day_ahead_capacity_factor,
)
from model import SimpleCNNLSTM, EncoderDecoderCNNLSTM, get_device

ZONE = "IT-NORD"
SEQ_LEN = 168
HORIZON = 336
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_norm_params(path: Path):
    data = np.load(path)
    return data["mean"], data["std"]


def build_test_sample(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Build encoder history + test horizon for Oct 27–Nov 10.

    Returns:
        X_enc_raw: (168, F)
        X_dec_raw: (336, F)
        y_true_MW: (336,)
        day_ahead_cf: (336,)
        inst_cap: (336,)  # installed capacity per hour
        dates: DatetimeIndex length 336
        n_features: int
    """
    train_df = train_df.sort_values("date").reset_index(drop=True)
    test_df = test_df.sort_values("date").reset_index(drop=True)

    assert len(test_df) == HORIZON, f"Expected {HORIZON} test hours, got {len(test_df)}"

    train_tail = train_df.tail(SEQ_LEN)

    features = WEATHER_FEATURES + ENGINEERED_FEATURES
    features = [f for f in features if f in train_tail.columns]

    X_enc_raw = train_tail[features].values         # [168, F]
    X_dec_raw = test_df[features].values            # [336, F]

    y_true_MW = test_df["actual"].values            # [336]

    day_ahead_cf_full = compute_day_ahead_capacity_factor(test_df)
    day_ahead_cf = day_ahead_cf_full                # [336]

    inst_cap = test_df["installed_capacity_mw"].values  # [336]
    dates = pd.to_datetime(test_df["date"].values)

    return X_enc_raw, X_dec_raw, y_true_MW, day_ahead_cf, inst_cap, dates, len(features)


def load_simple_model(n_features: int, device: torch.device) -> SimpleCNNLSTM:
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


def metrics(y_true: np.ndarray, y_pred: np.ndarray):
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return mae, rmse, r2


def plot_forecast(dates, y_true_MW, y_pred_MW, mae, rmse, r2, title_prefix: str, out_path: Path):
    plt.figure(figsize=(12, 4))
    plt.plot(dates, y_true_MW, label="Actual", linewidth=1.5)
    plt.plot(dates, y_pred_MW, label="Predicted", linewidth=1.5)

    plt.ylabel("Solar Production (MW)")
    plt.title(
        f"{title_prefix} (Oct 27 – Nov 10, 2025)\n"
        f"MAE: {mae:.2f} MW | RMSE: {rmse:.2f} MW | R²: {r2:.3f}"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    device = get_device()

    # 1) Load train + test
    train_df = load_zone_data(ZONE, split="train")
    test_df = load_zone_data(ZONE, split="test")   # Oct 27–Nov 10, 2025

    X_enc_raw, X_dec_raw, y_true_MW, day_ahead_cf, inst_cap, dates, n_features = \
        build_test_sample(train_df, test_df)

    # -------- SIMPLE RESIDUAL MODEL --------
    print("\n=== SIMPLE residual model on Oct 27–Nov 10 ===")
    mean_s, std_s = load_norm_params(MODELS_DIR / "norm_simple_resid.npz")
    mean_s = mean_s.reshape(1, 1, -1)
    std_s = std_s.reshape(1, 1, -1)

    # simple only uses encoder window
    X_enc_simple = (X_enc_raw[None, ...] - mean_s) / std_s  # [1, 168, F]

    simple_model = load_simple_model(n_features, device)

    with torch.no_grad():
        Xs = torch.tensor(X_enc_simple, dtype=torch.float32, device=device)
        resid_cf_simple = simple_model(Xs).cpu().numpy().reshape(-1)  # [336]

    cf_pred_simple = np.clip(day_ahead_cf + resid_cf_simple, 0.0, 1.2)
    y_pred_simple_MW = cf_pred_simple * inst_cap

    mae_s, rmse_s, r2_s = metrics(y_true_MW, y_pred_simple_MW)
    print(f"Simple model -> MAE: {mae_s:.2f} MW | RMSE: {rmse_s:.2f} MW | R²: {r2_s:.3f}")

    out_simple = RESULTS_DIR / f"{ZONE}_forecast_oct27_nov10_simple.png"
    plot_forecast(
        dates,
        y_true_MW,
        y_pred_simple_MW,
        mae_s,
        rmse_s,
        r2_s,
        f"{ZONE} – 14-Day Forecast (Simple residual)",
        out_simple,
    )

    # -------- ENCODER RESIDUAL MODEL --------
    print("\n=== ENCODER residual model on Oct 27–Nov 10 ===")
    mean_e, std_e = load_norm_params(MODELS_DIR / "norm_encoder_resid.npz")
    mean_e = mean_e.reshape(1, 1, -1)
    std_e = std_e.reshape(1, 1, -1)

    X_enc_encoder = (X_enc_raw[None, ...] - mean_e) / std_e
    X_dec_encoder = (X_dec_raw[None, ...] - mean_e) / std_e

    encoder_model = load_encoder_model(n_features, device)

    with torch.no_grad():
        Xe = torch.tensor(X_enc_encoder, dtype=torch.float32, device=device)
        Xd = torch.tensor(X_dec_encoder, dtype=torch.float32, device=device)
        resid_cf_encoder = encoder_model(Xe, Xd).cpu().numpy().reshape(-1)

    cf_pred_encoder = np.clip(day_ahead_cf + resid_cf_encoder, 0.0, 1.2)
    y_pred_encoder_MW = cf_pred_encoder * inst_cap

    mae_e, rmse_e, r2_e = metrics(y_true_MW, y_pred_encoder_MW)
    print(f"Encoder model -> MAE: {mae_e:.2f} MW | RMSE: {rmse_e:.2f} MW | R²: {r2_e:.3f}")

    out_encoder = RESULTS_DIR / f"{ZONE}_forecast_oct27_nov10_encoder.png"
    plot_forecast(
        dates,
        y_true_MW,
        y_pred_encoder_MW,
        mae_e,
        rmse_e,
        r2_e,
        f"{ZONE} – 14-Day Forecast (Encoder residual)",
        out_encoder,
    )


if __name__ == "__main__":
    main()
