"""
PyTorch implementation of Encoder-Decoder CNN-LSTM model for solar forecasting.
Optimized for Apple Silicon MPS acceleration.
"""

import torch
import torch.nn as nn
import numpy as np


def get_device():
    """
    Get the best available device (MPS for Apple Silicon, CUDA for NVIDIA, CPU otherwise).
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def print_model_summary(model, sequence_length, n_features):
    """Print model summary."""
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE")
    print("="*80)
    print(model)
    print("\n" + "="*80)
    print(f"Total parameters: {model.count_parameters():,}")
    print(f"Input shape: (batch_size, {sequence_length}, {n_features})")
    print(f"Output shape: (batch_size, {model.forecast_horizon})")
    print("="*80 + "\n")


def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: True values (numpy array)
        y_pred: Predicted values (numpy array)
        
    Returns:
        Dictionary of metrics
    """
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # MAPE (avoiding division by zero)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }


class EncoderDecoderCNNLSTM(nn.Module):
    """
    IMPROVED Encoder–decoder CNN-LSTM model with enhanced decoder capacity.

    Encoder:
        - CNN + LSTM over past weather (168h) -> context vector h_enc.

    Decoder:
        - For each future hour, take weather features + h_enc
        - Pass through a DEEPER MLP with batch normalization to predict capacity_factor.
    """

    def __init__(
        self,
        enc_sequence_length: int = 168,
        dec_sequence_length: int = 336,
        n_features: int = 15,
        encoder_hidden: int = 128,  # Increased from 64
        decoder_hidden: int = 256,  # Increased from 128
        dropout: float = 0.15,  # Slightly increased
    ):
        super().__init__()

        self.enc_sequence_length = enc_sequence_length
        self.dec_sequence_length = dec_sequence_length
        self.n_features = n_features
        self.forecast_horizon = dec_sequence_length

        # --- Encoder: CNN + LSTM over past weather ---
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # after two poolings of size 2: length = enc_sequence_length // 4
        self.encoder_lstm = nn.LSTM(
            input_size=128,
            hidden_size=encoder_hidden,
            batch_first=True,
        )

        self.encoder_dropout = nn.Dropout(dropout)

        # --- Decoder: ENHANCED MLP over future weather + context ---
        # For each future timestep we will concat [weather_t, h_enc]
        decoder_input_dim = n_features + encoder_hidden

        # IMPROVED: Deeper decoder with batch normalization
        self.decoder_fc1 = nn.Linear(decoder_input_dim, decoder_hidden)
        self.decoder_bn1 = nn.BatchNorm1d(dec_sequence_length)  # Batch norm over time
        self.decoder_dropout1 = nn.Dropout(dropout)
        
        self.decoder_fc2 = nn.Linear(decoder_hidden, decoder_hidden // 2)
        self.decoder_bn2 = nn.BatchNorm1d(dec_sequence_length)
        self.decoder_dropout2 = nn.Dropout(dropout)
        
        self.decoder_fc3 = nn.Linear(decoder_hidden // 2, decoder_hidden // 4)  # NEW layer
        self.decoder_bn3 = nn.BatchNorm1d(dec_sequence_length)
        self.decoder_dropout3 = nn.Dropout(dropout)
        
        self.decoder_out = nn.Linear(decoder_hidden // 4, 1)

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)

    def encode(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        x_enc: [batch, enc_T, n_features]
        returns h_enc: [batch, encoder_hidden]
        """
        # CNN expects [batch, channels, time]
        x = x_enc.transpose(1, 2)  # [B, F, T]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        # back to [batch, T', C] for LSTM
        x = x.transpose(1, 2)

        # LSTM over encoded sequence
        _, (h_n, _) = self.encoder_lstm(x)   # h_n: [num_layers, B, H]
        h_enc = h_n[-1]                      # [B, H] last layer

        h_enc = self.encoder_dropout(h_enc)
        return h_enc

    def forward(self, x_enc: torch.Tensor, x_dec: torch.Tensor) -> torch.Tensor:
        """
        x_enc: [batch, enc_T, n_features]
        x_dec: [batch, dec_T, n_features]
        returns y_pred: [batch, dec_T]
        """
        # 1) encode past weather
        h_enc = self.encode(x_enc)  # [B, H]

        B, T_dec, F = x_dec.shape

        # 2) repeat context for each future timestep
        h_rep = h_enc.unsqueeze(1).repeat(1, T_dec, 1)  # [B, T_dec, H]

        # 3) concatenate future weather with context
        dec_input = torch.cat([x_dec, h_rep], dim=-1)   # [B, T_dec, F+H]

        # 4) apply IMPROVED time-distributed MLP with batch norm
        z = self.decoder_fc1(dec_input)
        z = self.decoder_bn1(z)  # Batch norm
        z = self.leaky_relu(z)   # LeakyReLU for better gradients
        z = self.decoder_dropout1(z)

        z = self.decoder_fc2(z)
        z = self.decoder_bn2(z)
        z = self.leaky_relu(z)
        z = self.decoder_dropout2(z)

        z = self.decoder_fc3(z)  # NEW layer
        z = self.decoder_bn3(z)
        z = self.leaky_relu(z)
        z = self.decoder_dropout3(z)

        z = self.decoder_out(z)      # [B, T_dec, 1]
        y_pred = z.squeeze(-1)       # [B, T_dec]

        return y_pred

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
