"""
PyTorch implementation of CNN-LSTM model for solar forecasting.
Much more stable on Apple Silicon than TensorFlow!
"""

import torch
import torch.nn as nn
import numpy as np


class CNNLSTM(nn.Module):
    """
    CNN-LSTM hybrid model for solar forecasting.
    
    Architecture:
    - CNN layers extract spatial patterns from weather features
    - LSTM layers capture temporal dependencies
    - Dense layers produce 14-day forecast
    """
    
    def __init__(self, 
                 sequence_length=168,
                 n_features=14,
                 forecast_horizon=336,
                 cnn_filters=[64, 128],
                 lstm_units=[128, 64],
                 dropout=0.2):
        super(CNNLSTM, self).__init__()
        
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.forecast_horizon = forecast_horizon
        
        # CNN layers (1D convolutions over time)
        self.conv1 = nn.Conv1d(n_features, cnn_filters[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_filters[0])
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(cnn_filters[0], cnn_filters[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_filters[1])
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Calculate sequence length after pooling
        pooled_length = sequence_length // 4  # Two pooling layers of size 2
        
        # LSTM layers
        self.lstm1 = nn.LSTM(cnn_filters[1], lstm_units[0], batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(lstm_units[0], lstm_units[1], batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        
        # Dense layers
        self.fc1 = nn.Linear(lstm_units[1], 256)
        self.dropout3 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, forecast_horizon)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, sequence_length, n_features)
            
        Returns:
            Output tensor (batch_size, forecast_horizon)
        """
        # Reshape for CNN: (batch, features, time)
        x = x.transpose(1, 2)
        
        # CNN layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        # Reshape for LSTM: (batch, time, features)
        x = x.transpose(1, 2)
        
        # LSTM layers
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        
        # Use last LSTM output
        x = x[:, -1, :]
        
        # Dense layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleCNNLSTM(nn.Module):
    """
    Enhanced CNN-LSTM model optimized for capturing day-to-day weather variations.
    Reduced regularization to allow more variation in predictions.
    """
    
    def __init__(self, 
                 sequence_length=168,
                 n_features=14,
                 forecast_horizon=336,
                 dropout=0.1):  # Much lower dropout - allow variation!
        super(SimpleCNNLSTM, self).__init__()
        
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.forecast_horizon = forecast_horizon
        
        # Enhanced CNN layers - more filters to capture weather patterns
        self.conv1 = nn.Conv1d(n_features, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Enhanced LSTM - 2 layers for better temporal learning
        self.lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        
        # Larger dense layers
        self.fc1 = nn.Linear(128, 256)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(128, forecast_horizon)
        
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        # Reshape for CNN: (batch, features, time)
        x = x.transpose(1, 2)
        
        # CNN layers - extract spatial weather patterns
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.pool2(x)
        
        # Reshape for LSTM: (batch, time, features)
        x = x.transpose(1, 2)
        
        # LSTM - capture temporal dependencies
        x, _ = self.lstm(x)
        x = self.dropout1(x)
        
        # Use last LSTM output
        x = x[:, -1, :]
        
        # Dense layers - map to forecast horizon
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout2(x)
        
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.dropout3(x)
        
        x = self.fc3(x)
        
        return x
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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
    Encoder–decoder CNN-LSTM model.

    Encoder:
        - CNN + LSTM over past weather (168h) -> context vector h_enc.

    Decoder:
        - For each future hour, take weather features + h_enc
        - Pass through a small MLP to predict capacity_factor for that hour.
    """

    def __init__(
        self,
        enc_sequence_length: int = 168,
        dec_sequence_length: int = 336,
        n_features: int = 15,
        encoder_hidden: int = 64,
        decoder_hidden: int = 128,
        dropout: float = 0.1,
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

        # --- Decoder: MLP over future weather + context ---
        # For each future timestep we will concat [weather_t, h_enc]
        decoder_input_dim = n_features + encoder_hidden

        self.decoder_fc1 = nn.Linear(decoder_input_dim, decoder_hidden)
        self.decoder_dropout1 = nn.Dropout(dropout)
        self.decoder_fc2 = nn.Linear(decoder_hidden, decoder_hidden // 2)
        self.decoder_dropout2 = nn.Dropout(dropout)
        self.decoder_out = nn.Linear(decoder_hidden // 2, 1)

        self.relu = nn.ReLU()

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

        # 4) apply time-distributed MLP
        z = self.decoder_fc1(dec_input)
        z = self.relu(z)
        z = self.decoder_dropout1(z)

        z = self.decoder_fc2(z)
        z = self.relu(z)
        z = self.decoder_dropout2(z)

        z = self.decoder_out(z)      # [B, T_dec, 1]
        y_pred = z.squeeze(-1)       # [B, T_dec]

        return y_pred

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
