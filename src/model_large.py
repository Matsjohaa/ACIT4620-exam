"""
Large PyTorch CNN-LSTM model (~1M parameters) for direct solar forecasting.
Designed to test if a much larger model can predict accurately without day-ahead.
"""

import torch
import torch.nn as nn
import numpy as np


class LargeEncoderDecoderCNNLSTM(nn.Module):
    """
    Large Encoder-Decoder CNN-LSTM with ~1M parameters for direct prediction.
    
    Architecture improvements:
    - Deeper CNN encoder (4 conv layers instead of 2)
    - Larger LSTM with 2 layers
    - Much deeper decoder (6 FC layers)
    - More hidden units throughout
    - Residual connections in decoder
    - Layer normalization
    
    Target: ~1,000,000 parameters
    """

    def __init__(
        self,
        enc_sequence_length: int = 168,
        dec_sequence_length: int = 336,
        n_features: int = 15,
        encoder_hidden: int = 256,      # 4x larger
        lstm_layers: int = 2,            # 2 layers
        decoder_hidden: int = 512,       # 4x larger
        dropout: float = 0.3,            # Increased from 0.2 to prevent overfitting
    ):
        super().__init__()

        self.enc_sequence_length = enc_sequence_length
        self.dec_sequence_length = dec_sequence_length
        self.n_features = n_features
        self.forecast_horizon = dec_sequence_length

        # --- DEEPER CNN Encoder ---
        # Conv block 1
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Conv block 2
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Conv block 3 (NEW)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        # Conv block 4 (NEW)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(512)

        # --- LARGER LSTM (2 layers) ---
        self.encoder_lstm = nn.LSTM(
            input_size=512,
            hidden_size=encoder_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        self.encoder_dropout = nn.Dropout(dropout)

        # --- MUCH DEEPER DECODER with RESIDUAL CONNECTIONS ---
        decoder_input_dim = n_features + encoder_hidden

        # Layer 1
        self.decoder_fc1 = nn.Linear(decoder_input_dim, decoder_hidden)
        self.decoder_ln1 = nn.LayerNorm(decoder_hidden)
        self.decoder_dropout1 = nn.Dropout(dropout)
        
        # Layer 2
        self.decoder_fc2 = nn.Linear(decoder_hidden, decoder_hidden)
        self.decoder_ln2 = nn.LayerNorm(decoder_hidden)
        self.decoder_dropout2 = nn.Dropout(dropout)
        
        # Layer 3
        self.decoder_fc3 = nn.Linear(decoder_hidden, decoder_hidden // 2)
        self.decoder_ln3 = nn.LayerNorm(decoder_hidden // 2)
        self.decoder_dropout3 = nn.Dropout(dropout)
        
        # Layer 4
        self.decoder_fc4 = nn.Linear(decoder_hidden // 2, decoder_hidden // 2)
        self.decoder_ln4 = nn.LayerNorm(decoder_hidden // 2)
        self.decoder_dropout4 = nn.Dropout(dropout)
        
        # Layer 5
        self.decoder_fc5 = nn.Linear(decoder_hidden // 2, decoder_hidden // 4)
        self.decoder_ln5 = nn.LayerNorm(decoder_hidden // 4)
        self.decoder_dropout5 = nn.Dropout(dropout)
        
        # Layer 6
        self.decoder_fc6 = nn.Linear(decoder_hidden // 4, decoder_hidden // 8)
        self.decoder_ln6 = nn.LayerNorm(decoder_hidden // 8)
        self.decoder_dropout6 = nn.Dropout(dropout)
        
        # Output layer
        self.decoder_out = nn.Linear(decoder_hidden // 8, 1)

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.gelu = nn.GELU()  # Better activation for deep networks

    def encode(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        Deep CNN + LSTM encoder
        x_enc: [batch, enc_T, n_features]
        returns h_enc: [batch, encoder_hidden]
        """
        # CNN expects [batch, channels, time]
        x = x_enc.transpose(1, 2)  # [B, F, T]
        
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)  # T/2

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)  # T/4
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)  # T/8
        
        # Conv block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        # Back to [batch, T', C] for LSTM
        x = x.transpose(1, 2)

        # 2-layer LSTM
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
        # 1) Encode past weather
        h_enc = self.encode(x_enc)  # [B, H]

        B, T_dec, F = x_dec.shape

        # 2) Repeat context for each future timestep
        h_rep = h_enc.unsqueeze(1).repeat(1, T_dec, 1)  # [B, T_dec, H]

        # 3) Concatenate future weather with context
        dec_input = torch.cat([x_dec, h_rep], dim=-1)   # [B, T_dec, F+H]

        # 4) Deep decoder with residual connections
        # Layer 1
        z = self.decoder_fc1(dec_input)
        z = self.decoder_ln1(z)
        z = self.gelu(z)
        z = self.decoder_dropout1(z)
        residual1 = z  # Save for skip connection

        # Layer 2 (with residual)
        z = self.decoder_fc2(z)
        z = self.decoder_ln2(z)
        z = self.gelu(z)
        z = self.decoder_dropout2(z)
        z = z + residual1  # Residual connection

        # Layer 3
        z = self.decoder_fc3(z)
        z = self.decoder_ln3(z)
        z = self.gelu(z)
        z = self.decoder_dropout3(z)
        residual2 = z

        # Layer 4 (with residual)
        z = self.decoder_fc4(z)
        z = self.decoder_ln4(z)
        z = self.gelu(z)
        z = self.decoder_dropout4(z)
        z = z + residual2  # Residual connection

        # Layer 5
        z = self.decoder_fc5(z)
        z = self.decoder_ln5(z)
        z = self.gelu(z)
        z = self.decoder_dropout5(z)

        # Layer 6
        z = self.decoder_fc6(z)
        z = self.decoder_ln6(z)
        z = self.gelu(z)
        z = self.decoder_dropout6(z)

        # Output
        z = self.decoder_out(z)      # [B, T_dec, 1]
        y_pred = z.squeeze(-1)       # [B, T_dec]

        return y_pred

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
