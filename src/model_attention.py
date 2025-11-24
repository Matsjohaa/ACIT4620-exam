"""
Encoder-Decoder CNN-LSTM model with Attention mechanism for solar forecasting.
Attention allows decoder to focus on relevant parts of encoder history.
"""

import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    """
    Bahdanau-style attention mechanism.
    
    For each decoder timestep, compute attention weights over all encoder timesteps,
    then produce a context vector as weighted sum of encoder outputs.
    """
    
    def __init__(self, encoder_hidden: int):
        super().__init__()
        
        # Simplified attention scoring function
        self.attention = nn.Sequential(
            nn.Linear(encoder_hidden * 2, encoder_hidden),
            nn.Tanh(),
            nn.Linear(encoder_hidden, 1)
        )
        
    def forward(self, encoder_outputs: torch.Tensor, query: torch.Tensor) -> tuple:
        """
        Args:
            encoder_outputs: [batch, enc_seq_len, encoder_hidden]
            query: [batch, encoder_hidden] - query vector (e.g., mean of encoder outputs)
            
        Returns:
            context: [batch, encoder_hidden] - weighted sum of encoder outputs
            attention_weights: [batch, enc_seq_len] - attention distribution
        """
        batch_size = encoder_outputs.size(0)
        enc_seq_len = encoder_outputs.size(1)
        
        # Expand query to match encoder sequence length
        # [batch, encoder_hidden] -> [batch, enc_seq_len, encoder_hidden]
        query_expanded = query.unsqueeze(1).repeat(1, enc_seq_len, 1)
        
        # Concatenate encoder outputs with query
        # [batch, enc_seq_len, encoder_hidden * 2]
        combined = torch.cat([encoder_outputs, query_expanded], dim=2)
        
        # Compute attention scores
        # [batch, enc_seq_len, 1]
        scores = self.attention(combined)
        
        # Apply softmax to get attention weights
        # [batch, enc_seq_len]
        attention_weights = torch.softmax(scores.squeeze(2), dim=1)
        
        # Compute context as weighted sum of encoder outputs
        # [batch, encoder_hidden]
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch, 1, enc_seq_len]
            encoder_outputs                   # [batch, enc_seq_len, encoder_hidden]
        ).squeeze(1)
        
        return context, attention_weights


class EncoderDecoderAttentionCNNLSTM(nn.Module):
    """
    Enhanced Encoder-Decoder with Attention mechanism.
    
    Key improvement over base model:
    - Decoder can attend to ALL encoder timesteps, not just final hidden state
    - For each forecast hour, attention computes which historical hours matter most
    - Should help model understand temporal dependencies (e.g., yesterday's pattern)
    
    Architecture:
    1. Encoder: CNN + LSTM over past 168h -> sequence of hidden states
    2. Attention: For each forecast hour, compute attention over encoder states
    3. Decoder: MLP using [future_weather, attention_context] -> capacity_factor
    """
    
    def __init__(
        self,
        enc_sequence_length: int = 168,
        dec_sequence_length: int = 336,
        n_features: int = 19,  # Updated for temporal features
        encoder_hidden: int = 128,
        decoder_hidden: int = 256,
        dropout: float = 0.15,
    ):
        super().__init__()
        
        self.enc_sequence_length = enc_sequence_length
        self.dec_sequence_length = dec_sequence_length
        self.n_features = n_features
        self.forecast_horizon = dec_sequence_length
        
        # --- Encoder: CNN + LSTM ---
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # LSTM returns ALL hidden states (not just last)
        self.encoder_lstm = nn.LSTM(
            input_size=128,
            hidden_size=encoder_hidden,
            batch_first=True,
            bidirectional=False,  # Keep simple to avoid overfitting
        )
        
        self.encoder_dropout = nn.Dropout(dropout)
        
        # --- Attention Mechanism ---
        self.attention = AttentionLayer(encoder_hidden)
        
        # --- Decoder: MLP with attention context ---
        # Input: [future_weather, attention_context]
        decoder_input_dim = n_features + encoder_hidden
        
        self.decoder_fc1 = nn.Linear(decoder_input_dim, decoder_hidden)
        self.decoder_bn1 = nn.BatchNorm1d(dec_sequence_length)
        self.decoder_dropout1 = nn.Dropout(dropout)
        
        self.decoder_fc2 = nn.Linear(decoder_hidden, decoder_hidden // 2)
        self.decoder_bn2 = nn.BatchNorm1d(dec_sequence_length)
        self.decoder_dropout2 = nn.Dropout(dropout)
        
        self.decoder_fc3 = nn.Linear(decoder_hidden // 2, decoder_hidden // 4)
        self.decoder_bn3 = nn.BatchNorm1d(dec_sequence_length)
        self.decoder_dropout3 = nn.Dropout(dropout)
        
        self.decoder_out = nn.Linear(decoder_hidden // 4, 1)
        
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def encode(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        Encode past weather sequence.
        
        Args:
            x_enc: [batch, enc_T, n_features]
            
        Returns:
            encoder_outputs: [batch, enc_T//4, encoder_hidden] - ALL LSTM hidden states
        """
        # CNN: [batch, features, time]
        x = x_enc.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        # LSTM: [batch, time, features]
        x = x.transpose(1, 2)
        encoder_outputs, (h_n, c_n) = self.encoder_lstm(x)
        encoder_outputs = self.encoder_dropout(encoder_outputs)
        
        # Return ALL hidden states for attention
        return encoder_outputs
    
    def decode_with_attention(
        self, 
        x_dec: torch.Tensor, 
        encoder_outputs: torch.Tensor
    ) -> tuple:
        """
        Decode future sequence using attention over encoder outputs.
        
        Args:
            x_dec: [batch, dec_T, n_features] - future weather
            encoder_outputs: [batch, enc_T, encoder_hidden]
            
        Returns:
            predictions: [batch, dec_T] - capacity factor predictions
            attention_weights: [batch, dec_T, enc_T] - attention distributions
        """
        batch_size = x_dec.size(0)
        dec_T = x_dec.size(1)
        
        # We'll use a simple decoder state (mean of encoder outputs)
        # More complex: use LSTM decoder, but risks overfitting
        decoder_state = encoder_outputs.mean(dim=1)  # [batch, encoder_hidden]
        
        # Process each decoder timestep
        outputs = []
        attention_weights_list = []
        
        for t in range(dec_T):
            # Get weather features for this timestep
            weather_t = x_dec[:, t, :]  # [batch, n_features]
            
            # Compute attention context
            context, attn_weights = self.attention(encoder_outputs, decoder_state)
            
            # Combine weather + context
            decoder_input = torch.cat([weather_t, context], dim=1)
            
            # MLP decoder
            h = self.decoder_fc1(decoder_input)
            h = self.relu(h)
            
            h = self.decoder_fc2(h)
            h = self.relu(h)
            
            h = self.decoder_fc3(h)
            h = self.leaky_relu(h)
            
            out = self.decoder_out(h)  # [batch, 1]
            
            outputs.append(out)
            attention_weights_list.append(attn_weights)
        
        # Stack outputs
        predictions = torch.cat(outputs, dim=1)  # [batch, dec_T]
        attention_weights = torch.stack(attention_weights_list, dim=1)  # [batch, dec_T, enc_T]
        
        return predictions, attention_weights
    
    def forward(self, x_enc: torch.Tensor, x_dec: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x_enc: [batch, enc_T, n_features] - past weather
            x_dec: [batch, dec_T, n_features] - future weather
            
        Returns:
            predictions: [batch, dec_T] - capacity factor predictions
        """
        encoder_outputs = self.encode(x_enc)
        predictions, _ = self.decode_with_attention(x_dec, encoder_outputs)
        return predictions
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_attention_model(
    enc_sequence_length: int = 168,
    dec_sequence_length: int = 336,
    n_features: int = 19,
    encoder_hidden: int = 128,
    decoder_hidden: int = 256,
    dropout: float = 0.15,
) -> EncoderDecoderAttentionCNNLSTM:
    """
    Factory function to create attention model.
    """
    return EncoderDecoderAttentionCNNLSTM(
        enc_sequence_length=enc_sequence_length,
        dec_sequence_length=dec_sequence_length,
        n_features=n_features,
        encoder_hidden=encoder_hidden,
        decoder_hidden=decoder_hidden,
        dropout=dropout,
    )


if __name__ == "__main__":
    # Test model
    model = get_attention_model(n_features=19)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Test forward pass
    batch_size = 4
    x_enc = torch.randn(batch_size, 168, 19)
    x_dec = torch.randn(batch_size, 336, 19)
    
    output = model(x_enc, x_dec)
    print(f"Output shape: {output.shape}")  # Should be [4, 336]
    
    # Test attention weights
    encoder_outputs = model.encode(x_enc)
    predictions, attention_weights = model.decode_with_attention(x_dec, encoder_outputs)
    print(f"Attention weights shape: {attention_weights.shape}")  # Should be [4, 336, 42]
