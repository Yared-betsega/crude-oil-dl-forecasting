import math
import torch
import torch.nn as nn

INPUT_SIZE  = 5
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.2


class VanillaRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
                          batch_first=True, dropout=DROPOUT)
        self.bn  = nn.BatchNorm1d(HIDDEN_SIZE)
        self.fc  = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.bn(out[:, -1, :])
        return self.fc(out)


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
                            batch_first=True, dropout=DROPOUT)
        self.bn   = nn.BatchNorm1d(HIDDEN_SIZE)
        self.fc   = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.bn(out[:, -1, :])
        return self.fc(out)


class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
                          batch_first=True, dropout=DROPOUT)
        self.bn  = nn.BatchNorm1d(HIDDEN_SIZE)
        self.fc  = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.bn(out[:, -1, :])
        return self.fc(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerForecaster(nn.Module):
    """Decoder-only Transformer (GPT-style).

    Architecture:
      Input → Linear embedding → Sinusoidal PE
        → N × [Masked Multi-Head Self-Attention → Add & Norm
               → Feed-Forward Network          → Add & Norm]
        → last-token readout → Dense → scalar prediction

    A zero-filled prediction slot is appended to the input sequence so
    the model produces the forecast at the final (T+1-th) position while
    maintaining the causal / autoregressive property throughout.
    """

    def __init__(self, input_size=INPUT_SIZE, d_model=64, n_heads=8,
                 num_layers=2, dim_feedforward=128, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder_stack = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # Append zero-filled prediction slot: (B, T) -> (B, T+1)
        pred_slot = torch.zeros(x.size(0), 1, x.size(2), device=x.device, dtype=x.dtype)
        x = torch.cat([x, pred_slot], dim=1)            # (B, T+1, input_size)

        x = self.input_proj(x)                          # (B, T+1, d_model)
        x = self.pos_enc(x)

        # Causal mask: each position may only attend to itself and prior positions
        T = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)

        x = self.decoder_stack(x, mask=causal_mask)     # (B, T+1, d_model)
        return self.fc(x[:, -1, :])                     # forecast from last token


MODEL_REGISTRY = {
    "RNN":         VanillaRNN,
    "LSTM":        LSTMModel,
    "GRU":         GRUModel,
    "Transformer": TransformerForecaster,
}


def build_model(name: str) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]()
