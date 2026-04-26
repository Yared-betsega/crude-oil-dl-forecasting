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
    def __init__(self, input_size=5, d_model=64, n_heads=4,
                 num_encoder_layers=2, dim_feedforward=128, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)


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
