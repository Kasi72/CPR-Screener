"""
lstm_model.py — LSTM architecture shared between training (train_phase3.py)
and inference (predict_server.py).

Keeping the model class here avoids predict_server importing from training code.
"""

import torch
import torch.nn as nn

HIDDEN_DIM = 64
DROPOUT    = 0.3


class LSTMSignalModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_layers=2, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(hidden_dim, 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out, _  = self.lstm(x)                            # [B, T, H]
        attn_w  = torch.softmax(self.attn(out), dim=1)    # [B, T, 1]
        ctx     = (attn_w * out).sum(dim=1)               # [B, H]
        return self.head(ctx).squeeze(-1)                  # [B]
