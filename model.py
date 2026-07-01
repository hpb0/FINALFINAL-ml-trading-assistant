"""CNN-LSTM binary direction model with Monte-Carlo dropout inference."""

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """Conv1d -> ReLU -> Dropout -> LSTM -> Dropout -> Linear(1) logit.

    Trained with BCEWithLogitsLoss + Adam. The single output is the logit
    for P(next day up).
    """

    def __init__(self, n_features, conv_channels=32, hidden_size=64, dropout=0.3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=n_features, out_channels=conv_channels,
            kernel_size=3, padding=1,
        )
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=conv_channels, hidden_size=hidden_size, batch_first=True,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = x.transpose(1, 2)        # -> (batch, n_features, seq_len) for Conv1d
        x = self.relu(self.conv(x))
        x = self.dropout1(x)
        x = x.transpose(1, 2)        # -> (batch, seq_len, conv_channels) for LSTM
        out, _ = self.lstm(x)
        last = out[:, -1, :]         # last time-step hidden state
        last = self.dropout2(last)
        return self.fc(last)         # (batch, 1) logit

    @torch.no_grad()
    def mc_predict(self, x, n_samples=50):
        """Monte-Carlo dropout inference.

        Keeps dropout ACTIVE (unlike normal eval) and runs n_samples
        forward passes. Returns (mean P(up), std), each shape (batch, 1);
        the std is an estimate of model uncertainty.
        """
        self.eval()
        for m in self.modules():     # re-enable dropout layers only
            if isinstance(m, nn.Dropout):
                m.train()
        probs = []
        for _ in range(n_samples):
            probs.append(torch.sigmoid(self.forward(x)))
        probs = torch.stack(probs)   # (n_samples, batch, 1)
        return probs.mean(0), probs.std(0)
