import torch.nn as nn
import torch
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# ---- LSTM Model Constants ----
LSTM_UNITS = 64
HORIZON = 1
MIN_COVERAGE = 0.5
LOOKBACK = 12

MODEL_FILE = Path(__file__).parent / "models" / "lstm_model.pth"


class TrafficLSTM(nn.Module):
    def __init__(self, lstm_units: int = LSTM_UNITS, horizon: int = HORIZON):
        super().__init__()

        self.lstm1 = nn.LSTM(1, lstm_units, batch_first=True)
        self.drop1 = nn.Dropout(0.2)

        self.lstm2 = nn.LSTM(lstm_units, lstm_units // 2, batch_first=True)
        self.drop2 = nn.Dropout(0.2)

        self.fc = nn.Linear(lstm_units // 2, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x, _ = self.lstm1(x)
        x = self.drop1(x)

        x, _ = self.lstm2(x)

        x = self.drop2(x[:, -1, :])

        return self.fc(x)


def load_model(path: Path = MODEL_FILE):

    if not path.exists():
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TrafficLSTM().to(device)

    model.load_state_dict(
        torch.load(path, map_location=device, weights_only=True)
    )

    model.eval()

    return model, device