import torch.nn as nn
import torch
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# ---- GRU Model Constants ----
GRU_UNITS = 64
HORIZON = 1
MIN_COVERAGE = 0.5
LOOKBACK = 12

MODEL_FILE = Path(__file__).parent / "models" / "gru_model.pth"  # for load_model func.


class TrafficGRU(nn.Module):
    def __init__(self, gru_units: int = GRU_UNITS, horizon: int = HORIZON):
        super().__init__()
        self.gru1 = nn.GRU(1, gru_units, batch_first=True)
        self.drop1 = nn.Dropout(0.2)
        self.gru2 = nn.GRU(gru_units, gru_units // 2, batch_first=True)
        self.drop2 = nn.Dropout(0.2)
        self.fc = nn.Linear(gru_units // 2, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gru1(x)  # (B, 12, 64)
        x = self.drop1(x)
        x, _ = self.gru2(x)  # (B, 12, 32)
        x = self.drop2(x[:, -1, :])  # last timestep → (B, 32)
        return self.fc(x)  # (B, 1)


def load_model(path: Path = MODEL_FILE) -> tuple[TrafficGRU, torch.device] | None:
    """
    Load trained weights.  Returns None if the file doesn't exist. Tries to move to CUDA if available.
    """
    if not path.exists():
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrafficGRU().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model, device


def rebuild_scalers(wide_df: pd.DataFrame) -> dict[str, MinMaxScaler]:
    """
    Recreate the per-column MinMaxScalers fitted on the training split
    """
    scalers: dict[str, MinMaxScaler] = {}
    n_rows = len(wide_df)
    for col in wide_df.columns:
        series = wide_df[col].dropna().values.astype(np.float32)
        if len(series) / n_rows < MIN_COVERAGE:
            continue
        train_end = int(len(series) * 0.70)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(series[:train_end].reshape(-1, 1))
        scalers[col] = scaler
    return scalers


def predict_volumes(
    model: TrafficGRU,
    device: torch.device,
    scalers: dict[str, MinMaxScaler],
    wide_df: pd.DataFrame,
    query_time: pd.Timestamp,
) -> dict[str, float]:
    """
    For each approach with a scaler, take the LOOKBACK intervals ending at
    (or nearest to) query_time and predict the next 15-min volume.

    Returns {column_name: predicted_volume_vehicles_per_15min}.
    """
    idx = wide_df.index.get_indexer([query_time], method="nearest")[0]
    predictions: dict[str, float] = {}

    with torch.no_grad():
        for col, scaler in scalers.items():
            raw = wide_df[col].values.astype(np.float32)
            window = raw[max(0, idx - LOOKBACK + 1) : idx + 1]

            if len(window) < LOOKBACK or np.any(np.isnan(window)):
                continue

            scaled = scaler.transform(window.reshape(-1, 1)).flatten()
            x = (
                torch.FloatTensor(scaled)
                .unsqueeze(0)  # batch dim
                .unsqueeze(-1)  # feature dim → (1, 12, 1)
                .to(device)
            )

            pred = model(x).cpu().numpy()
            vol = float(scaler.inverse_transform(pred).flatten()[0])
            predictions[col] = max(0.0, vol)

    return predictions
