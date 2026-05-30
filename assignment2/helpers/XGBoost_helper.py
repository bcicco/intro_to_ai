import joblib
import pandas as pd
import numpy as np
from pathlib import Path


LOOKBACK = 12

MODEL_FILE = Path(__file__).parent.parent / "models" / "xgboost_model.pkl"


def load_xgb_models(path: Path = MODEL_FILE) -> dict | None:

  """Loadining the XGBoost models from the specified path. and return none if there is not file that exist in the path"""

  if not path.exists():
    return None

  return joblib.load(path)

def create_xbg_prediction_features(series: pd.Series, query_time: pd.Timestamp, lookback: int = LOOKBACK) -> pd.DataFrame | None:
  """Creating one prediction rows for the model"""

  idx = series.index.get_indexer([query_time], method="nearest")[0]

  window = series.iloc[max(0, idx - lookback + 1): idx + 1]

  if len(window) < lookback or window.isna().any():
    return None
  
  feature_data = {}

  for i in range(1, lookback + 1):
    feature_data[f"lag{i}"] = window.iloc[-i]


  feature_data["hour"] = query_time.hour
  feature_data["minute"] = query_time.minute
  feature_data["day_of_week"] = query_time.dayofweek
  feature_data["is_weekend"] = int(query_time.dayofweek >= 5)

  return pd.DataFrame([feature_data])

def predict_volumes_xgb(models: dict, wide_df: pd.DataFrame, query_time: pd.Timestamp, lookback: int = LOOKBACK) -> pd.DataFrame | None:
  """Predicting the volumes for the next 15 minutes using the XGBoost models"""

  predictions: dict[str, float] = {}

  query_time = pd.Timestamp(query_time)

  for col, model in models.items():
    if col not in wide_df.columns:
      continue

    series = wide_df[col].dropna()

    if len(series) < lookback:
      continue

    x_pred = create_xbg_prediction_features(series=series, query_time=query_time, lookback=lookback)

    if x_pred is None:
      continue

    pred = model.predict(x_pred)[0]

    predictions[col] = max(0.0, float(pred))

  return predictions