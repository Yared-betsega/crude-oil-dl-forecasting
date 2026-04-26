import os
import pickle

import torch
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from model.input_fn   import get_data_loaders, TARGET_COL, LOOK_BACK
from model.model_fn   import build_model, MODEL_REGISTRY
from model.evaluation import full_eval, print_metrics

CSV_PATH   = "wti_crude_daily.csv"
MODELS_DIR = "saved_models"


def load_dl_models(models_dir: str) -> dict:
    models = {}
    for name in MODEL_REGISTRY:
        path = os.path.join(models_dir, f"{name.lower()}_model.pt")
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        model = build_model(name)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        models[name] = model
        print(f"  Loaded {name} ← {path}")
    return models


def load_scaler(models_dir: str):
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


def arima_eval(data: dict) -> dict:
    scaled    = data["scaled"]
    split     = data["split"]
    scaler    = data["scaler"]

    close_scaled  = scaled[:, TARGET_COL]
    train_series  = close_scaled[:split + LOOK_BACK]
    test_series   = close_scaled[split + LOOK_BACK:]
    prev_arima    = np.concatenate([[train_series[-1]], test_series[:-1]])

    print("Running ARIMA walk-forward evaluation …")
    history     = list(train_series)
    arima_preds = []
    for t in range(len(test_series)):
        yhat = ARIMA(history, order=(5, 1, 0)).fit().forecast(steps=1)[0]
        arima_preds.append(yhat)
        history.append(test_series[t])

    arima_preds = np.array(arima_preds, dtype=np.float32)

    from sklearn.metrics import mean_absolute_error
    from model.utils import directional_accuracy, sharpe_ratio

    mae = mean_absolute_error(test_series, arima_preds)
    da  = directional_accuracy(arima_preds, test_series, prev_arima)
    sr  = sharpe_ratio(arima_preds, prev_arima, test_series, scaler)
    return {"mae": mae, "dir_acc": da, "sharpe": sr, "preds": arima_preds}


def main():
    print("Loading data …")
    data    = get_data_loaders(CSV_PATH)
    scaler  = load_scaler(MODELS_DIR)

    X_test_t   = data["X_test"]
    y_test_t   = data["y_test"]
    X_test_raw = data["X_test_raw"]

    print("\nLoading saved DL models …")
    models  = load_dl_models(MODELS_DIR)
    results = {}

    for name, model in models.items():
        metrics = full_eval(model, X_test_t, y_test_t, X_test_raw, scaler, TARGET_COL)
        results[name] = metrics

    results["ARIMA"] = arima_eval(data)

    print("\n\n=== Evaluation Results ===")
    print_metrics(results)


if __name__ == "__main__":
    main()
