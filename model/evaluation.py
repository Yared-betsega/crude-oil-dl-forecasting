import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error

from model.utils import directional_accuracy, sharpe_ratio


def evaluate_model(model: nn.Module, X_t: torch.Tensor,
                   y_t: torch.Tensor):
    model.eval()
    with torch.no_grad():
        preds = model(X_t).numpy().flatten()
    actual = y_t.numpy().flatten()
    mae    = mean_absolute_error(actual, preds)
    return mae, preds, actual


def full_eval(model: nn.Module, X_t: torch.Tensor, y_t: torch.Tensor,
              X_raw: np.ndarray, scaler, target_col: int = 3):
    mae, preds, actual = evaluate_model(model, X_t, y_t)

    prev = X_raw[:, -1, target_col]
    da   = directional_accuracy(preds, actual, prev)
    sr   = sharpe_ratio(preds, prev, actual, scaler)

    return {
        "mae":    mae,
        "dir_acc": da,
        "sharpe": sr,
        "preds":  preds,
        "actual": actual,
    }


def print_metrics(results: dict):
    header = f"{'Model':<14} {'MAE':>10}  {'Dir Acc %':>10}  {'Sharpe':>10}"
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for name, r in results.items():
        da_str = f"{r['dir_acc']*100:.1f}%" if "dir_acc" in r else "  —"
        sr_str = f"{r['sharpe']:.3f}"        if "sharpe"  in r else "  —"
        print(f"{name:<14} {r['mae']:>10.4f}  {da_str:>10}  {sr_str:>10}")
    print(sep)
