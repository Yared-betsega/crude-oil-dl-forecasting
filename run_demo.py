"""
run_demo.py — Quick end-to-end demo for grading.

Usage:
    python run_demo.py                # uses saved models if present, trains otherwise
    python run_demo.py --retrain      # forces a fresh training run
    python run_demo.py --sample       # uses the 100-row sample dataset

What this script does:
  1. Loads (or trains) all four deep-learning models: RNN, LSTM, GRU, Transformer.
  2. Runs an ARIMA(5,1,0) walk-forward baseline.
  3. Reports MSE, Directional Accuracy, and Annualised Sharpe Ratio for every model.
  4. Saves a comparison bar-chart to demo_results.png.
"""

import argparse
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

from model.input_fn   import get_data_loaders, TARGET_COL, LOOK_BACK
from model.model_fn   import build_model, MODEL_REGISTRY
from model.training   import train_step, LOSS_CHOICES
from model.evaluation import full_eval, print_metrics
from model.utils      import directional_accuracy, sharpe_ratio

MODELS_DIR = "saved_models"
SAMPLE_CSV = os.path.join("data", "test", "sample_test_data.csv")
FULL_CSV   = "wti_crude_daily.csv"


# ── helpers ───────────────────────────────────────────────────────────────────

def load_or_train(data: dict, retrain: bool, loss: str = "mse",
                  use_scheduler: bool = True) -> dict:
    """Return dict of {name: nn.Module}, loading from disk or training fresh."""
    models = {}
    scaler       = data["scaler"]
    train_loader = data["train_loader"]
    X_test_t     = data["X_test"]
    y_test_t     = data["y_test"]

    for name in MODEL_REGISTRY:
        path = os.path.join(MODELS_DIR, f"{name.lower()}_model.pt")
        model = build_model(name)
        if not retrain and os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            print(f"  Loaded  {name:12s} ← {path}")
        else:
            print(f"\n  Training {name} …")
            train_step(model, train_loader, loss=loss, use_scheduler=use_scheduler, verbose=True)
            os.makedirs(MODELS_DIR, exist_ok=True)
            torch.save(model.state_dict(), path)
            print(f"  Saved   {name:12s} → {path}")
        models[name] = model
    return models


def run_arima(data: dict) -> dict:
    scaled       = data["scaled"]
    split        = data["split"]
    scaler       = data["scaler"]
    close_scaled = scaled[:, TARGET_COL]
    train_series = close_scaled[:split + LOOK_BACK]
    test_series  = close_scaled[split + LOOK_BACK:]
    prev_arima   = np.concatenate([[train_series[-1]], test_series[:-1]])

    print("\n  Running ARIMA(5,1,0) walk-forward …")
    history     = list(train_series)
    preds       = []
    for t in range(len(test_series)):
        yhat = ARIMA(history, order=(5, 1, 0)).fit().forecast(steps=1)[0]
        preds.append(yhat)
        history.append(test_series[t])
    preds = np.array(preds, dtype=np.float32)

    mse = mean_squared_error(test_series, preds)
    da  = directional_accuracy(preds, test_series, prev_arima)
    sr  = sharpe_ratio(preds, prev_arima, test_series, scaler)
    return {"mse": mse, "dir_acc": da, "sharpe": sr, "preds": preds}


def save_chart(results: dict):
    model_names = list(results.keys())
    mse_vals    = [results[n]["mse"]      for n in model_names]
    da_vals     = [results[n]["dir_acc"] * 100 for n in model_names]
    sr_vals     = [results[n]["sharpe"]   for n in model_names]
    colors      = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"][:len(model_names)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, vals, title, ylabel in zip(
        axes,
        [mse_vals, da_vals, sr_vals],
        ["MSE (lower is better)", "Dir. Accuracy % (higher is better)", "Sharpe (higher is better)"],
        ["MSE (scaled Close)", "Directional Accuracy (%)", "Annualised Sharpe Ratio"],
    ):
        bars = ax.bar(model_names, vals, color=colors, alpha=0.85, edgecolor="black")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (max(vals) * 0.01),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)

    axes[1].axhline(50, color="gray", linestyle="--", linewidth=1, label="random (50%)")
    axes[1].legend(fontsize=8)
    axes[2].axhline(0, color="gray", linestyle="--", linewidth=1)

    fig.suptitle("WTI Crude Oil Futures — Model Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("demo_results.png", dpi=150)
    print("\nChart saved → demo_results.png")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Demo evaluation script")
    parser.add_argument("--retrain", action="store_true",
                        help="Force retraining even if saved models exist")
    parser.add_argument("--sample",  action="store_true",
                        help="Use the 100-row sample dataset instead of the full CSV")
    parser.add_argument("--loss", default="mse", choices=LOSS_CHOICES,
                        help="Training loss function (default: mse)")
    parser.add_argument("--no-scheduler", action="store_true",
                        help="Disable ReduceLROnPlateau (use fixed learning rate)")
    args = parser.parse_args()

    csv_path = SAMPLE_CSV if args.sample else FULL_CSV
    if not os.path.exists(csv_path):
        print(f"Dataset not found: {csv_path}")
        print("Run  python build_dataset.py  to fetch it from Yahoo Finance.")
        return

    if args.sample:
        print("NOTE: --sample uses 100 rows (smoke-test only).")
        print("      Metrics on this tiny slice are not meaningful.")
        print("      Run without --sample for proper evaluation.\n")

    print(f"\nDataset : {csv_path}")
    print("Loading & preprocessing …")
    data = get_data_loaders(csv_path)

    print("\n── Deep-learning models ─────────────────────────────────────────")
    models  = load_or_train(data, retrain=args.retrain, loss=args.loss,
                            use_scheduler=not args.no_scheduler)

    results = {}
    for name, model in models.items():
        metrics = full_eval(
            model, data["X_test"], data["y_test"],
            data["X_test_raw"], data["scaler"], TARGET_COL,
        )
        results[name] = metrics

    print("\n── ARIMA baseline ───────────────────────────────────────────────")
    results["ARIMA"] = run_arima(data)

    print("\n\n══════════════════════════════════════════════════════════════")
    print("  FINAL RESULTS")
    print("══════════════════════════════════════════════════════════════")
    print_metrics(results)

    save_chart(results)


if __name__ == "__main__":
    main()
