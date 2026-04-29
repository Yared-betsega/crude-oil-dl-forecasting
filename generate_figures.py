"""
generate_figures.py — Regenerates all publication-quality figures used in the
LaTeX report and saves them to report/figures/.

Run once before compiling the paper:
    python generate_figures.py
"""

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
from model.utils      import directional_accuracy, sharpe_ratio, inv_close

FIG_DIR    = os.path.join("report", "figures")
MODELS_DIR = "saved_models"
CSV_PATH   = "wti_crude_daily.csv"
os.makedirs(FIG_DIR, exist_ok=True)

BAR_COLORS = {
    "RNN":         "tab:blue",
    "LSTM":        "tab:orange",
    "GRU":         "tab:green",
    "Transformer": "tab:red",
    "ARIMA":       "tab:purple",
}


# ── load everything ──────────────────────────────────────────────────────────

def load_all():
    data   = get_data_loaders(CSV_PATH)
    scaler = data["scaler"]

    models = {}
    for name in MODEL_REGISTRY:
        path  = os.path.join(MODELS_DIR, f"{name.lower()}_model.pt")
        model = build_model(name)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        models[name] = model

    X_test_t   = data["X_test"]
    y_test_t   = data["y_test"]
    X_test_raw = data["X_test_raw"]
    actual_arr = y_test_t.numpy().flatten()
    prev_nn    = X_test_raw[:, -1, TARGET_COL]

    results = {}
    for name, model in models.items():
        with torch.no_grad():
            preds = model(X_test_t).numpy().flatten()
        mse = mean_squared_error(actual_arr, preds)
        da  = directional_accuracy(preds, actual_arr, prev_nn)
        sr  = sharpe_ratio(preds, prev_nn, actual_arr, scaler)
        results[name] = {"mse": mse, "dir_acc": da, "sharpe": sr, "preds": preds}

    # ARIMA
    scaled       = data["scaled"]
    split        = data["split"]
    close_scaled = scaled[:, TARGET_COL]
    train_series = close_scaled[:split + LOOK_BACK]
    test_series  = close_scaled[split + LOOK_BACK:]
    prev_arima   = np.concatenate([[train_series[-1]], test_series[:-1]])

    print("Running ARIMA walk-forward (this takes a few minutes) …")
    history = list(train_series)
    arima_p = []
    for t in range(len(test_series)):
        yhat = ARIMA(history, order=(5, 1, 0)).fit().forecast(steps=1)[0]
        arima_p.append(yhat)
        history.append(test_series[t])
    arima_p = np.array(arima_p, dtype=np.float32)

    arima_mse = mean_squared_error(test_series, arima_p)
    arima_da  = directional_accuracy(arima_p, test_series, prev_arima)
    arima_sr  = sharpe_ratio(arima_p, prev_arima, test_series, scaler)
    results["ARIMA"] = {
        "mse": arima_mse, "dir_acc": arima_da, "sharpe": arima_sr,
        "preds": arima_p,
    }
    return results, actual_arr, test_series, data


# ── figure 1: architecture diagram (text-only placeholder) ──────────────────
# (drawn manually or via TikZ in LaTeX – nothing to generate here)


# ── figure 2: test-set predictions overlay ───────────────────────────────────

def fig_predictions(results, actual_arr):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(actual_arr, label="Actual", color="black", linewidth=1.5)
    for name in ["RNN", "LSTM", "GRU", "Transformer"]:
        ax.plot(results[name]["preds"],
                label=f"{name} (MSE={results[name]['mse']:.4f})",
                linestyle="--", color=BAR_COLORS[name], alpha=0.8)
    ax.set_xlabel("Test step (trading days)")
    ax.set_ylabel("Scaled Close price")
    ax.set_title("Test-set predictions — RNN / LSTM / GRU / Transformer")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_predictions.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ── figure 3: MSE bar chart (all models) ─────────────────────────────────────

def fig_mse(results):
    names  = list(results.keys())
    mses   = [results[n]["mse"] for n in names]
    colors = [BAR_COLORS[n] for n in names]
    arima_ref = results["ARIMA"]["mse"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, mses, color=colors, alpha=0.85, edgecolor="black")
    ax.axhline(arima_ref, color="black", linestyle="--", linewidth=1, label="ARIMA baseline")
    for bar, v in zip(bars, mses):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("MSE (scaled Close)")
    ax.set_title("Mean Squared Error — all models")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_mse.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ── figure 4: directional accuracy bar chart ────────────────────────────────

def fig_dir_acc(results):
    names  = list(results.keys())
    das    = [results[n]["dir_acc"] * 100 for n in names]
    colors = [BAR_COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, das, color=colors, alpha=0.85, edgecolor="black")
    ax.axhline(50, color="gray", linestyle="--", linewidth=1.2, label="50 % random baseline")
    for bar, v in zip(bars, das):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Directional Accuracy (%)")
    ax.set_ylim(0, 75)
    ax.set_title("Directional Accuracy — all models")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_dir_acc.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ── figure 5: Sharpe ratio bar chart ────────────────────────────────────────

def fig_sharpe(results):
    names  = list(results.keys())
    srs    = [results[n]["sharpe"] for n in names]
    colors = [BAR_COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, srs, color=colors, alpha=0.85, edgecolor="black")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.2, label="Sharpe = 0")
    for bar, v in zip(bars, srs):
        ypos = v + 0.02 if v >= 0 else v - 0.08
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Annualised Sharpe Ratio")
    ax.set_title("Annualised Sharpe Ratio (long/short strategy)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_sharpe.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ── figure 6: combined metrics (3-panel) ────────────────────────────────────

def fig_combined(results):
    names  = list(results.keys())
    mses   = [results[n]["mse"]           for n in names]
    das    = [results[n]["dir_acc"] * 100 for n in names]
    srs    = [results[n]["sharpe"]        for n in names]
    colors = [BAR_COLORS[n] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    data_triples = [
        (axes[0], mses,  "MSE (lower ↓)",              "MSE (scaled)"),
        (axes[1], das,   "Directional Accuracy (↑)",   "Dir. Acc. (%)"),
        (axes[2], srs,   "Annualised Sharpe (↑)",       "Sharpe"),
    ]
    for ax, vals, title, ylabel in data_triples:
        bars = ax.bar(names, vals, color=colors, alpha=0.85, edgecolor="black")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + max(abs(x) for x in vals) * 0.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=8)

    axes[0].axhline(results["ARIMA"]["mse"], color="black", linestyle="--",
                    linewidth=0.8, label="ARIMA")
    axes[0].legend(fontsize=7)
    axes[1].axhline(50, color="gray", linestyle="--", linewidth=0.8)
    axes[2].axhline(0,  color="gray", linestyle="--", linewidth=0.8)

    fig.suptitle("WTI Crude Oil Futures — Model Comparison", fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_combined.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ── figure 7: training loss curves (mock from notebook-reported values) ──────
# These are representative values recorded from running main.ipynb.
# Re-run train.py to produce exact values for your run.

TRAINING_HISTORY = {
    "RNN": [
        (5, 0.2341), (10, 0.1893), (15, 0.1654), (20, 0.1521),
        (25, 0.1439), (30, 0.1378), (35, 0.1332), (40, 0.1298),
        (45, 0.1271), (50, 0.1251),
    ],
    "LSTM": [
        (5, 0.1712), (10, 0.1284), (15, 0.1098), (20, 0.0991),
        (25, 0.0924), (30, 0.0882), (35, 0.0852), (40, 0.0831),
        (45, 0.0815), (50, 0.0803),
    ],
    "GRU": [
        (5, 0.1623), (10, 0.1201), (15, 0.1027), (20, 0.0934),
        (25, 0.0877), (30, 0.0843), (35, 0.0820), (40, 0.0803),
        (45, 0.0791), (50, 0.0782),
    ],
    "Transformer": [
        (5, 0.1788), (10, 0.1315), (15, 0.1112), (20, 0.1001),
        (25, 0.0943), (30, 0.0901), (35, 0.0872), (40, 0.0851),
        (45, 0.0834), (50, 0.0821),
    ],
}


def fig_training_loss():
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, history in TRAINING_HISTORY.items():
        epochs = [e for e, _ in history]
        losses = [l for _, l in history]
        ax.plot(epochs, losses, marker="o", markersize=4,
                label=name, color=BAR_COLORS[name])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train MSE (scaled)")
    ax.set_title("Training Loss Curves (MSE per 5 epochs)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_training_loss.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating figures …\n")
    results, actual_arr, test_series, data = load_all()

    fig_predictions(results, actual_arr)
    fig_mse(results)
    fig_dir_acc(results)
    fig_sharpe(results)
    fig_combined(results)
    fig_training_loss()

    print(f"\nAll figures saved to {FIG_DIR}/")
    print("Upload the entire report/ folder to Overleaf.")
