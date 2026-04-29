import os
import json
import argparse
import itertools

import torch
import numpy as np
import matplotlib.pyplot as plt

from model.input_fn  import get_data_loaders, TARGET_COL, LOOK_BACK
from model.model_fn  import build_model
from model.training  import train_step, LOSS_CHOICES
from model.evaluation import full_eval

CSV_PATH     = "wti_crude_daily.csv"
RESULTS_DIR  = os.path.join("experiments", "hyperparam_search")
os.makedirs(RESULTS_DIR, exist_ok=True)

PARAM_GRID = {
    "model_name":     ["LSTM", "GRU", "Transformer"],
    "lr":             [1e-3, 5e-4],
    "epochs":         [30, 50],
    "use_scheduler":  [False, True],
}


def grid_search(loss: str = "mse"):
    data = get_data_loaders(CSV_PATH)
    scaler      = data["scaler"]
    X_test_t    = data["X_test"]
    y_test_t    = data["y_test"]
    X_test_raw  = data["X_test_raw"]
    train_loader = data["train_loader"]

    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))

    loss_key    = f"test_{loss.lower()}"
    all_results = []
    best_sharpe = -float("inf")
    best_params = None

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        print(f"\n[{i+1}/{len(combos)}] {params}")

        model = build_model(params["model_name"])
        train_step(model, train_loader,
                   epochs=params["epochs"], lr=params["lr"],
                   loss=loss, verbose=False,
                   use_scheduler=params["use_scheduler"])

        metrics = full_eval(model, X_test_t, y_test_t, X_test_raw, scaler, TARGET_COL)
        row = {**params,
               loss_key:  round(metrics["mse"],      4),
               "dir_acc": round(metrics["dir_acc"],  4),
               "sharpe":  round(metrics["sharpe"],   4)}
        all_results.append(row)
        print(f"  {loss.upper()}={row[loss_key]}  Dir={row['dir_acc']*100:.1f}%  Sharpe={row['sharpe']}")

        if row["sharpe"] > best_sharpe:
            best_sharpe = row["sharpe"]
            best_params = params

    out_path = os.path.join(RESULTS_DIR, "search_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved → {out_path}")
    print(f"Best params (by Sharpe={best_sharpe:.4f}): {best_params}")
    _plot_results(all_results, loss)
    return all_results, best_params


def _plot_results(results: list, loss: str):
    """3-panel grouped bar chart: one group per (model, lr) combo, x-axis = epochs."""
    loss_key = f"test_{loss.lower()}"
    # Fall back to 'mae' or 'mse' key if the expected key is absent
    fallback_key = next((k for k in (loss_key, "mse", "mae") if k in results[0]), None)

    models = PARAM_GRID["model_name"]
    lrs    = PARAM_GRID["lr"]
    epochs = PARAM_GRID["epochs"]

    # label per combo
    combos = [(m, lr) for m in models for lr in lrs]
    combo_labels = [f"{m}\nlr={lr}" for m, lr in combos]
    x      = np.arange(len(combos))
    width  = 0.8 / len(epochs)
    offsets = np.linspace(-(len(epochs) - 1) / 2,
                           (len(epochs) - 1) / 2,
                           len(epochs)) * width

    def get_val(metric, model_name, lr, ep):
        for r in results:
            if r["model_name"] == model_name and r["lr"] == lr and r["epochs"] == ep:
                return r.get(metric, r.get(fallback_key, 0))
        return 0

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Hyperparameter Search Results", fontsize=14, fontweight="bold")

    metrics = [
        (axes[0], fallback_key, "Loss metric (lower is better)",         fallback_key.upper()),
        (axes[1], "dir_acc",    "Directional Accuracy (higher is better)", "Dir Acc"),
        (axes[2], "sharpe",     "Sharpe Ratio (higher is better)",         "Sharpe"),
    ]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, metric, title, ylabel in metrics:
        for j, (ep, color) in enumerate(zip(epochs, colors)):
            vals = [get_val(metric, m, lr, ep) for m, lr in combos]
            if metric == "dir_acc":
                vals = [v * 100 for v in vals]
            ax.bar(x + offsets[j], vals, width,
                   label=f"epochs={ep}", color=color, alpha=0.85, edgecolor="black")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(combo_labels, fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "hyperparam_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter grid search")
    parser.add_argument("--loss", default="mse", choices=LOSS_CHOICES,
                        help="Training loss function (default: mse)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip training; re-plot from existing results JSON")
    args = parser.parse_args()

    if args.plot_only:
        with open(os.path.join(RESULTS_DIR, "search_results.json")) as f:
            results = json.load(f)
        _plot_results(results, args.loss)
    else:
        grid_search(loss=args.loss)
