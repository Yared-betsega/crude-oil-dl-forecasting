"""
Experiment: compare look-back window sizes [5, 10, 20, 50] across all models.

For each window size the data loaders are rebuilt with that look_back value,
every model is trained from scratch, and test metrics are recorded.

Results are saved to:
    experiments/lookback_search/lookback_results.json

Usage:
    python search_lookback.py [--loss mse|huber|mae] [--epochs N]
"""

import os
import json
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

import model.input_fn as input_fn_module
from model.input_fn  import get_data_loaders, TARGET_COL
from model.model_fn  import build_model, MODEL_REGISTRY
from model.training  import train_step, LOSS_CHOICES
from model.evaluation import full_eval

CSV_PATH    = "wti_crude_daily.csv"
RESULTS_DIR = os.path.join("experiments", "lookback_search")
os.makedirs(RESULTS_DIR, exist_ok=True)

LOOKBACK_SIZES = [5, 10, 20, 50]


def run(loss: str = "mse", epochs: int = 50):
    all_results = []

    for lb in LOOKBACK_SIZES:
        print(f"\n{'='*55}")
        print(f"  LOOK_BACK = {lb}")
        print("=" * 55)

        # Patch the module-level constant so get_data_loaders picks it up
        input_fn_module.LOOK_BACK = lb

        data         = get_data_loaders(CSV_PATH)
        scaler       = data["scaler"]
        train_loader = data["train_loader"]
        X_test_t     = data["X_test"]
        y_test_t     = data["y_test"]
        X_test_raw   = data["X_test_raw"]

        for model_name in MODEL_REGISTRY:
            print(f"\n  [{model_name}]")
            model = build_model(model_name)
            train_step(model, train_loader,
                       epochs=epochs, loss=loss, verbose=False)

            metrics = full_eval(
                model, X_test_t, y_test_t, X_test_raw, scaler, TARGET_COL
            )

            row = {
                "look_back": lb,
                "model":     model_name,
                "mse":       round(metrics["mse"],      6),
                "dir_acc":   round(metrics["dir_acc"],  4),
                "sharpe":    round(metrics["sharpe"],   4),
            }
            all_results.append(row)
            print(f"    MSE={row['mse']:.6f}  Dir={row['dir_acc']*100:.1f}%  "
                  f"Sharpe={row['sharpe']:.4f}")

    out_path = os.path.join(RESULTS_DIR, "lookback_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # Print summary table
    print(f"\n{'look_back':<12} {'model':<13} {'MSE':<12} {'Dir%':<8} {'Sharpe'}")
    print("-" * 55)
    for row in all_results:
        print(f"{row['look_back']:<12} {row['model']:<13} "
              f"{row['mse']:<12.6f} {row['dir_acc']*100:<8.1f} {row['sharpe']:.4f}")

    _plot_results(all_results, RESULTS_DIR)
    return all_results


def _plot_results(results: list, out_dir: str):
    model_names = list(MODEL_REGISTRY.keys())
    lookbacks   = LOOKBACK_SIZES
    x           = np.arange(len(lookbacks))
    width       = 0.8 / len(model_names)
    offsets     = np.linspace(-(len(model_names) - 1) / 2,
                               (len(model_names) - 1) / 2,
                               len(model_names)) * width

    # Organise data: metrics[model][lb_idx]
    def gather(metric):
        return {
            m: [next(r[metric] for r in results
                     if r["model"] == m and r["look_back"] == lb)
                for lb in lookbacks]
            for m in model_names
        }

    mse_data     = gather("mse")
    dir_acc_data = gather("dir_acc")
    sharpe_data  = gather("sharpe")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Look-back Window Comparison", fontsize=14, fontweight="bold")

    specs = [
        (axes[0], mse_data,     "MSE (lower is better)",              "MSE"),
        (axes[1], dir_acc_data, "Directional Accuracy (higher is better)", "Dir Acc"),
        (axes[2], sharpe_data,  "Sharpe Ratio (higher is better)",    "Sharpe"),
    ]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, data, title, ylabel in specs:
        for i, (model_name, color) in enumerate(zip(model_names, colors)):
            bars = ax.bar(x + offsets[i], data[model_name], width,
                          label=model_name, color=color, alpha=0.85, edgecolor="black")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Look-back window (days)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([str(lb) for lb in lookbacks])
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "lookback_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Look-back window search")
    parser.add_argument("--loss",      default="mse", choices=LOSS_CHOICES,
                        help="Training loss function (default: mse)")
    parser.add_argument("--epochs",    type=int, default=50,
                        help="Epochs per model per window size (default: 50)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip training; re-plot from existing results JSON")
    args = parser.parse_args()

    if args.plot_only:
        results_path = os.path.join(RESULTS_DIR, "lookback_results.json")
        with open(results_path) as f:
            results = json.load(f)
        _plot_results(results, RESULTS_DIR)
    else:
        run(loss=args.loss, epochs=args.epochs)
