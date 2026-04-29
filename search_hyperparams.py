import os
import json
import argparse
import itertools

import torch

from model.input_fn  import get_data_loaders, TARGET_COL, LOOK_BACK
from model.model_fn  import build_model
from model.training  import train_step, LOSS_CHOICES
from model.evaluation import full_eval

CSV_PATH     = "wti_crude_daily.csv"
RESULTS_DIR  = os.path.join("experiments", "hyperparam_search")
os.makedirs(RESULTS_DIR, exist_ok=True)

PARAM_GRID = {
    "model_name": ["LSTM", "GRU", "Transformer"],
    "lr":         [1e-3, 5e-4],
    "epochs":     [30, 50],
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
                   loss=loss, verbose=False)

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
    return all_results, best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter grid search")
    parser.add_argument("--loss", default="mse", choices=LOSS_CHOICES,
                        help="Training loss function (default: mse)")
    args = parser.parse_args()
    grid_search(loss=args.loss)
